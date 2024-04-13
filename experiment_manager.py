import logging
import random
from typing import List, Dict, Tuple

import os
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import distributed as dist
import torch

from tasks.dataset_loader import ClassDS, GenerationDS, LABEL_TOKENS, TEXT_BETWEEN_SHOTS_CLASS, TEXT_BETWEEN_SHOTS_CLASS_GLM, TEXT_BETWEEN_SHOTS, N_TOKENS, PROMPTS
from model.llama_model import  LlamaLong, LlamaTokenizer, RestrictiveTokensLogitsProcessor
from transformers import PreTrainedTokenizer
from configure_data import make_eval_data_loader
from utils import init_distributed, exact_match, f1_score, bleu, rouge, Timer

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

N_WINDOWS = 3

def encode_labels(tokenizer, labels: List[str]) -> List[List[int]]:
    if isinstance(tokenizer, LlamaTokenizer):
        # sentence piece - adds a space at the beginning of the sentence
        return [tokenizer.encode(f'{label.lstrip()}', add_special_tokens=False) for label in labels]
    return [tokenizer.encode(f' {label.lstrip()}', add_special_tokens=False) for label in labels]

class ExperimentManager:
    def __init__(self, test_df: pd.DataFrame, train_df: pd.DataFrame, model, tokenizer, 
                 labels: List[str] = None, random_seed: int = 42, subsample_test_set: int = 250,
                 n_shots_per_window: int = None, batch_size_per_gpu: int = 1, ensemble: bool = False):
        if subsample_test_set < len(test_df):
            np.random.seed(random_seed)
            test_df = test_df.sample(subsample_test_set)
        self.test_df = test_df
        self.train_df = train_df
        self.batch_test_ds = None
        self.batch_size_per_gpu = batch_size_per_gpu
        self.model = model
        self.base_random_seed = random_seed
        self.n_shots_per_window = n_shots_per_window
        self.tokenizer = tokenizer
        self.ensemble = ensemble
        self.divide_text = None
        self._initialize_labels_and_logit_processor(labels)

    def _initialize_labels_and_logit_processor(self, labels: List[str]) -> None:
        labels_tokens = encode_labels(self.tokenizer, labels)
        # labels_tokens = [self.tokenizer.encode(f'{label.lstrip()}', add_special_tokens=False) for label in labels] # no space??

        labels_tokens_array = self.minimize_labels_tokens(labels_tokens)
        _logger.info(f"Shortened labels average n_tokens: {np.round(np.mean([len(lt) for lt in labels_tokens]), 3)}")
        # we fix the labels accordingly in the test set:
        shorten_label_tokens = [t[t != self.tokenizer.eos_token_id].tolist() for t in labels_tokens_array]
        _logger.info(
            f"shortened labels average n_tokens: {np.round(np.mean([len(lt) for lt in shorten_label_tokens]), 3)}")
        # Moving the test set label tokens to their shorter version:
        map_labels = {old_label: self.tokenizer.decode(t).lstrip() for old_label, t in
                      zip(labels, shorten_label_tokens)}
        self.text_labels = map_labels.values()
        _logger.info(f"Shortened labels: {self.text_labels}")
        self.test_df[LABEL_TOKENS] = self.test_df[LABEL_TOKENS].map(map_labels)
        pad = len(max(shorten_label_tokens, key=len))
        labels_tokens_array = np.array(
            [i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in shorten_label_tokens])
        self.max_n_tokens = pad
        labels_tokens_array = self.pad_contained_labels_with_stop_seq(shorten_label_tokens, labels_tokens_array)
        # [labels_num, padded_label_tokens]
        if isinstance(self.model, LlamaLong):
            self.divide_text = TEXT_BETWEEN_SHOTS_CLASS
            self.logit_processor = RestrictiveTokensLogitsProcessor(restrictive_token_ids=labels_tokens_array,
                                                                    eos_token_id=self.tokenizer.eos_token_id)

    def minimize_labels_tokens(self, labels_tokens: List[List[int]]) -> npt.NDArray:
        """
        Minimize the number of tokens per label to be the shortest possible unique one.
        """
        pad = len(max(labels_tokens, key=len))
        labels_tokens_array = np.array([i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in labels_tokens])
        for i, tokens in enumerate(labels_tokens):
            for j in range(len(tokens)):
                labels_with_shared_beginnings = np.sum(
                    np.all(labels_tokens_array[:, :j] == np.array(tokens[:j]), axis=1))
                if labels_with_shared_beginnings == 1:
                    labels_tokens_array[i, j:] = self.tokenizer.eos_token_id
                    break
        return labels_tokens_array

    def pad_contained_labels_with_stop_seq(self, labels_tokens: List, labels_tokens_array: npt.NDArray) \
            -> npt.NDArray:
        """
        In case we have two labels, where one label contains the other label (for example: "A" and "A B") we need
        to allow the restrictive decoding to produce the output "A". We support it by adding "\n" to the shorter label.
        """
        STOP_SEQUENCE = '\n'
        stop_seq_token_id = self.tokenizer.encode(STOP_SEQUENCE, add_special_tokens=False)
        try:
            assert len(stop_seq_token_id) == 1
        except AssertionError:
            stop_seq_token_id = stop_seq_token_id[-1:]
        stop_seq_token_id = stop_seq_token_id[0]
        for i, tokens in enumerate(labels_tokens):
            labels_with_shared_beginnings = np.sum(
                np.all(labels_tokens_array[:, :len(tokens)] == np.array(tokens), axis=1))
            if labels_with_shared_beginnings > 1:
                _logger.info(f"label{self.tokenizer.decode(tokens)} is the beginning of one of the other labels,"
                             f"adding stop sequence to its end")
                labels_tokens_array[i, len(tokens)] = stop_seq_token_id
        return labels_tokens_array

    def _set_random_seed(self, random_seed: int) -> None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    def get_few_shots_acc(self, windows_few_shot: List[str]) -> float:
        predicted_labels, res = self.get_predicted_labels(windows_few_shot)
        predicted_labels = pd.Series(predicted_labels, index=self.test_df.index)
        predicted_labels.to_csv('output/pred.csv', index=False)
        return self.calc_acc(res)

    def get_predicted_labels(self, windows_few_shots: List[str]) -> List[str]:
        if not self.ensemble:
            windows_cache = self.model.get_contexts_cache(windows_few_shots, n_batch=self.batch_size_per_gpu)
        else:
            assert self.batch_size_per_gpu == 1
            windows_cache = self.model.get_ensmeble_cache(windows_few_shots)
        self.batch_test_ds = ClassDS(self.test_df)
        dataloader = make_eval_data_loader(self.batch_test_ds, batch_size=self.batch_size_per_gpu, distributed=False)
        predicted_labels, res = [], []
        for texts, labels in dataloader:
            if len(texts) != self.batch_size_per_gpu:
                windows_cache = self.model.get_contexts_cache(windows_few_shots, n_batch=len(texts))
            predicted_label = self.predict_label(texts, windows_cache)
            predicted_labels += predicted_label
            for pred, truth in zip(predicted_label, labels):
                assert pred in self.text_labels
                res += [1 if pred == truth else 0 ]
        assert len(res) == len(self.test_df)
        return predicted_labels, res
    
    def predict_label(self, task_text, cache: Dict) -> List[str]:
        with torch.no_grad():
            if not self.ensemble:
                res = self.model.pcw_batch_generate(task_text=task_text, 
                                                    contexts_cache=cache,
                                                    restrictive_logit_preprocessor=self.logit_processor,
                                                    temperature=0, 
                                                    max_new_tokens=self.max_n_tokens)
            else:
                res = self.model.pcw_ensemble_generate(task_text=task_text, 
                                                    contexts_cache=cache,
                                                    restrictive_logit_preprocessor=self.logit_processor,
                                                    temperature=0, 
                                                    max_new_tokens=self.max_n_tokens,
                                                    n_windows=N_WINDOWS)   
        return res

    def calc_acc(self, res: List) -> float:
        acc = np.sum(res) / len(res)
        _logger.info(f"accuracy = {np.round(acc, 3)}")
        return acc

    def run_experiment_across_shots(self, n_shots_to_test: List[int], n_runs: int,
                                    too_long_patience: float = 0.2):
        timer = Timer()
        accuracies = np.zeros((len(n_shots_to_test), n_runs))
        assert len(n_shots_to_test) == len(self.n_shots_per_window)
        for i, n_shots in enumerate(tqdm(n_shots_to_test)):
            _logger.info(f"starting with n = {n_shots}")
            self._set_random_seed(self.base_random_seed + n_shots)
            j = 0
            n_errors = 0
            n_shots_per_window = self.n_shots_per_window[i]
            while j < n_runs:
                timer.start()
                few_shots_idx = self.sample_n_shots(n_shots, n_shots_per_window)
                few_shots_prompts = list(self.train_df.loc[few_shots_idx, PROMPTS])
                windows_few_shots = self.build_windows_few_shots_text(self.divide_text, few_shots_prompts, n_shots_per_window)
                longest_window_n_tokens = max(n_tokens_in_prompt(self.tokenizer, window)
                                              for window in windows_few_shots)
                n_tokens_between_shots = n_tokens_in_prompt(self.tokenizer, self.divide_text)
                if (longest_window_n_tokens + n_tokens_between_shots +
                        self.test_df[N_TOKENS].max() + self.max_n_tokens) > self.model.context_window_size:
                    _logger.warning("Drawn training shots were too long, trying again")
                    n_errors += 1
                    assert n_errors <= too_long_patience * n_runs, "too many long inputs were drawn!"
                    continue
                accuracies[i, j] = self.get_few_shots_acc(windows_few_shots)
                j += 1
                from utils import get_gpu_memory_usage
                get_gpu_memory_usage()
                timer.stop('Per run #')

        return accuracies

    def sample_n_shots(self, n_shots: int, n_shots_per_window: int) -> npt.NDArray:
        few_shots_df = self.train_df.sample(n_shots)
        assert few_shots_df.index.is_unique, "few shots samples were not unique!"
        n_windows = int(len(few_shots_df) / n_shots_per_window)
        if n_windows == 1:
            return few_shots_df.index
        return self.balance_windows_sizes(n_windows, n_shots, few_shots_df)

    def balance_windows_sizes(self, n_windows: int, n_shots: int, few_shots_df: pd.DataFrame) -> npt.NDArray:
        few_shots_df.sort_values(by=N_TOKENS, inplace=True, ascending=False)
        n_shots_per_window = int(n_shots / n_windows)
        shape = (n_shots_per_window, n_windows)
        indexes = np.array(few_shots_df.index).reshape(shape)
        sizes = few_shots_df.loc[indexes.flatten()].n_tokens.values.reshape(indexes.shape)
        for i in range(1, n_shots_per_window):
            order = np.argsort((np.sum(sizes[:i, :], axis=0)))
            sizes[i, :] = sizes[i, order]
            indexes[i, :] = indexes[i, order]
        # shuffle the order in each window:
        for i in range(n_windows):
            np.random.shuffle(indexes[:, i])
        indexes = indexes.T.flatten()
        return indexes

    @staticmethod
    def build_windows_few_shots_text(divide_text: str, few_shots_prompts: List, window_size: int) -> List[str]:
        if window_size is None:
            window_size = len(few_shots_prompts)
        return [divide_text.join(few_shots_prompts[i: i + window_size]) for i in
                range(0, len(few_shots_prompts), window_size)]


class GenManager:
    def __init__(self, test_df: pd.DataFrame, example_df: pd.DataFrame, model, tokenizer, 
                 instruction: str = '', random_seed: int = 42, subsample_test_set: int = 250, 
                 n_shots_per_window: int = None, batch_size_per_gpu: int = 1, max_new_tokens: int = 10, save_path: str = ''):
        if subsample_test_set < len(test_df):
            np.random.seed(random_seed)
            test_df = test_df.sample(subsample_test_set)
        self.test_df = test_df
        self.example_df = example_df
        self.batch_test_ds = GenerationDS(self.test_df, instruction)
        self.batch_size_per_gpu = batch_size_per_gpu
        self.model = model
        self.base_random_seed = random_seed
        self.n_shots_per_window = n_shots_per_window
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.save_path = save_path
        self.eval_func = ['rouge']

    def _set_random_seed(self, random_seed: int) -> None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    def eval_generation(self, windows_few_shot) -> float:
        comps, score = self.get_generation_completions(windows_few_shot)
        score = [np.mean(value) for _, value in score.items()]
        _logger.info(f"score = {np.round(score, 3)}")
        ## save predictions and scores
        self.save_predictions(comps, self.save_path)
        return score
    
    def get_eval_score(self, pred, truth, metric):
        if metric == 'em':
            return exact_match(pred, truth)
        elif metric == 'f1':
            return f1_score(pred, truth)[0]
        elif metric == 'rouge':
            return rouge(pred, truth)
        elif metric == 'bleu':
            return bleu(pred, truth)
        else:
            raise NotImplementedError

    def get_generation_completions(self, windows_few_shots: List[str]) -> List[str]:
        windows_cache = self.model.get_contexts_cache(windows_few_shots, n_batch=self.batch_size_per_gpu)
        comps = []
        metric = {func: [] for func in self.eval_func}
        dataloader = make_eval_data_loader(self.batch_test_ds, batch_size=self.batch_size_per_gpu, distributed=False)
        for texts, labels in dataloader:
            if len(texts) != self.batch_size_per_gpu:
                windows_cache = self.model.get_contexts_cache(windows_few_shots, n_batch=len(texts))
            preds = self.predict_completion(texts, windows_cache)
            for pred, truth in zip(preds, labels):
                comps += [pred]
                for func in self.eval_func:
                    if func == 'rouge' and len(pred) <= 0:
                        metric[func] += [0.]
                    else:
                        metric[func] += [self.get_eval_score(pred, truth, func)]
        return comps, metric
    
    def predict_completion(self, task_text:List[str], cache: Dict) -> List[str]:
        with torch.no_grad():
            eos_token_id = self.tokenizer.encode('\n')[-1]
            res = self.model.pcw_batch_generate(task_text=task_text, 
                                                contexts_cache=cache,
                                                generation=True,
                                                eos_token_id=eos_token_id,
                                                temperature=0,
                                                max_new_tokens=self.max_new_tokens)
        return res
    

    def run_experiment_across_shots(self, n_shots_to_test: List[int], n_runs: int,
                                    too_long_patience: float = 0.2):
        timer = Timer()
        scores = np.zeros((len(n_shots_to_test), n_runs, len(self.eval_func)))
        for i, n_shots in enumerate(tqdm(n_shots_to_test)):
            _logger.info(f"starting with n = {n_shots}")
            self._set_random_seed(self.base_random_seed + n_shots)
            j = 0
            n_errors = 0
            n_shots_per_window = self.n_shots_per_window[i]
            while j < n_runs:
                timer.start()
                if self.example_df is not None:
                    few_shots_idx = self.sample_n_shots(n_shots, n_shots_per_window)
                    few_shots_prompts = list(self.example_df.loc[few_shots_idx, PROMPTS])
                    windows_few_shots = self.build_windows_few_shots_text(few_shots_prompts, n_shots_per_window)
                    longest_window_n_tokens = max(n_tokens_in_prompt(self.tokenizer, window)
                                                for window in windows_few_shots)
                    n_tokens_between_shots = n_tokens_in_prompt(self.tokenizer, TEXT_BETWEEN_SHOTS)
                    if (longest_window_n_tokens + n_tokens_between_shots +
                            self.test_df[N_TOKENS].max() + self.max_new_tokens) > self.model.context_window_size:
                        _logger.warning("Drawn training shots were too long, trying again")
                        n_errors += 1
                        assert n_errors <= too_long_patience * n_runs, "too many long inputs were drawn!"
                        continue
                else:
                    windows_few_shots = ['']
                scores[i, j] = self.eval_generation(windows_few_shots)
                j += 1
                from utils import get_gpu_memory_usage
                get_gpu_memory_usage()
                timer.stop('Per run #')
        np.save(self.save_path + '.npy', scores)
        return scores

    @staticmethod
    def build_windows_few_shots_text(few_shots_prompts: List, window_size: int) -> List[str]:
        if window_size is None:
            window_size = len(few_shots_prompts)
        return [TEXT_BETWEEN_SHOTS.join(few_shots_prompts[i: i + window_size]) for i in
                range(0, len(few_shots_prompts), window_size)]

    def sample_n_shots(self, n_shots: int, n_shots_per_window: int) -> npt.NDArray:
        few_shots_df = self.example_df.sample(n_shots)
        assert few_shots_df.index.is_unique, "few shots samples were not unique!"
        n_windows = int(len(few_shots_df) / n_shots_per_window)
        if n_windows == 1:
            return few_shots_df.index
        return self.balance_windows_sizes(n_windows, n_shots, few_shots_df)

    def balance_windows_sizes(self, n_windows: int, n_shots: int, few_shots_df: pd.DataFrame) -> npt.NDArray:
        few_shots_df.sort_values(by=N_TOKENS, inplace=True, ascending=False)
        n_shots_per_window = int(n_shots / n_windows)
        shape = (n_shots_per_window, n_windows)
        indexes = np.array(few_shots_df.index).reshape(shape)
        sizes = few_shots_df.loc[indexes.flatten()].n_tokens.values.reshape(indexes.shape)
        for i in range(1, n_shots_per_window):
            order = np.argsort((np.sum(sizes[:i, :], axis=0)))
            sizes[i, :] = sizes[i, order]
            indexes[i, :] = indexes[i, order]
        # shuffle the order in each window:
        for i in range(n_windows):
            np.random.shuffle(indexes[:, i])
        indexes = indexes.T.flatten()
        return indexes

    def save_predictions(self, preds, save_path):
        pred_df = self.test_df.copy()
        pred_df['pred'] = preds
        pred_df.to_csv(save_path + '.csv', index=False)


class CoTManager(GenManager):
    def __init__(self, test_df: pd.DataFrame, example_df: pd.DataFrame, model, tokenizer, 
                 instruction: str = '', random_seed: int = 42, subsample_test_set: int = 250, y_prefix = '',
                 n_shots_per_window: int = None, batch_size_per_gpu: int = 1, max_new_tokens: int = 10, save_path: str = ''):
        super().__init__(test_df, example_df, model, tokenizer, instruction=instruction, random_seed=random_seed, save_path=save_path, max_new_tokens=max_new_tokens,
                        n_shots_per_window=n_shots_per_window, subsample_test_set=subsample_test_set, batch_size_per_gpu=batch_size_per_gpu)
        self.y_prefix = y_prefix
        self.eval_func = ['em', 'f1']

    def predict_completion(self, task_text:List[str], cache: Dict, max_new_tokens = None) -> List[str]:
        with torch.no_grad():
            eos_token_id = self.tokenizer.encode('\n')[-1]
            res = self.model.pcw_batch_generate(task_text=task_text, 
                                                contexts_cache=cache,
                                                generation=True,
                                                eos_token_id=eos_token_id,
                                                temperature=0,
                                                max_new_tokens=max_new_tokens or self.max_new_tokens)
        return res

    def get_generation_completions(self, windows_few_shots: List[str]) -> List[str]:
        windows_cache = self.model.get_contexts_cache(windows_few_shots, n_batch=self.batch_size_per_gpu)
        comps, answers = [], []
        metric = {func: [] for func in self.eval_func}
        dataloader = make_eval_data_loader(self.batch_test_ds, batch_size=self.batch_size_per_gpu, distributed=False)
        for texts, labels in dataloader:
            n_batch = len(texts)
            ## Two steps!
            ## Generate Thought
            if n_batch != self.batch_size_per_gpu:
                windows_cache = self.model.get_contexts_cache(windows_few_shots, n_batch)
            preds = self.predict_completion(texts, windows_cache)
            ## Generate Answer, add \n bacuse we delete it as eos_token in generate
            thoughts = [text + pred + '\n' + self.y_prefix.rstrip() for text, pred in zip(texts, preds)]
            preds = self.predict_completion(thoughts, windows_cache, max_new_tokens=10)
            for pred, thought, truth in zip(preds, thoughts, labels):
                comps += [thought.lstrip(TEXT_BETWEEN_SHOTS) + pred]
                answers += [pred]
                for func in self.eval_func:
                    metric[func] += [self.get_eval_score(pred, truth, func)]
        return comps, answers, metric

    def eval_generation(self, windows_few_shot) -> float:
        comps, answers, score = self.get_generation_completions(windows_few_shot)
        score = [np.mean(value) for _, value in score.items()]
        _logger.info(f"score = {np.round(score, 3)}")
        ## save predictions and scores
        np.save(self.save_path + '.npy', score)
        self.save_predictions(comps, answers, self.save_path)
        return score

    def save_predictions(self, comps, answers, save_path):
        pred_df = self.test_df.copy()
        pred_df['pred'] = answers
        pred_df['thought'] = comps
        pred_df.to_csv(save_path + '.csv', index=False)


def get_max_n_shots(train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer: PreTrainedTokenizer,
                    prompt_size: int) -> int:
    n_tokens_between_shots = n_tokens_in_prompt(tokenizer, TEXT_BETWEEN_SHOTS_CLASS)
    shot_lengths = train_df[N_TOKENS] + n_tokens_between_shots
    prompt_length_percentile = shot_lengths.quantile(0.9)
    longest_test_prompt = test_df[N_TOKENS].max()
    _logger.info(f"longest_test_prompt = {longest_test_prompt}")
    max_possible_shots_length = prompt_size - longest_test_prompt
    return int(np.floor(max_possible_shots_length / prompt_length_percentile))

def filter_extremely_long_samples(df: pd.DataFrame, tokenizer: PreTrainedTokenizer, quantile = 0.99, no_filter: bool=False) -> pd.DataFrame:
    df[N_TOKENS] = df[PROMPTS].map(lambda x: n_tokens_in_prompt(tokenizer, x))
    if no_filter:
        _logger.info(f"Prompt total length according to tokenizer: {df[N_TOKENS].sum()}")        
        return df
    mask = df[N_TOKENS] <= df[N_TOKENS].quantile(quantile)
    _logger.info(f"filtered {sum(~mask)} from dataset due to extreme length")
    df = df.loc[mask].copy()
    df = df.reset_index(drop=True)
    _logger.info(f"longest remaining prompt according to tokenizer: {df[N_TOKENS].max()}")
    return df

def n_tokens_in_prompt(tokenizer: PreTrainedTokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt, add_special_tokens=False))

def load_results(dataset_name: str, output_dir: str, model: str, wbits=16, generation=False, plot=False) -> Tuple[npt.NDArray, List[int]]:
    output_dir = os.path.join(output_dir, model, str(wbits))
    all_results = os.listdir(output_dir)
    results_path = [r for r in all_results if r.startswith(f'{dataset_name}_') and r.endswith('.npy')]
    for res_path in sorted(results_path, key=lambda x: int(''.join(filter(str.isdigit, x)))): # to int
        mode = res_path.split('.')[-2].split('_')[-1]
        results = np.load(os.path.join(output_dir, res_path))
        n_shots = [int(d) for d in res_path.split('.')[-2].split('_') if d.isdigit()]
        if generation:
            print(f'{n_shots[0]} shots')
            print(mode)
            print(res_path.split('.')[-2].split('_')[-2])
            if 'hotpot' in res_path:
                if len(results.shape) > 1:
                    results = results[0].T
                    print('em: ', np.mean(results[0]), np.std(results[0]))
                    print('f1: ', np.mean(results[1]), np.std(results[1]))
            else:
                print('rouge-l: ', np.mean(results[0]), np.std(results[0]))
            preds = pd.read_csv(os.path.join(output_dir,res_path.replace('npy', 'csv')))
            print(len(preds))  
        else:   
            for i, acc in enumerate(results):
                print(f'{n_shots[i]} shots')
                print(mode)
                print(np.mean(acc), np.std(acc))
        if plot:
            plot_results_graph(results, dataset_name, n_shots)
        print(results)
        print('-'*30)

def save_results(dataset: str, n_shots: List[int], results: npt.NDArray, output_dir: str, parallel: str, 
                 model: str = '', wbits=16, plot_results: bool = True) -> None:
    model = model.split('/')[-2]
    if plot_results:
        plot_results_graph(results, dataset, n_shots, model)
        plt.show()
    if not dist.is_initialized() or dist.get_rank() == 0:
        # in case we use multiple GPUs - we only save one file
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/{model}", exist_ok=True)
        os.makedirs(f"{output_dir}/{model}/{wbits}", exist_ok=True)
        output_path = f"{output_dir}/{model}/{wbits}/{dataset}_results_{'_'.join([str(i) for i in n_shots])}_{parallel}.npy"
        np.save(output_path, results)

def parse_save_path(output_dir, model, wbits):
    model = model.split('/')[-2]
    if not dist.is_initialized() or dist.get_rank() == 0:
        # in case we use multiple GPUs - we only save one file
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/{model}", exist_ok=True)
        os.makedirs(f"{output_dir}/{model}/{wbits}", exist_ok=True)
    return f"{output_dir}/{model}/{wbits}"

def plot_results_graph(results, dataset_name, n_shots, model='') -> None:
    plt.figure()
    plt.errorbar(n_shots, np.mean(results, axis=1), np.std(results, axis=1), fmt='*')
    plt.xlabel("# shots")
    plt.xticks(n_shots)
    metric = 'Accuracy'
    plt.ylabel(f"{dataset_name} {metric}")
    plt.title(f"{metric} {dataset_name} {model}")

