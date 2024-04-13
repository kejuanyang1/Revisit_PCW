import argparse
import logging
from typing import List
import os

import pandas as pd
import torch
from transformers import PreTrainedTokenizer, AutoConfig, LlamaTokenizer, LlamaForCausalLM

from tasks.dataset_loader import DATASET_NAMES2LOADERS, N_TOKENS
from experiment_manager import ExperimentManager, CoTManager, get_max_n_shots, filter_extremely_long_samples, save_results, parse_save_path
from model.llama_model import LlamaLong
from utils import Timer, get_device_map


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_inference_wrapper_and_tokenizer(model_path: str, wbits: int, checkpoint_path: str = None):
    timer = Timer()
    timer.start()
    if 'llama' in model_path or 'vicuna' in model_path:
        config = AutoConfig.from_pretrained(model_path, max_sequence_length = 6144)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        if wbits == 16:
            model = LlamaLong.from_pretrained(model_path, config=config, tokenizer=tokenizer, torch_dtype=torch.float16, device_map='auto',
                                            max_memory=get_device_map())
            model.eval()
        # quantization
        elif wbits == 8:
            model = LlamaLong.from_pretrained(model_path, config=config, tokenizer=tokenizer, torch_dtype=torch.float16, 
                                            load_in_8bit=True, device_map='auto',
                                            max_memory=get_device_map())
        else:
            raise NotImplementedError
    else: 
        raise NotImplementedError
    timer.stop('LM load time: ')
    return model, tokenizer

def get_dataset(dataset: str, tokenizer: PreTrainedTokenizer, no_filter_trainset: bool = False, quantile: float = 0.99):
    da = DATASET_NAMES2LOADERS[dataset]()
    # Filter extremely long samples from both train and test samples:
    _logger.info("filtering test set:")
    test_df = filter_extremely_long_samples(da.test_df, tokenizer, quantile)
    if no_filter_trainset:
        example_df = filter_extremely_long_samples(da.example_df, tokenizer, no_filter=True)
        return test_df, example_df, da.instruction, da.y_prefix
    _logger.info("filtering train set:")
    if hasattr(da, 'train_df'):
        train_df = filter_extremely_long_samples(da.train_df, tokenizer)
        return test_df, train_df, da.labels
    else:
        train_df = filter_extremely_long_samples(da.example_df, tokenizer, quantile)
        return test_df, train_df

def run_pcw_class(dataset: str, model: str, subsample_test_set: int, output_dir: str,
                       n_shots: List[int], n_runs: int, random_seed: int, 
                       parallel: str, wbits: int, batch_size_per_gpu: int, **kargs) -> None:
    
    # pcw parameter, n_windows
    phi = 3
    model_name = model.split('/')[-2]
    print(model_name)
    modes = {'seq': 4, 'window': 1, 'ensemble': 1}
    parallel_modes = {k: v for k, v in modes.items()}

    pcw_model, tokenizer = get_inference_wrapper_and_tokenizer(model, wbits)

    # labels: dict_values
    test_df, train_df, labels = get_dataset(dataset, tokenizer)
    max_shots_for_seq = get_max_n_shots(train_df, test_df, tokenizer, pcw_model.context_window_size)
    _logger.info(f"Found max n shot for sequential = {max_shots_for_seq}")

    for parallel in parallel_modes.keys():
        batch_size_per_gpu = parallel_modes[parallel]
        print(parallel)
        if parallel == 'parallel':
            n_shots_per_window = [1]
            n_shots = [max_shots_for_seq * phi]
        elif parallel == 'window' or parallel == 'ensemble':
            n_shots_per_window = [max_shots_for_seq]
            n_shots = [max_shots_for_seq * phi]
        else:
            n_shots = [max_shots_for_seq]
            n_shots_per_window = n_shots
        print(n_shots_per_window, n_shots)
        # n_shots: exp setting
        em = ExperimentManager(test_df, train_df, pcw_model, tokenizer, labels, random_seed=random_seed, ensemble=(parallel=='ensemble'),
                            n_shots_per_window=n_shots_per_window, subsample_test_set=subsample_test_set, batch_size_per_gpu=batch_size_per_gpu)

        accuracies = em.run_experiment_across_shots(n_shots, n_runs)
        save_results(dataset, n_shots, accuracies, output_dir, parallel, model, wbits)


def run_pcw_cot(dataset: str, model: str, subsample_test_set: int, output_dir: str,
                       n_shots: List[int], n_runs: int, random_seed: int, 
                       parallel: str, wbits: int, batch_size_per_gpu: int, **kargs) -> None:
    pcw_model, tokenizer = get_inference_wrapper_and_tokenizer(model, wbits)
    test_df, example_df, instruction, y_prefix = get_dataset(dataset, tokenizer, no_filter_trainset=True)

    if kargs['shots_num_per_window'] > 0:
        span = kargs['shots_num_per_window']
    else:
        span = 6
    assert len(example_df) == 18
    if parallel == 'parallel':
        n_shots_per_window = [1]
    elif parallel == 'window':
        n_shots_per_window = [span]
    else:
        n_shots_per_window = n_shots
    print(n_shots, n_shots_per_window)

    save_path = os.path.join(parse_save_path(output_dir, model, wbits), f"{dataset}_results_{'_'.join([str(i) for i in n_shots])}_{parallel}_{span}")
    print(example_df)
    rm = CoTManager(test_df, example_df, pcw_model, tokenizer, random_seed=random_seed, save_path=save_path,
                        max_new_tokens=80, instruction=instruction, y_prefix=y_prefix, 
                        n_shots_per_window=n_shots_per_window, subsample_test_set=subsample_test_set, batch_size_per_gpu=batch_size_per_gpu)
    
    scores = rm.run_experiment_across_shots(n_shots, n_runs)
    print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', required=True,
                        help=f'Name of dataset (for example sst2).'
                             f' The supported datasets are: {DATASET_NAMES2LOADERS.keys()}')
    parser.add_argument('--model', dest='model', 
                        help='HF model path')
    parser.add_argument('--subsample-test-set', dest='subsample_test_set', required=False, type=int,
                        help='Size of test set to use to speed up eval. None means using all test set.')
    parser.add_argument('--output-dir', dest='output_dir', required=False, help="Directory for saving the results",
                        default='./temp', type=str)
    parser.add_argument('--random-seed', dest='random_seed', required=False, default=42, type=int)
    parser.add_argument('--n-runs', dest='n_runs', help="Number of times experiments are repeated for every number of windows", action='store',
                        type=int, default=1)
    parser.add_argument('--n-shots', dest='n_shots', help="Number of parallel context windows",
                        action='append', type=int)
    parser.add_argument('--parallel', help="Parallel mode, choose from [seq, window, ensemble]", 
                        default='seq', type=str)
    parser.add_argument('--wbits', type=int, default=16, choices=[8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--checkpoint', dest='checkpoint', default=None,
                        help='.pt checkpoint path')
    parser.add_argument('--batch-size-per-gpu', dest='batch_size_per_gpu', 
                        type=int, default=1)
    parser.add_argument('--shots-num-per-window', dest='shots_num_per_window', 
                        type=int, default=-1)
    parser.add_argument('--generation', dest='generation', action='store_true')                    
    args = parser.parse_args()
    if args.generation:
        run_pcw_cot(**vars(args))
    else:
        run_pcw_class(**vars(args))