from typing import List, Tuple, Optional, Dict
from abc import ABC
import math

import numpy as np
import numpy.typing as npt
import torch
from transformers import GPT2LMHeadModel, LogitsProcessor, PreTrainedTokenizerBase, GPT2Tokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaRMSNorm, \
    LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from model.llama_causal import LlamaForCausalLMPCW

LLM_WINDOW_SIZE = 2048
LOGIT_BIAS = 100


# concat past keys and values of n windows
def combine_past_key_values(past_lst: List[Tuple[Tuple[torch.Tensor]]],
                            n_batch: int,
                            contains_bos_token: bool = True) -> Tuple[Tuple[torch.Tensor]]:
    if contains_bos_token:
        # We eliminate all but one bos token from windows to avoid multiple bos, which deterred our results.
        n_layers = len(past_lst[0])
        first_window = past_lst[0]
        return tuple(
            (torch.cat([first_window[i][0]] + [c[i][0][:, :, 1:, :] for c in past_lst[1:]], dim=2).expand(n_batch, -1, -1, -1),
             torch.cat([first_window[i][1]] + [c[i][1][:, :, 1:, :] for c in past_lst[1:]], dim=2).expand(n_batch, -1, -1, -1))
            for i in range(n_layers))
    return tuple(
        (torch.cat([c[i][0] for c in past_lst], dim=2).expand(n_batch, -1, -1, -1), 
         torch.cat([c[i][1] for c in past_lst], dim=2).expand(n_batch, -1, -1, -1))
        for i in range(len(past_lst[0])))

def longest_combine_past_key_values(past_lst: List[Tuple[Tuple[torch.Tensor]]], longest_window_id: int, n_batch: int = 1) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    # We eliminate all but one bos token from windows to avoid multiple bos, which deterred our results.
    n_layers = len(past_lst[0])
    longest_window = past_lst[longest_window_id]
    all_windows_except_longest = past_lst[:longest_window_id] + past_lst[longest_window_id + 1:]
    return tuple(
        (torch.cat([longest_window[i][0]] + [c[i][0][:, :, 1:, :] for c in all_windows_except_longest], dim=2).expand(n_batch, -1, -1, -1),
         torch.cat([longest_window[i][1]] + [c[i][1][:, :, 1:, :] for c in all_windows_except_longest], dim=2).expand(n_batch, -1, -1, -1))
        for i in range(n_layers))

def get_valid_length(input, pad_token_id):
    pad_mask = torch.ne(input, pad_token_id)
    valid_input_lengths = pad_mask.sum(dim=1)
    return valid_input_lengths

def seq_logits(res, tokenizer, inputs_len: int, max_new_tokens: int):
    # get new generated sequence logits under RestrictiveTokensLogitsProcessor, then use majority vote ensembling to determine the label token
    scores = res['scores']
    res = res['sequences'][:, inputs_len:]
    
    result_scores = torch.empty_like(res, dtype=torch.float16)
    for i in range(res.shape[1]):
        result_scores[:, i] = scores[i][torch.arange(res.shape[0]), res[:, i]]
    averages = torch.empty(res.shape[0], device=res.device, dtype=torch.float16)
    for i in range(result_scores.shape[0]):
        mask = result_scores[i] > LOGIT_BIAS
        selected_elements = result_scores[i][mask]
        averages[i] = torch.mean(selected_elements) if selected_elements.numel() > 0 else torch.tensor(float('nan'), device=res.device)

    res_list = []
    for i, r in enumerate(res):
        r = [x for x in r if x != tokenizer.eos_token_id]
        new_text = tokenizer.decode(r).lstrip().strip('\n')
        res_list.append(new_text)

    scores_dict = {}
    for score, candidate in zip(averages, res_list):
        if candidate in scores_dict:
            scores_dict[candidate] += score.item()
        else:
            scores_dict[candidate] = score.item()
    final_prediction = max(scores_dict, key=scores_dict.get)
    return [final_prediction]

class RestrictiveTokensLogitsProcessor(LogitsProcessor):
    """ Restrictive decoding is done by adding logits_bias to the relevant tokens. Based on:
    https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability
    """

    def __init__(self,
                 restrictive_token_ids: npt.NDArray,
                 eos_token_id: int,
                 prompt_length_to_skip: int = 0,
                 logits_bias: int = LOGIT_BIAS):
        self.restrictive_token_ids = restrictive_token_ids
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id
        self.logits_bias = logits_bias
        self.prompt_length_to_skip = prompt_length_to_skip
        self.mask = np.ones(restrictive_token_ids.shape[0], dtype=bool) # [batch]

        self._preprocess_restrictive_array()

    def _preprocess_restrictive_array(self):
        # extend restrictive_token_ids to include eos as last token for each sequence
        if not (self.restrictive_token_ids[:, -1] == self.eos_token_id).all():
            self.restrictive_token_ids = np.column_stack(
                (self.restrictive_token_ids, np.ones(self.restrictive_token_ids.shape[0]) * self.eos_token_id)).\
                astype(int)

    def update_new_prompt_length_to_skip(self, prompt_length_to_skip: List[int]):
        self.prompt_length_to_skip = prompt_length_to_skip
        self.mask = np.ones((len(self.prompt_length_to_skip), self.restrictive_token_ids.shape[0]), dtype=bool)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # newly generated tokens, self.prompt_length_to_skip is input_ids' length
        new_tokens_length = (get_valid_length(input_ids, self.pad_token_id) - self.prompt_length_to_skip).tolist()
        for i, k_length in enumerate(new_tokens_length):
            if k_length > 0:
                self.mask[i] = self.mask[i] & (self.restrictive_token_ids[:, k_length - 1] == input_ids[
                    i, -1].item())
            # next token score
            scores[i, self.restrictive_token_ids[self.mask[i], k_length]] += self.logits_bias
        return scores


class LlamaLong(LlamaForCausalLMPCW):
    def __init__(self,
                 config: PretrainedConfig,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 add_bos_token: bool = True,
                 ):
        super().__init__(config)
        self.tokenizer = tokenizer
        # The default behaviour of GPT2 is not to add bos_token in the beginning of the sequence, but most LLMs
        # have bos token and use it, so we chose to change this default behaviour.
        self.add_bos_token = add_bos_token
        self.context_window_size = LLM_WINDOW_SIZE

    def _get_windows(self, texts: List[str]) -> List[Dict]:
        windows = []
        for text in texts:
            encoded_input_window = self.tokenizer(text, return_tensors='pt').to(self.device)
            # lm forward
            output = self(**encoded_input_window)
            windows.append({'text': text,
                            'encoded_input': encoded_input_window,
                            'attention_mask': encoded_input_window['attention_mask'],
                            'window_size': encoded_input_window['input_ids'].shape[1],
                            'output': output,
                            'past': output['past_key_values']})
        return windows
    
    def get_ensmeble_cache(self, texts: List[str]) -> List[Dict]:
        encoded_input_window = self.tokenizer.batch_encode_plus(texts, padding=True, return_tensors='pt').to(self.device)
        # add_special_tokens=False, might leads to small difference!
        output = self(**encoded_input_window)
        cache ={'past_attention_mask': encoded_input_window['attention_mask'],
                'max_window_size': encoded_input_window['input_ids'].shape[1],
                'past_key_values': output['past_key_values']}
        cache['sum_windows_size'] = cache['max_window_size']
        return cache        

    def get_contexts_cache(self, contexts: List[str], n_batch:int = 1, longest: bool = True) -> Dict:
        windows = self._get_windows(contexts)
        if longest:
                windows_sizes = [window['window_size'] for window in windows]
                j = np.argmax(windows_sizes)
                return {'past_key_values': longest_combine_past_key_values([window['past'] for window in windows], j, n_batch = n_batch),
                        'max_window_size': max(windows_sizes),
                        'past_attention_mask': torch.cat(
                            [windows[j]['attention_mask']] + [window['attention_mask'][:, 1:] for window in
                                                            windows[:j] + windows[j + 1:]], dim=1),
                        'sum_windows_size': sum(windows_sizes) - (len(windows) - 1)}
        else:
            res = {'past_key_values': combine_past_key_values([window['past'] for window in windows],
                                                            n_batch = n_batch,
                                                            contains_bos_token=self.add_bos_token),
                'max_window_size': max(window['window_size'] for window in windows)}
        if self.add_bos_token:  # if windows contain bos tokens, we remove all but one to avoid multiple bos
            res['past_attention_mask'] = torch.cat([windows[0]['attention_mask']] + [window['attention_mask'][:, 1:]
                                                                                     for window in windows[1:]], dim=1)
            res['sum_windows_size'] = sum(window['window_size'] for window in windows) - (len(windows) - 1)
        else:
            res['past_attention_mask'] = torch.cat([window['attention_mask'] for window in windows], dim=1)
            res['sum_windows_size'] = sum(window['window_size'] for window in windows)
        return res
    
    def pcw_ensemble_generate(self,
                     contexts: Optional[List[str]] = None,
                     task_text: Optional[List[str]] = None,
                     contexts_cache: Optional[Dict] = None,
                     restrictive_logit_preprocessor: Optional[RestrictiveTokensLogitsProcessor] = None,
                     n_windows: int = 3,
                     **kwargs
                     ) -> str:
        assert (contexts is None) != (
                contexts_cache is None), "pcw_generate should work with contexts or cache, not with both!"
        cache = contexts_cache # or self.get_contexts_cache(contexts, n_batch)
        encoded_task_text = self.tokenizer.batch_encode_plus(task_text, add_special_tokens=False, padding=True, return_tensors='pt').to(self.device)
        input_ids = encoded_task_text['input_ids'].repeat(n_windows, 1)
        if restrictive_logit_preprocessor: 
            valid_input_lengths = get_valid_length(input_ids, self.tokenizer.pad_token_id)
            restrictive_logit_preprocessor.update_new_prompt_length_to_skip(valid_input_lengths)
            kwargs['logits_processor'] = [restrictive_logit_preprocessor]
        combined_attention_mask = torch.cat((cache['past_attention_mask'], encoded_task_text['attention_mask'].repeat(n_windows, 1)), dim=1)
        kwargs['output_scores'] = True
        kwargs['return_dict_in_generate'] = True
        res = self.generate(input_ids=input_ids,
                            attention_mask=combined_attention_mask,
                            windows_key_values=cache['past_key_values'],
                            max_window_size=cache['max_window_size'],
                            sum_windows_size=cache['sum_windows_size'],
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id,
                            ensemble = True,
                            **kwargs)
        inputs_len = encoded_task_text['input_ids'].shape[1]
        res_list = seq_logits(res, self.tokenizer, inputs_len, kwargs['max_new_tokens'])   
        return res_list     

    def pcw_batch_generate(self,
                     contexts: Optional[List[str]] = None,
                     task_text: Optional[List[str]] = None,
                     contexts_cache: Optional[Dict] = None,
                     restrictive_logit_preprocessor: Optional[RestrictiveTokensLogitsProcessor] = None,
                     generation: bool = False, 
                     **kwargs
                     ) -> str:
        assert (contexts is None) != (
                contexts_cache is None), "pcw_generate should work with contexts or cache, not with both!"
        n_batch = len(task_text)
        cache = contexts_cache # or self.get_contexts_cache(contexts, n_batch)
        ## TODO: to(self.device) = cuda:0
        encoded_task_text = self.tokenizer.batch_encode_plus(task_text, add_special_tokens=False, padding=True, return_tensors='pt').to(self.device)
        if restrictive_logit_preprocessor: 
            valid_input_lengths = get_valid_length(encoded_task_text['input_ids'], self.tokenizer.pad_token_id)
            restrictive_logit_preprocessor.update_new_prompt_length_to_skip(valid_input_lengths)
            kwargs['logits_processor'] = [restrictive_logit_preprocessor]
        combined_attention_mask = torch.cat((cache['past_attention_mask'].repeat(n_batch, 1), encoded_task_text['attention_mask']), dim=1)
        
        if generation:
            eos_token_id = kwargs.pop('eos_token_id')
        else:
            eos_token_id = self.tokenizer.eos_token_id
        res = self.generate(input_ids=encoded_task_text['input_ids'],
                            attention_mask=combined_attention_mask,
                            windows_key_values=cache['past_key_values'],
                            max_window_size=cache['max_window_size'],
                            sum_windows_size=cache['sum_windows_size'],
                            eos_token_id=eos_token_id,
                            pad_token_id=self.tokenizer.eos_token_id,
                            **kwargs)
        inputs_len = encoded_task_text['input_ids'].shape[1]
        res_list = []
        for r in res:
            r = r[inputs_len:]
            r = [x for x in r if x != self.tokenizer.eos_token_id]
            # r = r[:-1] if r[-1] == self.tokenizer.eos_token_id else r
            new_text = self.tokenizer.decode(r).lstrip().strip('\n')
            res_list.append(new_text)
        return res_list

    def pcw_generate(self,
                     contexts: Optional[List[str]] = None,
                     task_text: Optional[str] = None,
                     contexts_cache: Optional[Dict] = None,
                     restrictive_logit_preprocessor: Optional[RestrictiveTokensLogitsProcessor] = None,
                     **kwargs
                     ) -> str:
        """Note: Batching is not supported by PCW at the moment. """
        assert (contexts is None) != (
                contexts_cache is None), "pcw_generate should work with contexts or cache, not with both!"
        cache = contexts_cache or self.get_contexts_cache(contexts)
        encoded_task_text = self.tokenizer(task_text, add_special_tokens=False, return_tensors='pt').to(self.device)
        if restrictive_logit_preprocessor:
            restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_task_text['input_ids'].shape[1])
            kwargs['logits_processor'] = [restrictive_logit_preprocessor]
        combined_attention_mask = torch.cat((cache['past_attention_mask'], encoded_task_text['attention_mask']), dim=1)
        res = self.generate(input_ids=encoded_task_text['input_ids'],
                            attention_mask=combined_attention_mask,
                            windows_key_values=cache['past_key_values'],
                            max_window_size=cache['max_window_size'],
                            sum_windows_size=cache['sum_windows_size'],
                            pad_token_id=self.tokenizer.eos_token_id,
                            **kwargs)[0]
        res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
        return self.tokenizer.decode(res[encoded_task_text['input_ids'].shape[1]:])

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.LongTensor,
                                      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      windows_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                                      max_window_size: Optional[int] = None,
                                      sum_windows_size: Optional[int] = None,
                                      ensemble:  Optional[bool] = False, 
                                      **kwargs
                                      ) -> Dict:
        """input_ids:
            ids of task_tokens.
         attention_mask:
            concatenation of windows + task tokens attentions masks.

         Note (past_key_values vs windows_key_values):
             In the first token generation, past_key_values is None while windows_key_values contains the combined past
             key values of context windows. During following generations, past_key_values is the concatenation of
             windows_key_values + previous generations. Thus, windows_key_values is practically ignored.
             """
        # only last token for inputs_ids if past_key_values is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")

        if attention_mask is not None and position_ids is None:
            # create PCW's position_ids on the fly
            position_ids = attention_mask.long().cumsum(-1) - 1
            if not ensemble:
                n_task_tokens = [l + 1 - sum_windows_size for l in position_ids[:, -1].cpu().tolist()]
                for i, k in enumerate(n_task_tokens):
                    position_ids[i, -k:] = torch.arange(max_window_size, max_window_size + k, 1)
            else:
                n_task_tokens = position_ids.shape[1] - sum_windows_size
                for i in range(position_ids.shape[0]):
                    position_ids[i, -n_task_tokens:] = torch.arange(max_window_size, max_window_size + n_task_tokens, 1)

            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:  # i.e., first token is already generated
                position_ids = position_ids[:, -1].unsqueeze(-1)
            elif windows_key_values:  # i.e., we are in the first token generation
                position_ids = position_ids[:, sum_windows_size:]
        else:
            position_ids = None

        if windows_key_values and not past_key_values:
            past_key_values = windows_key_values
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }