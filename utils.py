# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for logging and serialization"""

import os
import random
import time
import numpy as np
import torch
import json
import subprocess
import socket
from tqdm import tqdm
import logging
import re, string
from collections import Counter

import torch.distributed as dist

from sat import mpu
from tensorboardX import SummaryWriter

SUMMARY_WRITER_DIR_NAME = 'runs'

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 

def bleu(prediction, references):
    return sentence_bleu(references, prediction)

def rouge(prediction, ground_truth):
    rouge = Rouge()
    return rouge.get_scores(prediction, ground_truth)[0]['rouge-l']['f']

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_log_dir(name, base):
    return os.path.join(base, SUMMARY_WRITER_DIR_NAME, name)


def get_sample_writer(log_dir, iteration=0):
    """Returns a tensorboard summary writer
    """
    return SummaryWriter(
        log_dir=log_dir, purge_step=iteration)


def tqdm_rank_0(iterable, **kwargs):
    if torch.distributed.get_rank() == 0:
        return tqdm(iterable, **kwargs)
    else:
        return iterable

def print_rank_0(*message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)

def is_local_main_process():
    return (torch.distributed.get_rank() == 0)

def get_gpu_memory_usage():
    command = "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader"
    result = subprocess.check_output(command.split())
    memory_usage = [int(x) for x in result.decode("utf-8").strip().split("\n")]
    print(f"显存使用情况：{memory_usage}")
    return None

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    ans_labels = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs in all_label_probs:
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        ans_labels.append(ans_label)
    return ans_labels


def all_gather_nd(tensor):
    """ Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions
    in the tensors. All the other dimensions should be equal between the tensors.
    Args:
        tensor (Tensor): Tensor to be broadcast from current process.
    """
    world_size = dist.get_world_size()
    local_size = tensor.size(0)
    max_size = torch.tensor(local_size, dtype=torch.int64).cuda()

    dist.barrier()
    dist.all_reduce(max_size, op=dist.ReduceOp.MAX)
    max_size = max_size.item()
    sizes = [local_size] * world_size
    
    dist.all_gather_object(sizes, local_size)
    padded_tensor = torch.zeros((max_size,) + tuple(tensor.size()[1:]), dtype=tensor.dtype).cuda()
    padded_tensor[:local_size] = tensor
    gathered_tensor_list = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensor_list, padded_tensor)

    gathered_tensor_list = [t[:s] for t,s in zip(gathered_tensor_list, sizes)]
    
    return gathered_tensor_list

def init_distributed(world_size):
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size, rank=0,
        init_method=init_method)
    
    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(model_parallel_size_=1)
    return True

def get_device_map(max_mem = 20):
    num_gpus = torch.cuda.device_count()
    gpu_dict = {i: f'{max_mem}GB' for i in range(num_gpus)}
    return gpu_dict

class Timer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self, message=''):
        elapsed_time = time.time() - self.start_time
        self.logger.info(f'{message} Elapsed time: {elapsed_time:.4f} seconds')


def get_hostname():
    hostname_cmd = ["hostname -I"]
    result = subprocess.check_output(hostname_cmd, shell=True)
    master_addr = result.decode('utf-8').split()[0]
    return master_addr


def check_port_in_use(port, host='127.0.0.1'):
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, int(port)))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()


def get_spare_port(args):
    if torch.distributed.get_rank() == 0:
        port = random.randrange(10000, 65535)
        while port == args.master_port or check_port_in_use(port, args.master_ip):
            port = random.randrange(10000, 65535)
        port = torch.cuda.LongTensor([port])
    else:
        port = torch.cuda.LongTensor([0])
    torch.distributed.broadcast(port, 0)
    port = port.item()
    return port


def print_and_save_args(args, verbose=True, log_dir=None):
    """Print arguments."""
    if verbose:
        print('arguments:', flush=True)
        for arg in vars(args):
            dots = '.' * (29 - len(arg))
            print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)
    if log_dir is not None:
        json_file = os.path.join(log_dir, "config.json")
        with open(json_file, "w") as output:
            json.dump(vars(args), output, sort_keys=True)
        if args.deepspeed and args.deepspeed_config is not None:
            with open(args.deepspeed_config) as file:
                deepspeed_config = json.load(file)
            deepspeed_json_file = os.path.join(log_dir, "config_gpt_large.json")
            with open(deepspeed_json_file, "w") as output:
                json.dump(deepspeed_config, output)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, model-parallel,min, max, norm\n'
    optimizer_ = optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = param.data.norm()
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def get_checkpoint_tracker_filename(checkpoints_path):
    path = os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')
    if not os.path.isfile(path):
        path = os.path.join(checkpoints_path, 'latest')
    return path


def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {'iteration': iteration,
               'optimizer_state_dict': optimizer.state_dict()}
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    torch.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))


def save_checkpoint(iteration, model, optimizer, lr_scheduler, args, tag=None, barrier=True,
                    only_changed_parameters=False, no_deepspeed=False, no_save_optim=False):
    """Save a model checkpoint."""
    if tag is None:
        tag = str(iteration)
    if args.deepspeed and not no_deepspeed and not args.no_deepspeed_save:
        save_ds_checkpoint(iteration, model, lr_scheduler, args, tag=tag)
    else:
        # Only rank zer0 of the data parallel writes to the disk.

        if mpu.get_data_parallel_rank() == 0:
            checkpoint_name = get_checkpoint_name(args.save, tag)
            print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
                  format(torch.distributed.get_rank(), iteration, checkpoint_name))
            sd = {'iteration': iteration}
            if args.deepspeed:
                model = model.module
            state_dict = model.state_dict()
            if only_changed_parameters:
                requires_grad_dict = {}
                for name, parameter in model.named_parameters():
                    requires_grad_dict[name] = parameter.requires_grad
                state_dict = {key: value for key, value in state_dict.items() if requires_grad_dict.get(key, True)}
            sd['module'] = state_dict

            # Optimizer stuff.
            if not args.no_save_optim and not no_save_optim:
                if optimizer is not None:
                    sd['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    sd['lr_scheduler'] = lr_scheduler.state_dict()

            # rng states.
            if not args.no_save_rng:
                sd['random_rng_state'] = random.getstate()
                sd['np_rng_state'] = np.random.get_state()
                sd['torch_rng_state'] = torch.get_rng_state()
                sd['cuda_rng_state'] = torch.cuda.get_rng_state()
                if args.checkpoint_activations:
                    sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

            ensure_directory_exists(checkpoint_name)
            torch.save(sd, checkpoint_name)
            print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    if barrier:
        torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(tag)


def save_ds_checkpoint(iteration, model, lr_scheduler, args, tag):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    if lr_scheduler is not None:
        sd['client_lr_scheduler'] = lr_scheduler.state_dict()
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()
    model.save_checkpoint(args.save, tag, client_state=sd)


def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        if os.path.isdir(load_path):
            path = os.path.normpath(load_path)
            load_dir, tag = os.path.split(path)
            print_rank_0('Try to directly load the checkpoint from the directory')
            return load_dir, tag, False, True
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return load_path, 0, False, False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        release = metastring == 'release'
        # try:
        #     iteration = int(metastring)
        # except ValueError:
        #     release = metastring == 'release'
        #     if not release:
        #         print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
        #             tracker_filename))
        #         exit()

    # assert iteration > 0 or release, 'error parsing metadata file {}'.format(
    #     tracker_filename)

    return load_path, metastring, release, True


def load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=False, no_load_optim=False):
    """Load a model checkpoint."""

    load_dir, tag, release, success = get_checkpoint_iteration(args.load)

    if not success:
        return 0

    if args.deepspeed and not no_deepspeed and not args.old_checkpoint:
        load_optimizer_states = not args.no_load_optim and not no_load_optim
        checkpoint_name, sd = model.load_checkpoint(load_dir, tag,
                                                    load_optimizer_states=load_optimizer_states,
                                                    load_lr_scheduler_states=not args.no_load_lr_scheduler)
        if not load_optimizer_states and (args.fp16 or args.bf16) and optimizer is not None:
            print_rank_0("Refresh fp32 parameters")
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer._model_params_to_master_params()
        if not args.no_load_lr_scheduler and "client_lr_scheduler" in sd:
            lr_scheduler.load_state_dict(sd["client_lr_scheduler"])
            print_rank_0("Load lr scheduler state")
        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return tag

    else:

        # Checkpoint.
        checkpoint_name = get_checkpoint_name(load_dir, tag, release)

        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        sd = torch.load(checkpoint_name, map_location='cpu')
        sd = {key: value for key, value in sd.items() if value is not None}
        # Model.
        if args.deepspeed and hasattr(model, "module"):
            model = model.module

        # Process the checkpoint for GLM
        if args.block_lm and args.old_checkpoint:
            sd['module']['transformer.word_embeddings.weight'] = sd['module']['word_embeddings.weight']
            del sd['module']['word_embeddings.weight']
            sd['module']['mixins.block_position_embedding.block_position_embeddings.weight'] = sd['module'][
                'transformer.block_position_embeddings.weight']
            del sd['module']['transformer.block_position_embeddings.weight']

        missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
        if missing_keys or unexpected_keys:
            print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")

        # Optimizer.
        optimizer_loaded = False
        if not release and not args.finetune and not args.no_load_optim and not no_load_optim and optimizer is not None:
            try:
                optimizer.load_state_dict(sd['optimizer'])
                optimizer_loaded = True
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}'
                             'Specify --no-load-optim or --finetune to prevent '
                             'attempting to load the optimizer '
                             'state.'.format(checkpoint_name))
        if optimizer_loaded and not args.no_load_lr_scheduler and lr_scheduler is not None:
            try:
                lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0('Unable to load lr scheduler from checkpoint {}'.format(checkpoint_name))
        if not optimizer_loaded and (args.fp16 or args.bf16) and optimizer is not None:
            print_rank_0("Refresh fp32 parameters")
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer._model_params_to_master_params()
    # Iterations.
    if args.finetune or release or args.no_load_iteration:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                             ' from checkpoint {}, starting from 0 iteration'.format(checkpoint_name))
                iteration = 0

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load random state from checkpoint {}, exiting. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the random '
                         'state.'.format(checkpoint_name))

    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


def load_pretrained(model, checkpoint_path, args, optimizer=None):
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    if mpu.get_data_parallel_rank() == 0:
        print('global rank {} is loading pretrained model {}'.format(
            torch.distributed.get_rank(), checkpoint_name))
    # Load the checkpoint.
    sd = torch.load(checkpoint_name, map_location='cpu')

    # Model.
    if args.block_lm and args.old_checkpoint:
        sd['module']['transformer.word_embeddings.weight'] = sd['module']['word_embeddings.weight']
        del sd['module']['word_embeddings.weight']
        sd['module']['mixins.block_position_embedding.block_position_embeddings.weight'] = sd['module'][
            'transformer.block_position_embeddings.weight']
        del sd['module']['transformer.block_position_embeddings.weight']

    if hasattr(model, "pretrained_model"):
        model = model.pretrained_model

    missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
    if (args.fp16 or args.bf16) and optimizer is not None:
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        print_rank_0("Refresh fp32 parameters")
        if args.deepspeed:
            optimizer.refresh_fp32_params()
        else:
            optimizer._model_params_to_master_params()


def load_weights(src, dst, dst2src=False):
    """
    Loads weights from src to dst via in place copy.
    src is a huggingface gpt2model, while dst is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src is still untested
    """
    conv_layer = 'Conv1D' in str(type(src))
    for n, p in src.named_parameters():
        if dst2src:
            data = dst._parameters[n].data
            load = p.data
        else:
            data = p.data
            load = dst._parameters[n].data
        if conv_layer and 'weight' in n:
            data = data.t().contiguous()
        load.copy_(data)


#        dst._parameters[n].data.copy_(data)

def load_mlp(our, oai, dst2src=False):
    load_weights(oai.c_fc, our.dense_h_to_4h, dst2src)
    load_weights(oai.c_proj, our.dense_4h_to_h, dst2src)


def load_attention(our, oai, dst2src=False):
    load_weights(oai.c_attn, our.query_key_value, dst2src)
    load_weights(oai.c_proj, our.dense, dst2src)


def load_transformer_layer(our, oai, dst2src=False):
    load_weights(oai.ln_1, our.input_layernorm, dst2src)
    load_weights(oai.ln_2, our.post_attention_layernorm, dst2src)
    load_mlp(our.mlp, oai.mlp, dst2src)
    load_attention(our.attention, oai.attn, dst2src)


def move_weights(our, oai, dst2src=False):
    """
    Loads weights from `oai` to `our` via in place copy.
    `oai` is a huggingface gpt2model, while `our` is one of our models.
    dst2src=True loads parameters from our models into huggingface's.
    ^dst2src=True is still untested
    """
    #    while isinstance(our, (torchDDP, model.distributed.DistributedDataParallel, FP16_Module)):
    #        our=our.module
    transformer_model = oai.transformer
    load_weights(transformer_model.ln_f, our.transformer.final_layernorm, dst2src)
    load_weights(transformer_model.wte, our.word_embeddings, dst2src)
    load_weights(transformer_model.wpe, our.position_embeddings, dst2src)

    for our_layer, oai_layer in zip(our.transformer.layers, oai.transformer.h):
        load_transformer_layer(our_layer, oai_layer, dst2src)


def debug_pretrain_data(local_vars, batch_id, tokenizer):
    tokens, target_ids = local_vars["text"], local_vars["target"]
    attention_mask, logit_mask, position_ids = local_vars["attention_mask"], local_vars["loss_mask"], local_vars[
        "position_id"]
    sep = attention_mask[batch_id].item() - 16 * 16 - 1
    context, piece = "", []
    for i, token_id in enumerate(tokens[batch_id][:sep].tolist()):
        token = tokenizer.IdToToken(token_id)
        if "MASK" in token:
            context += " " + tokenizer.DecodeIds(piece)
            context += f" [MASK,{i}]"
            piece = []
        else:
            piece.append(token_id)
    if piece:
        context += " " + tokenizer.DecodeIds(piece)
    print(context)
    start_index = sep
    for i in range(sep, tokens.size(-1)):
        if logit_mask[batch_id][i].item() == 0:
            break
        target_id = target_ids[batch_id][i].item()
        if target_id == tokenizer.get_command("eop").Id:
            print(tokenizer.DecodeIds(tokens[batch_id][start_index: i + 1].tolist()), " | ",
                  tokenizer.DecodeIds(target_ids[batch_id][start_index: i + 1].tolist()),
                  position_ids[batch_id][:, start_index: i + 1]. tolist(),)
            start_index = i + 1


def debug_finetune_data(local_vars, batch_id, tokenizer):
    tokens, target_ids = local_vars["text"], local_vars["target"]
    attention_mask, logit_mask, position_ids = local_vars["attention_mask"], local_vars["loss_mask"], local_vars[
        "position_id"]
    sep = attention_mask[batch_id].item()
    for i, token in enumerate(tokens[batch_id][:sep].tolist()):
        token = tokenizer.IdToToken(token)
        if token == '[MASK]':
            print("mask position", i)
    print(tokenizer.DecodeIds(tokens[batch_id][:sep].tolist()))
    target_positions = []
    for i in range(sep, tokens.size(-1)):
        if logit_mask[batch_id][i]:
            target_positions.append(i)
    print(target_positions)
    print(tokenizer.DecodeIds(tokens[batch_id][target_positions].tolist()))
    print(tokenizer.DecodeIds(target_ids[batch_id][target_positions].tolist()))
    print(position_ids[batch_id][:, target_positions])


def compare_dict(dict1, dict2):
    for key in dict1:
        if key in dict2:
            if dict1[key] != dict2[key]:
                print(f"{key}, {dict1[key]}, {dict2[key]}")
        else:
            print(f"{key}, {dict1[key]}, null")
    for key in dict2:
        if key not in dict1:
            print(f"{key}, null, {dict2[key]}")