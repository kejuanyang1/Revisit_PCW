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

"""parses arguments and preps data loader"""
import torch
import torch.utils.data
import data_utils
from sat import mpu

def make_eval_data_loader(dataset, batch_size, distributed=False, num_iters=1, shuffle=False, collator=None, num_workers=0, keep_last=False, loader_scatter=None):
    if distributed:
        world_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        if loader_scatter is not None:
            loader_scatter = min(loader_scatter, mpu.get_data_parallel_world_size())
            rank = rank // loader_scatter
            world_size = world_size // loader_scatter
            batch_size = batch_size // loader_scatter
    if shuffle:
        sampler = data_utils.samplers.RandomSampler(dataset, replacement=True,
                                                    num_samples=batch_size * num_iters)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    drop_last = distributed and not keep_last
    # the GPUs in the same model parallel group receive the same data
    if distributed:
        batch_sampler = data_utils.samplers.DistributedBatchSampler(sampler, batch_size, drop_last, rank,
                                                                    world_size)
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                        batch_size,
                                                        drop_last)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              collate_fn=collator)

    return data_loader