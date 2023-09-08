# Inspired from maskrcnn_benchmark, fairseq
import contextlib
import logging
import os
import pickle
import socket
import subprocess
import warnings
from itertools import chain
import random
import numpy as np 

import torch
from src.common.registry import registry
from torch import distributed as dist
from packaging import version

#try:
#   import torch_xla.core.xla_model as xm
#except ImportError:
#    xm = None
logger = logging.getLogger(__name__)

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def load_fp16_scaler(training_config, device, optimizer):
    if training_config.fp16:
        assert version.parse(torch.__version__) >= version.parse(
            "1.6"
        ), f"Using fp16 requires torch version >- 1.6, found: {torch.__version__}"
        assert device != torch.device("cpu"), "fp16 cannot be used on cpu"

    set_torch_grad_scaler = True
    if training_config.fp16 and training_config.cuda.distributed:
        try:
            from fairscale.optim.grad_scaler import ShardedGradScaler
            from fairscale.optim.oss import OSS

            if isinstance(optimizer, OSS):
                scaler = ShardedGradScaler()
                set_torch_grad_scaler = False
                logger.info("Using FairScale ShardedGradScaler")
        except ImportError:
            logger.info("Using Pytorch AMP GradScaler")

    if set_torch_grad_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=training_config.fp16)
        autocast = torch.cuda.amp.autocast
    
    return scaler, autocast


# ------------------------------------ Distributed cuda training ----------------------------------------
def distributed_setup(config):
    """ Sets up for optional distributed training.
    For distributed training run as:
        python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    To kill zombie processes use:
        kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
    For data parallel training on GPUs or CPU training run as:
        python main.py --no_distributed
    """
    if config.training.cuda.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node
        world_size= int(os.environ.get('WORLD_SIZE'))
        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = None
        device = torch.device('cuda:{}'.format(config.training.cuda.device) if torch.cuda.is_available() else 'cpu')

    seed = 8  # 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False # keep this false; otherwise much slower
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank, world_size

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(dictionary):
    world_size = get_world_size()
    if world_size < 2:
        return dictionary

    with torch.no_grad():
        if len(dictionary) == 0:
            return dictionary

        keys, values = zip(*sorted(dictionary.items()))
        values = torch.stack(values, dim=0)

        #if is_xla():
        #    values = xm.all_reduce("sum", [values], scale=1.0 / world_size)[0]
        #else:
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(keys, values)}
    return reduced_dict