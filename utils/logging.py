import logging
import torch.distributed as dist

def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        print(msg)