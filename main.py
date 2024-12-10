import os
import argparse
from pathlib import Path
from tools.train_net import train
from utils.config import *

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        #If we know the input sizes don't change
        #benchmark mode is good whenever your input sizes for your network do not vary. 
        #Check: https://chatgpt.com/share/675749c3-a148-800d-80a2-2eb9d6b54b79
        torch.backends.cudnn.benchmark = True
    
    #<TODO> cudnn deterministic option and random seeds
    #<TODO> DISTRIBUTED INITIALIZATION 
    #<TODO> LOGGING (logging or wandb) (no need?)
    #<TODO> TENSORBOARDX (Summary Writer)

    # config
    config = get_config(args, logger = None)
    
    #Pretrain
    train()