import os
import argparse
from pathlib import Path
from tools.train_net import train
from utils.config import *
from utils import parser, dist_utils, misc
from tensorboardX import SummaryWriter
import torch
from utils.logging import *
import wandb

torch.autograd.set_detect_anomaly(True)

def main():

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="obitonet",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "PointMAE",
        "dataset": "TanksAndTemples",
        "epochs": 300,
        }
    )
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        #If we know the input sizes don't change
        #benchmark mode is good whenever your input sizes for your network do not vary. 
        #Check: https://chatgpt.com/share/675749c3-a148-800d-80a2-2eb9d6b54b79
        torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    
    #LOGGING (TODO - wandb)

    
    #TENSORBOARDX (Summary Writer)
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None

    # config
    config = get_config(args, logger = None)

    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size 
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs 
    
    # set random seeds
    if args.seed is not None:
        print_log(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    
    # if args.distributed:
    #     assert args.local_rank == torch.distributed.get_rank() 

    # Print the config
    # print("config: ", config)
    # return
    
    #Pretrain
    train(args, config, train_writer, val_writer)

if __name__ == '__main__':
    main()