import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logging import *
import cv2
import numpy as np

def test_net(args, config):
    logger = None
    # print_log('Tester start ... ', logger = logger)
    print('Tester start ... ')
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.experiment_model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_iter = iter(test_dataloader)

    # Get one batch of test data
    data = next(test_iter)

    # Send to GPU
    points = data.cuda()

    with torch.no_grad():
        # Forward pass
        loss = base_model.forward(points)

    # Print the loss
    print("loss: ", loss)