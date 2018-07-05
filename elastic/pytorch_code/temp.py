# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import inception_v3, resnet34
import torch.backends.cudnn as cudnn
# =============================================================================
# from ignite.handlers import EarlyStopping
# =============================================================================
from torchsummary import summary

import os
import time
import datetime
import shutil
import sys


from opts import args

from helper import LOG, log_summary, log_stats, AverageMeter, accuracy, save_checkpoint, adjust_learning_rate
from data_loader import get_train_valid_loader, get_test_loader
from models import *

args.batch_size = 1


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TUT thinkstation data folder path
# data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/data"

# narvi data folder path
# data_folder = "/home/zhouy/Elastic/data"

# XPS 15 laptop data folder path
data_folder = "D:\Elastic\data"


# =============================================================================
# train_loader, val_loader = get_train_valid_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False,
#                                                 random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
#                                                 num_workers=1,pin_memory=True)
# =============================================================================

test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = args.target_size,
                                num_workers=4,pin_memory=True)

model = resnet34(pretrained=True)

fc_features = model.fc.in_features

model.fc = nn.Linear(fc_features, 10)

model = model.to(device)
model.cuda()
if device == 'cuda':
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=False)# nesterov set False to keep align with keras default settting

model.train()

summary(model, (3,224,224))

# =============================================================================
# with open(data_folder + os.sep + "resnet34_model.txt", 'w') as f:
#     f.write(str(model))
# =============================================================================

