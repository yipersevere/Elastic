# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
plt.rcParams["figure.figsize"] = (12,6)

import os
import numpy as np
import scipy
import datetime
from io import StringIO
import shutil
import random
import sys
import time
import sys
import csv


import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

from helper import LOG, log_summary, log_stats, AverageMeter, accuracy, save_checkpoint
from data_loader import get_train_loader, get_test_loader
from models import *



def main(**kwargs):
    
#     global args
# # =============================================================================
# #     lowest_error1 = 100
# # =============================================================================
    
#     for arg, v in kwargs.items():
#         args.__setattr__(arg, v)

#     args.batch_size = 1
    
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # TUT thinkstation data folder path
#     data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/data"
    
#     # narvi data folder path
#     # data_folder = "/home/zhouy/Elastic/data"
    
#     # XPS 15 laptop data folder path
#     #data_folder = "D:\Elastic\data"
    
    
#     # =============================================================================
#     # train_loader, val_loader = get_train_valid_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False,
#     #                                                 random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
#     #                                                 num_workers=1,pin_memory=True)
#     # =============================================================================
    
#     test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = (299,299,3),
#                                     num_workers=4,pin_memory=True)
# # =============================================================================
# #     test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = args.target_size,
# #                                 num_workers=4,pin_memory=True)
# # =============================================================================

#     model = inception_v3()
    
#     fc_features = model.fc.in_features
    
#     model.fc = nn.Linear(fc_features, 100)
    
#     model = model.to(device)
#     model.cuda()
#     if device == 'cuda':
#         model = torch.nn.DataParallel(model).cuda()
#         cudnn.benchmark = True
    
    
#     criterion = nn.CrossEntropyLoss().cuda()
#     optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay,
#                                 nesterov=False)# nesterov set False to keep align with keras default settting
    
#     model.train()
#     for i, (input, target) in enumerate(test_loader):
#         # lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader))
    
#         # data_time.update(time.time() - end)
        
    
#         target = target.cuda(async=True)
#         input_var = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
#         #print("target size: ", target_var.shape())
#     # =============================================================================
#     #     losses = 0
#     # =============================================================================
#         optimizer.zero_grad()
    
#         outputs = model(input_var)
    
#         loss = criterion(outputs, target_var)
#         # 这里应该要再封装一下， 变成只有一个变量loss
    
#         loss.backward()
#         optimizer.step()

# =============================================================================
# summary(model, (3,224,224))
# =============================================================================

# =============================================================================
# with open(data_folder + os.sep + "resnet34_model.txt", 'w') as f:
#     f.write(str(model))
# =============================================================================
    pd.read

    
    
    
    
    
if __name__ == "__main__":
    main()
