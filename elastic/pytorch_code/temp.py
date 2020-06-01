# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# =============================================================================
# import matplotlib
# matplotlib.use("PDF")
# import matplotlib.pyplot as plt
# plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
# plt.rcParams["figure.figsize"] = (12,6)
# =============================================================================

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
from torchvision.models import inception_v3, vgg16_bn
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

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class IntermediateClassifier(nn.Module):

    def __init__(self, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """
        super(IntermediateClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.device = 'cuda'
        kernel_size = int(7168/self.num_channels)
            
        print("kernel_size for global pooling: " ,kernel_size)

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(kernel_size, kernel_size)),
            nn.Dropout(p=0.2, inplace=False)
        ).to(self.device)
        # print("num_channels: ", num_channels, "\n")
        # 在keras中这里还有dropout rate = 0.2，但是这里没有，需要添加一下
        self.classifier = torch.nn.Sequential(nn.Linear(self.num_channels, self.num_classes)).to(self.device)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """
        # get the width or heigh on that feaure map
        # kernel_size = x.size()[-1]
        # get the number of feature maps
        # num_channels = x.size()[-3]
        
        # print("kernel_size for global pooling: " ,kernel_size)
        

        # do global average pooling
        x = self.features(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # add intermediate classifier after pooling
            print("v, in_channels: ", v)
            layers += [IntermediateClassifier(v, 100)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    

    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def main(**kwargs):
    
    global args
# =============================================================================
#     lowest_error1 = 100
# =============================================================================
    
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    args.batch_size = 1
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TUT thinkstation data folder path
    data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/data"
    
    # narvi data folder path
    # data_folder = "/home/zhouy/Elastic/data"
    
    # XPS 15 laptop data folder path
    #data_folder = "D:\Elastic\data"
    
    
    # =============================================================================
    # train_loader, val_loader = get_train_valid_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False,
    #                                                 random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
    #                                                 num_workers=1,pin_memory=True)
    # =============================================================================
    
    test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = (224,224,3),
                                    num_workers=4,pin_memory=True)
# =============================================================================
#     test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = args.target_size,
#                                 num_workers=4,pin_memory=True)
# =============================================================================

   #model = vgg16_bn(pretrained=True)
    model = VGG(make_layers(cfg['D'], batch_norm=True))
    
    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, 100)

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
    for i, (input, target) in enumerate(test_loader):
        # lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader))

        # data_time.update(time.time() - end)
        summary(model, (3,224,224))

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        #print("target size: ", target_var.shape())
    # =============================================================================
    #     losses = 0
    # =============================================================================
        optimizer.zero_grad()

        outputs = model(input_var)

        loss = criterion(outputs, target_var)
        # 这里应该要再封装一下， 变成只有一个变量loss

        loss.backward()
        optimizer.step()

    summary(model, (3,224,224))

# =============================================================================
# with open(data_folder + os.sep + "resnet34_model.txt", 'w') as f:
#     f.write(str(model))
# =============================================================================

    
    
    
    
    
if __name__ == "__main__":
    main()
