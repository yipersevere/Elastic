import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet
import torch.backends.cudnn as cudnn
from ignite.handlers import EarlyStopping
from torchsummary import summary

import os
import time
import datetime
import shutil
import sys

from opts import args
from helper import LOG, log_summary, log_stats, Plot, AverageMeter, accuracy, save_checkpoint, adjust_learning_rate
from data_loader import get_train_valid_loader, get_test_loader
import models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# data_folder = "/home/zhouy/Elastic/data"

# XPS 15 laptop data folder path
data_folder = "D:\Elastic\data"

train_loader, val_loader = get_train_valid_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False,
                                                random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
                                                num_workers=1,pin_memory=True)
# test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True,
#                                 num_workers=1,pin_memory=True)



model = resnet50(pretrained=True)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 10)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=False)# nesterov set False to keep align with keras default settting

model.train()
for i, (input, target) in enumerate(train_loader):
    
    optimizer.zero_grad()
    x1_out = model.layer1.data
    

    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    
    output = model(input_var)
    loss = criterion(output, target_var)
    loss.backward()
    optimizer.step()


print("Done")


        

