import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def Elastic_MobileNet(args, logfile):
    """
    based on MobileNet Version1 and ImageNet pretrained weight, https://github.com/marvis/pytorch-mobilenet
    但是这里并没有实现 alpha 乘子和width 乘子
    """
    num_classes = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    pretrained_weight = args.pretrained_weight

    model = Net()

    if pretrained_weight == 1:
        
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        print("loaded ImageNet pretrained weights")
        LOG("loaded ImageNet pretrained weights", logfile)
        
    elif pretrained_weight == 0:
        print("not loading ImageNet pretrained weights")
        LOG("not loading ImageNet pretrained weights", logfile)

    else:
        print("parameter--pretrained_weight, should be 0 or 1")
        LOG("parameter--pretrained_weight, should be 0 or 1", logfile)
        NotImplementedError

    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    # print("=====> InceptionV3, successfully load pretrained imagenet weight")

    for param in model.parameters():
        param.requires_grad = False
    
    if add_intermediate_layers == 2:
        print("add any intermediate layer classifiers")    
        LOG("add intermediate layer classifiers", logfile)

        # get all extra classifiers params and final classifier params
        for inter_clf in model.intermediate_CLF:
            for param in inter_clf.parameters():
                param.requires_grad = True
        
        for param in model.fc.parameters():
            param.requires_grad = True 
    
    elif add_intermediate_layers == 0:
        print("not adding any intermediate layer classifiers")    
        LOG("not adding any intermediate layer classifiers", logfile)

        for param in model.fc.parameters():
            param.requires_grad = True         
    else:
        NotImplementedError
    return model
