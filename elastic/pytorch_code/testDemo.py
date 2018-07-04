import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet
import torch.backends.cudnn as cudnn
from torchsummary import summary
import os
import time
import datetime
import shutil
import sys

import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from opts import args

from data_loader import get_train_valid_loader, get_test_loader
# from models import *
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, inception_v3, densenet121
from collections import OrderedDict
from helper import LOG, log_summary, log_stats, AverageMeter, accuracy, save_checkpoint, adjust_learning_rate

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



__all__ = ['Elastic_ResNet18', 'Elastic_ResNet34', 'Elastic_ResNet101', 'Elastic_ResNet152']
global outputs
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# ===========================================================================below residual network source code ================
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class IntermediateClassifier(nn.Module):

    def __init__(self, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """
        kernel_size = 56
        if num_channels == 64:
            kernel_size = 56
        elif num_channels == 128:
            kernel_size = 28
        elif num_channels == 256:
            kernel_size = 14
        elif num_channels == 512:
            kernel_size = 7
        else:
            NotImplementedError

        super(IntermediateClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(kernel_size, kernel_size))
        )
        print("add 1 inter_CLF")

        self.classifier = nn.Linear(num_channels, 10)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.intermediate_CLF = []
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        # layers = [None] * (1+(blocks-1)*2) #自己添加的
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        print("blocks: ", 1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            print("blocks: ", i+1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
            # layers[i+blocks] = IntermediateClassifier(planes, 10)
            self.intermediate_CLF = IntermediateClassifier(planes, 10)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

# layer1 的3个intermediate classifiers
        layer1_0 = self.layer1[0]
        x1_layer1_0 = layer1_0(x)

        layer1_1 = self.layer1[1]
        x1_layer1_1 = layer1_1(x1_layer1_0)
        
        layer1_2 = self.layer1[2]
        x1_layer1_2 = layer1_2(x1_layer1_1)

        x2 = self.layer2(x1_layer1_2)
        
        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x6 = x.view(x5.size(0), -1)
        x7 = self.fc(x6)
        
        x7 = [x7] + [x1_layer1_0, x1_layer1_1, x1_layer1_2]
        return x7


def Elastic_ResNet50(args):
    
    num_classes = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    batch_size = args.batch_size
    
    model = ResNet(Bottleneck, [3, 4, 6, 3])
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    for param in model.parameters():
        param.requires_grad = True
    print("=====> successfully load pretrained imagenet weight")
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    
    return model


if __name__ == "__main__":
        
    args.batch_size = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TUT thinkstation data folder path
    # data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/data"

    # narvi data folder path
    # data_folder = "/home/zhouy/Elastic/data"

    # XPS 15 laptop data folder path
    data_folder = "D:\Elastic\data"

    # train_loader, val_loader = get_train_valid_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False,
    #                                                 random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
    #                                                 num_workers=1,pin_memory=True)
    test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = (229,229,3),
                                    num_workers=1,pin_memory=True)



    # model = resnet50(pretrained=True)
    # model = inception_v3(pretrained=True)

    # model = inception_v3(pretrained=True)

    # model = Net()
    # tar = torch.load('D:\Elastic\elastic\pytorch_code\models\pytorch-mobilenet\mobilenet_sgd_68.848.pth.tar')
    # state_dict = tar["state_dict"]
    
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    print("not with loading pretraining weights")
    model = Elastic_ResNet50(args)

    
    # fc_features = model.fc.in_features

    # model.fc = nn.Linear(fc_features, 10)






    # x1_out = model.layer1[0].relu
    # x2_out = model.layer1[1].relu

    # model = model.to(device)
    # model.cuda()
    # if device == 'cuda':
    #     model = torch.nn.DataParallel(model).cuda()
    #     cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()#.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)# nesterov set False to keep align with keras default settting

    model.train()
    output = None
    for i, (input, target) in enumerate(test_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)
        # loss = criterion(output, target_var)
        # print("loss: ", loss.item())
        # prec1 = accuracy(output.data, target)
        # print("precision: ", prec1[0].data[0].item())

        # # x1_out_1 = x1_out(input_var)
        # # block_out_1 = nn.AvgPool2d(kernel_size=(224, 224))(x1_out_1)
        # # block_out_1_1 = block_out_1.view(16, 3)
        # # inter_clf_block_1 = torch.nn.Linear(3, 10)(block_out_1_1)


        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # summary(model, (3, 229, 229))


print("Done")


        


