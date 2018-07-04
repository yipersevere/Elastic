from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from helper import LOG

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
        super(IntermediateClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.device = 'cuda'
        kernel_size = int(14336/self.num_channels)
        print("kernel_size for global pooling: " ,kernel_size)




        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=(kernel_size, kernel_size))
        ).to(self.device)
        # print("num_channels: ", num_channels, "\n")
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


class ResNet(nn.Module):

    def __init__(self, block, layers, cifar_classes, add_intermediate_layers, num_classes=1000):
        self.intermediate_CLF = []
        self.add_intermediate_layers = add_intermediate_layers
        self.cifar_classes = cifar_classes
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
        self.inplanes = planes * block.expansion
        
        if self.add_intermediate_layers == 2:
            self.intermediate_CLF.append(IntermediateClassifier(self.inplanes, self.cifar_classes))

        # print("blocks: ", 1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # print("blocks: ", i+1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
            if self.add_intermediate_layers == 2:
                self.intermediate_CLF.append(IntermediateClassifier(self.inplanes, self.cifar_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        i = 0
        intermediate_outputs = []
        # print("=====> # of intermediate classifiers: ", len(self.intermediate_CLF), ", total classifiers: ", len(self.intermediate_CLF)+1)
        
        # make sure insert an intermediate classifier after each residul block 
        # assert len(self.intermediate_CLF) == len(self.layer1)+len(self.layer2)+len(self.layer3)+len(self.layer4)

        for res_layer in self.layer1:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1

        for res_layer in self.layer2:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1

        for res_layer in self.layer3:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1                    
        
        for res_layer in self.layer4:
            x = res_layer(x)
            if self.add_intermediate_layers == 2:
                intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return intermediate_outputs+[x]


def Elastic_ResNet50(args, logfile):
    
    num_classes = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    pretrained_weight = args.pretrained_weight

    if add_intermediate_layers == 0: # not adding any intermediate layer classifiers
        print("not adding any intermediate layer classifiers")    
        LOG("not adding any intermediate layer classifiers", logfile)
    elif add_intermediate_layers == 2:
        print("add any intermediate layer classifiers")    
        LOG("add any intermediate layer classifiers", logfile)
    else:
        NotImplementedError

    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, add_intermediate_layers)



    if pretrained_weight == 1:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print("loaded ImageNet pretrained weights")
        LOG("loaded ImageNet pretrained weights", logfile)
        
    elif pretrained_weight == 0:
        print("not loading ImageNet pretrained weights")
        LOG("not loading ImageNet pretrained weights", logfile)
    else:
        print("parameter--pretrained_weight, should be 0 or 1")
        LOG("parameter--pretrained_weight, should be 0 or 1", logfile)
        NotImplementedError

    for param in model.parameters():
        param.requires_grad = True
    # print("=====> successfully load pretrained imagenet weight")
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    
    return model