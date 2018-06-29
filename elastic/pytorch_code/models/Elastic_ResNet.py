from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division


import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

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




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.layer1_inter_1 = self._make_layer(block, 64, layers[0])
        # self.intermediate_layer_1 = self.build_intermediate_layer(layers)
        self.layer2, = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.x1_out_avgpool = nn.AvgPool2d(kernel_size=(56,56))
        # self.x1_out_linear = nn.Linear(256, 10)
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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers), layers[0].relu

    def add_intermediate_layers(self, base_model):
        # x1 = self.layer1[0].relu
        # x1 = x1(base_model)
        
        block_out_1 = nn.AvgPool2d(kernel_size=(224, 224))(base_model)
        # block_out_1_1 = block_out_1.view(16, 3)
        inter_clf_block_1 = torch.nn.Linear(3, 10)(block_out_1)
        
        return inter_clf_block_1
    def build_intermediate_layer(self):

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1, x1_intermediate = self.layer1(x)
        # x1_out = nn.AvgPool2d(kernel_size=(56,56))(x1)
        # x1_out = Reshape(256).forward(x1_out)
        # x1_out = self.x1_out_linear(x1_out)
        # x1_out = nn.Linear(256, 10)(x1_out)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.avgpool(x4)
        x5 = x.view(x.size(0), -1)
        x6 = self.fc(x)
        
        intermediate_outputs = self.add_intermediate_layers(.layer1[0].relu)

        return x6, intermediate_outputs


class Elastic_ResNet50():
    def __init__(self, args):
        self.num_classes = args.num_classes
        self.add_intermediate_layers = args.add_intermediate_layers
        self.batch_size = args.batch_size
        self.model = self.build_model()

    def build_model(self):
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        for param in model.parameters():
            param.requires_grad = True
        print("=====> successfully load pretrained imagenet weight")
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, self.num_classes)
        
        return model
