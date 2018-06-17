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

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

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
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print("x shape: ", x.shape)
        x = self.avgpool(x)
        print("x shape: ", x.shape)
        x = x.view(x.size(0), -1)

        # ===============================================因为在这里开始将（16,2048）变成（16， 10）
        print("x shape: ", x.shape)
        x = self.fc(x)
        print("x shape: ", x.shape)

        return x

class CifarClassifier(nn.Module):

    def __init__(self, num_channels, num_classes):
        """
        Classifier of a cifar10/100 image.

        :param num_channels: Number of input channels to the classifier
        :param num_classes: Number of classes to classify
        """

        super(CifarClassifier, self).__init__()
        self.inner_channels = 128

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm2d(self.inner_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )

        self.classifier = nn.Linear(self.inner_channels, num_classes)

    def forward(self, x):
        """
        Drive features to classification.

        :param x: Input of the lowest scale of the last layer of
                  the last block
        :return: Cifar object classification result
        """

        x = self.features(x)
        x = x.view(x.size(0), self.inner_channels)
        x = self.classifier(x)
        return x


def Elastic_ResNet18():
    # model = ResNet(BasicBlock, [2,2,2,2])
    # # load pretrained imagenet weigt
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model = resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True
    
    #提取fc层中固定的参数
    num_classes = 10

    print("already loaded pretrained imagenet weight")
    fc_features = model.fc.in_features
    # model.avgpool = nn.AvgPool2d(4, stride=0)

    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Elastic_ResNet34():
    model = resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True
    
    #提取fc层中固定的参数
    num_classes = 10

    print("already loaded pretrained imagenet weight")
    fc_features = model.fc.in_features
    # model.avgpool = nn.AvgPool2d(4, stride=0)

    model.fc = nn.Linear(fc_features, num_classes)
    return model


def add_intermediate_layers(flag_num_intermediate_layers, base_model):

    intermediate_outputs = []
    # Add a classifier that belongs to the i'th block
    modules[i] = CifarClassifier(channels_in_last_layer, self.num_classes)
    return intermediate_outputs

def Elastic_ResNet50():

    model = resnet50(pretrained=True)
    
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    for param in model.parameters():
        param.requires_grad = True
    #提取fc层中固定的参数
    num_classes = 10

    print("already loaded pretrained imagenet weight")
    fc_features = model.fc.in_features
    # model.avgpool = nn.AvgPool2d(4, stride=0)

    model.fc = nn.Linear(fc_features, num_classes)

    add_intermediate_layers_number = 2
    add_intermediate_layers(add_intermediate_layers_number, model)



    return model



def Elastic_ResNet101():
    model = resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    #提取fc层中固定的参数
    num_classes = 10

    print("already loaded pretrained imagenet weight")
    fc_features = model.fc.in_features
    # model.avgpool = nn.AvgPool2d(4, stride=0)

    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Elastic_ResNet152():
    model = resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    #提取fc层中固定的参数
    num_classes = 10

    print("already loaded pretrained imagenet weight")
    fc_features = model.fc.in_features
    # model.avgpool = nn.AvgPool2d(4, stride=0)

    model.fc = nn.Linear(fc_features, num_classes)
    return model
    