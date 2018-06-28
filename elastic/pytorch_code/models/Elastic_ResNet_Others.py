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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
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

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        # x1_out = nn.AvgPool2d(kernel_size=(56,56))(x1)
        # x1_out = Reshape(256).forward(x1_out)
        # x1_out = self.x1_out_linear(x1_out)
        # x1_out = nn.Linear(256, 10)(x1_out)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, [x1, x2, x3]
# ========================================================================================end of source code======================


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
    intermediate_layers_name = [
                                'layer1[0]', 'layer1[1]', 'layer1[2]','layer1[3]',
                                'layer2[0]', 'layer2[1]', 'layer2[2]','layer2[3]',
                                'layer3[0]', 'layer3[1]', 'layer3[2]','layer3[3]', 'layer3[4]','layer3[5]',
                                'layer4[0]', 'layer4[1]', 'layer4[2]'
                                ]

    intermediate_outputs = []
    # add_layers = [layer for layer in base_model.layers if layer.name in intermediate_layers_name]
    # add_layers = [model.layer for layer in intermediate_layers_name]
    # Add a classifier that belongs to the i'th block
    channels_in_last_layer = 0
    num_classes = 10
    # modules = CifarClassifier(channels_in_last_layer, num_classes)
    return intermediate_outputs


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule.cuda()
        self.extracted_layers= extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            module = module.cuda()
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs



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
    
    def train(self, input_var):
        # self.input = input_var
        output, intermediate_layer_outputs = self.model.forward(input_var)
        intermediate_outputs = []
        
        if self.add_intermediate_layers == 2:

            linear_FC = [(256, 10), (512, 10), (1024, 10)]
            pooling_kernels = [(56, 56), (28, 28), (14, 14)]

            inter_clf_block_1 = torch.nn.Sequential(torch.nn.Linear(linear_FC[0][0], linear_FC[0][1])).to(device)
            inter_clf_block_2 = torch.nn.Sequential(torch.nn.Linear(linear_FC[1][0], linear_FC[1][1])).to(device)
            inter_clf_block_3 = torch.nn.Sequential(torch.nn.Linear(linear_FC[2][0], linear_FC[2][1])).to(device)
            
            block_out_1 = nn.AvgPool2d(kernel_size=pooling_kernels[0])(intermediate_layer_outputs[0])
            block_out_2 = nn.AvgPool2d(kernel_size=pooling_kernels[1])(intermediate_layer_outputs[1])
            block_out_3 = nn.AvgPool2d(kernel_size=pooling_kernels[2])(intermediate_layer_outputs[2])

            block_out_1 = block_out_1.view(self.batch_size, 256)
            block_out_2 = block_out_2.view(self.batch_size, 512)
            block_out_3 = block_out_3.view(self.batch_size, 1024)
            
            block_out_1 = inter_clf_block_1(block_out_1)
            block_out_2 = inter_clf_block_2(block_out_2)
            block_out_3 = inter_clf_block_3(block_out_3)   

            intermediate_outputs = [block_out_1, block_out_2, block_out_3]

        elif self.add_intermediate_layers == 0:
            # return empty list when no intermediate layers classifiers
            intermediate_outputs = []
            
        else:
            print("Error, args.add_intermediate_layers_number should be 0 or 2")
            NotImplementedError

        all_ouputs = intermediate_outputs + [output]
        # return all intermediate plus final output
        return all_ouputs

        






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
    