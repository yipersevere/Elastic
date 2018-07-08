import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import scipy
from keras.datasets import cifar10
import numpy as np
from PIL import Image
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
        self.inplanes = planes * block.expansion
        self.intermediate_CLF.append(IntermediateClassifier(self.inplanes, 10))
        # print("blocks: ", 1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # print("blocks: ", i+1, "/", blocks, ", self.inplanes: ", self.inplanes, ", planes: ", planes)
            self.intermediate_CLF.append(IntermediateClassifier(self.inplanes, 10))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        i = 0
        intermediate_outputs = []
        print("=====> # of intermediate classifiers: ", len(self.intermediate_CLF), ", total classifiers: ", len(self.intermediate_CLF)+1)
        # make sure insert an intermediate classifier after each residul block 
        assert len(self.intermediate_CLF) == len(self.layer1)+len(self.layer2)+len(self.layer3)+len(self.layer4)

        for res_layer in self.layer1:
            x = res_layer(x)
            intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1

        for res_layer in self.layer2:
            x = res_layer(x)
            intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1

        for res_layer in self.layer3:
            x = res_layer(x)
            intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1                    
        
        for res_layer in self.layer4:
            x = res_layer(x)
            intermediate_outputs.append(self.intermediate_CLF[i](x))
            i += 1

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return [x]+intermediate_outputs


def Elastic_ResNet50(args):
    
    num_classes = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    batch_size = args.batch_size
    
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    for param in model.parameters():
        param.requires_grad = True
    # print("=====> successfully load pretrained imagenet weight")
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    
    return model


def plot_figs(epochs_train_accs, epochs_train_losses, test_accs, epochs_test_losses, args, captionStrDict):
    """
    plot epoch test error after model testing is finished
    """
    #folder = args.savedir
    #fig, (ax0, ax1, ax2, ax3) = plt.subplots(2, 2, sharey=True)
    fig, ax0 = plt.subplots(1, sharex=True)
    colormap = plt.cm.tab20

    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(test_accs[0]))])

    last = len(test_accs[0])-1
    # elastic_last = len(test_accs[0])-2

    for k in range(len(test_accs[0])):
        # Plots
        x = np.arange(len(test_accs)) + 1
        y = np.array(test_accs)[:, k]

        if k == last:
            c_label = captionStrDict["elastic_final_layer_label"]
        else:
            c_label = captionStrDict["elastic_intermediate_layer_label"] + str(k)

        ax0.plot(x, y, label=c_label)

        # Legends
        # y = test_accs[-1][k]
        # x = len(test_accs)
        # ax0.text(x, y, "%d" % k)
    
    ax0.set_ylabel(captionStrDict["y_label"])
    ax0.set_xlabel(captionStrDict["x_label"])
    ax0.set_title(captionStrDict["fig_title"])

    ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig_size = plt.rcParams["figure.figsize"]

    plt.rcParams["figure.figsize"] = fig_size
    plt.tight_layout()

    plt.savefig("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/test.png")
    plt.close("all")    





if __name__ == "__main__":
        
    test_accs = [[23.3050, 25.8475, 26.5575,26.6150,27.7475,29.8550,32.8700,40.0975,44.1325,46.9125,48.8200,49.7600,51.7275,56.4175,57.3300,57.2550,51.0400], [23.3050, 25.8475, 26.5575,26.6150,27.7475,29.8550,32.8700,40.0975,44.1325,46.9125,48.8200,49.7600,51.7275,56.4175,57.3300,57.2550,51.0400]]

    captionStrDict = {
        "save_file_name" : "Pytorch_CIFAR_10_train_test_epoch_Loss_Original_ResNet50.pdf",
        "fig_title" : "Pytorch_CIFAR_10_Original_ResNet50",
        "x_label" : "epochs",
        "y_label" : "sum loss",
        'elastic_final_layer_label': "train_loss",
        "elastic_intermediate_layer_label" : "Elastic_ResNet-152_Intermediate_Layer_Classifier_",
        "original_layer_label" : "test_loss"
    }
    
    plot_figs(None, None, test_accs, None, args, captionStrDict)




    # args.batch_size = 16

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # TUT thinkstation data folder path
    # data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/data"



    # narvi data folder path
    # data_folder = "/home/zhouy/Elastic/data"

    # XPS 15 laptop data folder path
    # data_folder = "D:\Elastic\data"

    # train_loader, val_loader = get_train_valid_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False,
    #                                                 random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
    #                                                 num_workers=1,pin_memory=True)
    # test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = (224,224,3),
    #                                 num_workers=1,pin_memory=True)



    # # model = resnet50(pretrained=True)
    # # model = inception_v3(pretrained=True)

    # # model = inception_v3(pretrained=True)

    # # model = Net()
    # # tar = torch.load('D:\Elastic\elastic\pytorch_code\models\pytorch-mobilenet\mobilenet_sgd_68.848.pth.tar')
    # # state_dict = tar["state_dict"]
    
    # # new_state_dict = OrderedDict()
    # # for k, v in state_dict.items():
    # #     name = k[7:] # remove `module.`
    # #     new_state_dict[name] = v
    # # model.load_state_dict(new_state_dict)
    # # print("not with loading pretraining weights")
    # model = Elastic_ResNet50(args)

    
    # # fc_features = model.fc.in_features

    # # model.fc = nn.Linear(fc_features, 10)






    # # x1_out = model.layer1[0].relu
    # # x2_out = model.layer1[1].relu

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
    # output = None
    # for i, (input, target) in enumerate(test_loader):
    #     target = target.cuda(async=True)
    #     input_var = torch.autograd.Variable(input)
    #     target_var = torch.autograd.Variable(target)
        

    #     # output = model(input_var)
    #     # loss = criterion(output, target_var)
      
    #     # print("loss: ", loss)
    #     # prec1 = accuracy(output.data, target)
    #     # print("precision: ", prec1)


    #     outputs = model(input_var)

    #     losses = 0
    #     for i in range(len(outputs)):
    #         loss = criterion(outputs[i], target_var)
    #         losses += loss
    #         print("loss: ", i, ": ", loss.item())
    #         prec1 = accuracy(outputs[i].data, target)
    #         print("precision_", i, ": ", prec1[0].data[0].item())
        
        

    #     # loss_1 = criterion(output[1], target_var)
    #     # loss_2 = criterion(output[2], target_var)
    #     # loss = loss_0 + loss_1 + loss_2
    #     # print("loss_0: ", loss_0,", loss_1: ", loss_1, ", loss_2: ", loss_2, ", loss: ", loss)
    #     # prec0 = accuracy(output[0].data, target)
    #     # prec1 = accuracy(output[1].data, target)
    #     # prec2 = accuracy(output[2].data, target)
    #     # print("precision_0: ", prec0, ", precision_1: ", prec1, ", precision_2: ", prec2)

    #     # # x1_out_1 = x1_out(input_var)
    #     # # block_out_1 = nn.AvgPool2d(kernel_size=(224, 224))(x1_out_1)
    #     # # block_out_1_1 = block_out_1.view(16, 3)
    #     # # inter_clf_block_1 = torch.nn.Linear(3, 10)(block_out_1_1)


    #     optimizer.zero_grad()
    #     losses.backward()
    #     optimizer.step()
    #     # summary(model, (3, 229, 229))


print("Done")


        


