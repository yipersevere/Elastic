import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from helper import LOG

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


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


num_outputs= 1

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
        # 这里的self.features 就是 make_layers 函数
        intermediate_outputs = []
        
        origin_0 = nn.Sequential(
            self.features[0],
            self.features[1],
            self.features[2],
            self.features[3],
            self.features[4],
            self.features[5],
            self.features[6]
        )        
        x = origin_0(x)
        intermediate_outputs.append(self.features[7](x))
        origin_1 = nn.Sequential(
            self.features[8],
            self.features[9],
            self.features[10],
            self.features[11],
            self.features[12],
            self.features[13],
            self.features[14]
        )
        x = origin_1(x)
        intermediate_outputs.append(self.features[15](x))

        origin_2 = nn.Sequential(
            self.features[16],
            self.features[17],
            self.features[18],
            self.features[19],
            self.features[20],
            self.features[21],
            self.features[22],
            self.features[23],
            self.features[24],
            self.features[25]
        )
        x = origin_2(x)
        intermediate_outputs.append(self.features[26](x))        

        origin_3 = nn.Sequential(
            self.features[27],
            self.features[28],
            self.features[29],
            self.features[30],
            self.features[31],
            self.features[32],
            self.features[33],
            self.features[34],
            self.features[35],
            self.features[36]
        )
        x = origin_3(x)
        intermediate_outputs.append(self.features[37](x))  

        origin_4 = nn.Sequential(
            self.features[38],
            self.features[39],
            self.features[40],
            self.features[41],
            self.features[42],
            self.features[43],
            self.features[44],
            self.features[45],
            self.features[46],
            self.features[47]
        )
        x = origin_4(x)
        # 最后一个intermediate classifier不能接
        # intermediate_outputs.append(self.features[48](x))  

        # 需要把self.features 拆开
        # x = self.features[]
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return intermediate_outputs+[x]

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
        self.classifier = nn.Sequential(nn.Linear(self.num_channels, self.num_classes)).to(self.device)

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
    for v, i in zip(cfg, range(len(cfg))):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # add intermediate classifier after pooling
            if i != (len(cfg)-1):
                print("v: ", v)
                print("in_channels: ", in_channels)
                layers += [IntermediateClassifier(in_channels, 100)]
                global num_outputs
                num_outputs += 1

            # 做一个修改，在最后一个maxpooling中不再接intermediate classifier

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


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def Elastic_VGG16_bn(args, logfile):
    num_classes = args.num_classes
    add_intermediate_layers = args.add_intermediate_layers
    pretrained_weight = args.pretrained_weight

    model = VGG(make_layers(cfg['D'], batch_norm=True))
    
    if pretrained_weight == 1:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
        LOG("loaded ImageNet pretrained weights", logfile)
        
    elif pretrained_weight == 0:
        LOG("not loading ImageNet pretrained weights", logfile)

    else:
        LOG("parameter--pretrained_weight, should be 0 or 1", logfile)
        NotImplementedError
    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)
    print("number of outputs: ", num_outputs)

    for param in model.parameters():
        param.requires_grad = False
    
    if add_intermediate_layers == 2:
        print("add any intermediate layer classifiers")    
        LOG("add intermediate layer classifiers", logfile)

        # get all extra classifiers params and final classifier params
        for inter_clf in model.intermediate_CLF:
            for param in inter_clf.parameters():
                param.requires_grad = True
        
        for param in model.classifier.parameters():
            param.requires_grad = True 
    
    elif add_intermediate_layers == 0:
        print("not adding any intermediate layer classifiers")    
        LOG("not adding any intermediate layer classifiers", logfile)

        for param in model.classifier.parameters():
            param.requires_grad = True         
    else:
        NotImplementedError

    return model, num_outputs    




