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

from opts import args

from data_loader import get_train_valid_loader, get_test_loader
# from models import *
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, inception_v3, densenet121

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

    model = inception_v3(pretrained=True)

    fc_features = model.fc.in_features

    model.fc = nn.Linear(fc_features, 10)

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
    for i, (input, target) in enumerate(test_loader):
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)
        loss = criterion(output, target_var)
        

        # x1_out_1 = x1_out(input_var)
        # block_out_1 = nn.AvgPool2d(kernel_size=(224, 224))(x1_out_1)
        # block_out_1_1 = block_out_1.view(16, 3)
        # inter_clf_block_1 = torch.nn.Linear(3, 10)(block_out_1_1)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        summary(model, (3, 224, 224))


print("Done")


        

