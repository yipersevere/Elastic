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
from helper import LOG, log_summary, log_stats, AverageMeter, accuracy, save_checkpoint, adjust_learning_rate, plot_figs
from data_loader import get_train_valid_loader, get_test_loader
from models import *

# Init Torch/Cuda
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
best_prec1 = 0


def train(train_loader, model, criterion, optimizer, epoch):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()

    model.train()

    # end = time.time()
    lr = None
    all_acc = []
    all_loss = []

    for ix in range(17):
        all_loss.append(AverageMeter())
        all_acc.append(AverageMeter())
    
    LOG("==> train ", logFile)
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader))

        # data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        outputs = model(input_var)

        losses = 0
# 这里应该要再封装一下， 变成只有一个变量loss
        for ix in range(len(outputs)):
            loss = criterion(outputs[ix], target_var)
            all_loss[ix].update(loss.item(), input.size(0))

            losses += loss
            # print("loss: ", i, ": ", loss.item())
            prec1 = accuracy(outputs[ix].data, target)
            all_acc[ix].update(prec1[0].data[0].item(), input.size(0))
            # print("precision_", i, ": ", prec1[0].data[0].item())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        ### Measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        
    accs = []
    ls = []
    for i, j in zip(all_acc, all_loss):
        accs.append(i.avg)
        ls.append(j.avg)
    # 这里的avg loss 和avg acc 是一张图片分类的平均loss 
    return accs, ls, lr


def validate(val_loader, model, criterion):
    # batch_time = AverageMeter()
    model.eval()
    all_acc = []
    all_loss = []
    for ix in range(17):
        all_loss.append(AverageMeter())
        all_acc.append(AverageMeter())
    # end = time.time()
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)     
        
        outputs = model(input_var)

        losses = 0

        for ix in range(len(outputs)):
            loss = criterion(outputs[ix], target_var)
            all_loss[ix].update(loss.item(), input.size(0))

            losses += loss
            # print("loss: ", i, ": ", loss.item())
            prec1 = accuracy(outputs[ix].data, target)
            all_acc[ix].update(prec1[0].data[0].item(), input.size(0))
            # print("precision_", i, ": ", prec1[0].data[0].item())
    
        # if args.add_intermediate_layers_number == 2:

        # elif args.add_intermediate_layers_number == 0:
        #     output = model.forward(input_var)
        #     loss = criterion(output, target_var)
        # else:
        #     print("Error, args.add_intermediate_layers_number should be 0 or 2")
        #     NotImplementedError

    #     ### Measure elapsed time
    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     # if i % args.print_freq == 0:
    #     #     temp_result = 'Test: [{0}/{1}]\t' \
    #     #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #     #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
    #     #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
    #     #           'x1_out_Prec@1 {x1_out_top1.val:.3f} ({x1_out_top1.avg:.3f})'.format(
    #     #               i, len(val_loader), batch_time=batch_time, loss=losses,
    #     #               top1=top1, x1_out_top1=x1_out_top1)
    #     #     LOG(temp_result, logFile)

    #     #     print(temp_result)

    accs = []
    ls = []
    for i, j in zip(all_acc, all_loss):
        accs.append(i.avg)
        ls.append(j.avg)
        
    return accs, ls



def main(**kwargs):
    global args, best_prec1
    
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)

    program_start_time = time.time()
    instanceName = "Classification_Accuracy"
    folder_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + args.model
    
    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + args.model_name + os.sep + ts_str
    
    tensorboard_folder = path + os.sep + "Graph"
    os.makedirs(path)
    args.savedir = path

    global logFile
    logFile = path + os.sep + "log.txt"    
    args.filename = logFile
    
    print(args)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data == "cifar100" or args.data == "CIFAR100":
        fig_title_str = "classification error on CIFAR-100"

    elif args.data == "cifar10" or args.data == "CIFAR10":
        fig_title_str = "classification error on CIFAR-10"
    else:
        print("ERROR =============================dataset should be CIFAR10 or CIFAR100")
        NotImplementedError

    captionStrDict = {
        "fig_title" : fig_title_str,
        "x_label" : "epoch",
        'elastic_final_layer_label': "Final_Layer_Output_Classifier",
        "elastic_intermediate_layer_label" : "Intermediate_Layer_Classifier_"
    }

    # save input parameters into log file
    args_str = str(args)
    LOG(args_str, logFile)
    LOG("program start time: " + ts_str +"\n", logFile)


    if args.layers_weight_change == 1:
        LOG("weights for intermediate layers: 1/(34-Depth), giving different weights for different intermediate layers output, using the formula weigh = 1/(34-Depth)", logFile)
    elif args.layers_weight_change == 0:
        LOG("weights for intermediate layers: 1, giving same weights for different intermediate layers output as  1", logFile)
    else:
        print("Parameter --layers_weight_change, Error")
        sys.exit()   
    
    if args.model == "Elastic_ResNet18":
        elasicNN_ResNet18 = Elastic_ResNet18()
        model = elasicNN_ResNet18
        print("using Elastic_ResNet18 class")

    elif args.model == "Elastic_ResNet50":

        model = Elastic_ResNet50(args, logFile)
        print("using Elastic_ResNet50 class")

    # elif args.model == "Elastic_ResNet34":
    #     elasicNN_ResNet34 = Elastic_ResNet34(args)
    #     model = elasicNN_ResNet34
    #     print("using Elastic_ResNet34 class")

    # elif args.model == "Elastic_ResNet101":
    #     elasticNN_ResNet101 = Elastic_ResNet101(args)
    #     model = elasticNN_ResNet101
    #     print("using Elastic_ResNet101 class")
    elif args.model == "Elastic_InceptionV3":
        args.target_size = (229, 229, 3) # since pytorch inceptionv3 pretrained accepts image size (229, 229, 3) instead of (224, 224, 3)
        elasticNN_inceptionV3 = Elastic_InceptionV3(args)
        model = elasticNN_inceptionV3.model
    else:
        print("--model parameter should be in [Elastic_ResNet18, Elastic_ResNet34, Elastic_ResNet101]")
        exit()    
    
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    # implement early stop by own
    EarlyStopping_epoch_count = 0
    prev_val_loss = 100000

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)# nesterov set False to keep align with keras default settting
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # TUT thinkstation data folder path
    data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/data"

    # narvi data folder path
    # data_folder = "/home/zhouy/Elastic/data"

    # XPS 15 laptop data folder path
    # data_folder = "D:\Elastic\data"
    # args.batch_size = 1

    train_loader, val_loader = get_train_valid_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, augment=False, target_size = args.target_size,
                                                    random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=False,
                                                    num_workers=4,pin_memory=True)
    test_loader = get_test_loader(args.data, data_dir=data_folder, batch_size=args.batch_size, shuffle=True, target_size = args.target_size,
                                    num_workers=4,pin_memory=True)
    
    # EarlyStopping(patience=15, )

    epochs_train_accs = []
    epochs_train_losses = []
    epochs_val_accs = []
    epochs_val_losses = []    
    epochs_test_accs = []
    epochs_test_losses = []
    epochs_lr = []

    for epoch in range(0, args.epochs):
        
        # Train for one epoch
        accs, losses, lr = train(train_loader, model, criterion, optimizer, epoch)
        epochs_train_accs.append(accs)
        epochs_train_losses.append(losses)
        epochs_lr.append(lr)

        epoch_str = "==================================== epoch %d ==============================" % epoch
        print(epoch_str)
        epoch_result = "accuracy: " + str(accs) + ", loss: " + str(losses) + ", learning rate " + str(lr) 
        print(epoch_result)
        
        LOG(epoch_str, logFile)
        LOG(epoch_result, logFile)
        
        # Evaluate on validation set
        LOG("==> validate \n", logFile)
        val_accs, val_losses = validate(val_loader, model, criterion)
        epochs_val_accs.append(val_accs)
        epochs_val_losses.append(val_losses)

        val_str = "val_accuracy: " + str(val_accs) + ", val_loss" + str(val_losses)
        print(val_str)
        LOG(val_str, logFile)
        
        

        # Remember best prec@1 and save checkpoint
        # is_best = val_accs < best_prec1
        # best_prec1 = max(val_prec1, best_prec1)
        # model_filename = 'checkpoint_%03d.pth.tar' % epoch
        # save_checkpoint({
        #     'epoch': epoch,
        #     'model': args.model_name,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'intermediate_layer_classifier': val_x1_out_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f\n" %(val_prec1, tr_prec1_x1_out, tr_prec1, val_x1_out_prec1, loss, lr))

        # run on test dataset
        LOG("==> test \n", logFile)
        test_accs, test_losses = validate(test_loader, model, criterion)
        
        scheduler.step(test_losses[-1])
        
        epochs_test_accs.append(test_accs)
        epochs_test_losses.append(test_losses)

        test_result_str = "==> Test epoch, final output classifier acc: " + str(test_accs) + ", test_loss" +str(test_losses)

        print(test_result_str)
        LOG(test_result_str, logFile)
        log_stats(path, accs, losses, lr, test_accs, test_losses)
        # apply early_stop with monitoring val_loss
        # EarlyStopping(patience=15, score_function=score_function(val_loss), trainer=model)
        if epoch == 0:
            prev_test_loss = test_losses[-1]
        else:
            if test_losses[-1] >= prev_test_loss:
                EarlyStopping_epoch_count += 1
        if EarlyStopping_epoch_count > 10:
            print("No improving test_loss for more than 10 epochs, stop running model")
            break

    # n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    # FLOPS_result = 'Finished training! FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6)
    # LOG(FLOPS_result, logFile)
    # print(FLOPS_result)
    
    end_timestamp = datetime.datetime.now()
    end_ts_str = end_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    LOG("program end time: " + end_ts_str +"\n", logFile)

    # here plot figures
    plot_figs(epochs_train_accs, epochs_train_losses, epochs_test_accs, epochs_test_losses, args, captionStrDict)
    print("============Finish============")

if __name__ == "__main__":

    main()
