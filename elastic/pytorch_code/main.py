
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.resnet
import torch.backends.cudnn as cudnn
from ignite.handlers import EarlyStopping
from torchsummary import summary
from tensorboardX import SummaryWriter

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


# num_outputs = 1

def validate(val_loader, model, criterion):
    # batch_time = AverageMeter()
    model.eval()
    all_acc = []
    all_loss = []
    for ix in range(num_outputs):
        all_loss.append(AverageMeter())
        all_acc.append(AverageMeter())
    # end = time.time()
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)     
        
        losses = 0
        
        outputs = model(input_var)
        with torch.no_grad():
            for ix in range(len(outputs)):
                loss = criterion(outputs[ix], target_var)
                all_loss[ix].update(loss.item(), input.size(0))

                losses += loss
                # print("loss: ", i, ": ", loss.item())
                prec1 = accuracy(outputs[ix].data, target)
                all_acc[ix].update(prec1[0].data[0].item(), input.size(0))
                # print("precision_", i, ": ", prec1[0].data[0].item())

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
        accs.append(float(100-i.avg))
        ls.append(j.avg)
        
    return accs, ls



def train(train_loader, model, criterion, optimizer, epoch):
    # batch_time = AverageMeter()
    # data_time = AverageMeter()

    model.train()

    # end = time.time()
    lr = None
    all_acc = []
    all_loss = []

    for ix in range(num_outputs):
        all_loss.append(AverageMeter())
        all_acc.append(AverageMeter())
    

    LOG("==> train ", logFile)
    for i, (input, target) in enumerate(train_loader):
        # lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader))

        # data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        losses = 0
        optimizer.zero_grad()

        outputs = model(input_var)

        
        # 这里应该要再封装一下， 变成只有一个变量loss
        for ix in range(len(outputs)):
            loss = criterion(outputs[ix], target_var)
            all_loss[ix].update(loss.item(), input.size(0))

            losses += loss
            # print("loss: ", i, ": ", loss.item())
            prec1 = accuracy(outputs[ix].data, target)
            all_acc[ix].update(prec1[0].data[0].item(), input.size(0))
            # print("precision_", i, ": ", prec1[0].data[0].item())
        
        losses.backward()
        optimizer.step()

        ### Measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()
        
    accs = []
    ls = []
    for i, j in zip(all_acc, all_loss):
        accs.append(float(100-i.avg))
        ls.append(j.avg)

    try:
        lr = str(optimizer).split("\n")[-5].split(" ")[-1]
    except:
        lr = -1
    return accs, ls, lr


def main(**kwargs):
    global args
    lowest_error1 = 100
    
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

    writer = SummaryWriter(tensorboard_folder)

    global logFile
    logFile = path + os.sep + "log.txt"    
    args.filename = logFile
    global num_outputs
    # num_outputs = 1
    
    print(args)
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.data == "cifar100" or args.data == "CIFAR100":
        fig_title_str = " on CIFAR-100"

    elif args.data == "cifar10" or args.data == "CIFAR10":
        fig_title_str = " on CIFAR-10"
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


    # if args.layers_weight_change == 1:
    #     LOG("weights for intermediate layers: 1/(34-Depth), giving different weights for different intermediate layers output, using the formula weigh = 1/(34-Depth)", logFile)
    # elif args.layers_weight_change == 0:
    #     LOG("weights for intermediate layers: 1, giving same weights for different intermediate layers output as  1", logFile)
    # else:
    #     print("Parameter --layers_weight_change, Error")
    #     sys.exit()   
    
    if args.model == "Elastic_ResNet18" or args.model == "Elastic_ResNet34" or args.model == "Elastic_ResNet50" or args.model == "Elastic_ResNet101" or args.model == "Elastic_ResNet152":
        model = Elastic_ResNet(args, logFile)
        num_outputs = model.num_outputs
        print("num_outputs: ", num_outputs)
        # num_outputs = model_num_outputs
        print("successfully create model: ", args.model)

    elif args.model == "Elastic_InceptionV3":
        args.target_size = (229, 229, 3) # since pytorch inceptionv3 pretrained accepts image size (229, 229, 3) instead of (224, 224, 3)
        model = Elastic_InceptionV3(args, logFile)
        num_outputs = model.num_outputs
        print("num_outputs: ", num_outputs)
        print("successfully create model: ", args.model)

    else:
        print("--model parameter should be in [Elastic_ResNet18, Elastic_ResNet34, Elastic_ResNet101]")
        exit()    

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

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
    
    
    criterion = nn.CrossEntropyLoss().cuda()

    pretrain_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.pretrain_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)# nesterov set False to keep align with keras default settting

    print("==> Pretraining for 10 epoches    ")
    LOG("==> Pretraining for 10 epoches    \n", logFile)
    for pretrain_epoch in range(0, 1):
        accs, losses, lr = train(train_loader, model, criterion, pretrain_optimizer, pretrain_epoch)
        epoch_result = "    pretrain epoch: " + str(pretrain_epoch) + ", pretrain error: " + str(accs) + ", pretrain loss: " + str(losses) + ", pretrain learning rate: " + str(lr) + ", pretrain total train sum loss: " + str(sum(losses))
        print(epoch_result)
        LOG(epoch_result, logFile)
        
    
    print("==> Full training ")
    LOG("==> Full training    \n", logFile)
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)# nesterov set False to keep align with keras default settting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-4, patience=10)
    
    # implement early stop by own
    EarlyStopping_epoch_count = 0
    prev_val_loss = 100000

    epochs_train_accs = []
    epochs_train_losses = []
    epochs_val_accs = []
    epochs_val_losses = []    
    epochs_test_accs = []
    epochs_test_losses = []
    epochs_lr = []

    for epoch in range(0, args.epochs):
        

        epoch_str = "==================================== epoch %d ==============================" % epoch
        print(epoch_str)
        LOG(epoch_str, logFile)
        # Train for one epoch
        accs, losses, lr = train(train_loader, model, criterion, optimizer, epoch)
        epochs_train_accs.append(accs)
        epochs_train_losses.append(losses)
        epochs_lr.append(lr)


        writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'lr', lr, epoch)
        for i, a, l in zip(range(len(accs)), accs, losses):
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'train_error_' + str(i), a, epoch)
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'train_losses_' + str(i), l, epoch)
        
    
        epoch_result = "train error: " + str(accs) + ", loss: " + str(losses) + ", learning rate " + str(lr) + ", total train sum loss " + str(sum(losses))
        print(epoch_result)
        LOG(epoch_result, logFile)

        if num_outputs > 1:
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'train_total_sum_losses', sum(losses), epoch) 
            losses.append(sum(losses)) # add the total sum loss
            print("train_total_sum_losses: ", sum(losses))       
        
        # Evaluate on validation set
        LOG("==> validate \n", logFile)
        val_accs, val_losses = validate(val_loader, model, criterion)
        epochs_val_accs.append(val_accs)
        epochs_val_losses.append(val_losses)

        for i, a, l in zip(range(len(val_accs)), val_accs, val_losses):
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'val_error_' + str(i), a, epoch)
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'val_losses_' + str(i), l, epoch)
        

        val_str = "val_error: " + str(val_accs) + ", val_loss" + str(val_losses)
        print(val_str)
        LOG(val_str, logFile)

        if num_outputs > 1:
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'val_total_sum_losses', sum(val_losses), epoch) 
            val_losses.append(sum(val_losses)) # add the total sum loss
            print("val_total_sum_losses: ", sum(val_losses))       
                
        
        # run on test dataset
        LOG("==> test \n", logFile)
        test_accs, test_losses = validate(test_loader, model, criterion)
        
        epochs_test_accs.append(test_accs)
        epochs_test_losses.append(test_losses)

        for i, a, l in zip(range(len(test_accs)), test_accs, test_losses):
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'test_error_' + str(i), a, epoch)
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'test_losses_' + str(i), l, epoch)


        test_result_str = "==> Test epoch, final output classifier error: " + str(test_accs) + ", test_loss" +str(test_losses) + ", total test sum loss " + str(sum(test_losses))
        print(test_result_str)
        LOG(test_result_str, logFile)

        if num_outputs > 1:
            writer.add_scalar(tensorboard_folder + os.sep + "data" + os.sep + 'test_total_sum_losses', sum(test_losses), epoch) 
            test_losses.append(sum(test_losses)) # add the total sum loss
            print("test_total_sum_losses: ", sum(test_losses))   

        
        log_stats(path, accs, losses, lr, test_accs, test_losses)

        # Remember best prec@1 and save checkpoint
        is_best = test_accs[-1] < lowest_error1 #error not accuracy, but i don't want to change variable names
        lowest_error1 = test_accs[-1]  #但是有个问题，有时是倒数第二个CLF取得更好的结果
        
        if is_best:
            save_checkpoint({
                'epoch': epoch,
                'model': args.model_name,
                'state_dict': model.state_dict(),
                'best_prec1': lowest_error1,
                'optimizer': optimizer.state_dict(),
            }, args)

        # apply early_stop with monitoring val_loss
        # EarlyStopping(patience=15, score_function=score_function(val_loss), trainer=model)
        total_loss = sum(test_losses)

        scheduler.step(total_loss) # adjust learning rate with test_loss
        
        if epoch == 0:
            prev_epoch_loss = total_loss # use all intemediate classifiers sum loss instead of only one classifier loss
        else:
            if total_loss >= prev_epoch_loss: # means this current epoch doesn't reduce test losses
                EarlyStopping_epoch_count += 1
        if EarlyStopping_epoch_count > 20:
            print("No improving test_loss for more than 10 epochs, stop running model")
            break

    # n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    # FLOPS_result = 'Finished training! FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6)
    # LOG(FLOPS_result, logFile)
    # print(FLOPS_result)
    writer.close()

    end_timestamp = datetime.datetime.now()
    end_ts_str = end_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    LOG("program end time: " + end_ts_str +"\n", logFile)

    # here plot figures
    plot_figs(epochs_train_accs, epochs_train_losses, epochs_test_accs, epochs_test_losses, args, captionStrDict)
    print("============Finish============")

if __name__ == "__main__":

    main()