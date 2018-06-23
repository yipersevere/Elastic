import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.models.resnet
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from opts import args
from helper import load_data, multi_output_generator, LOG, log_summary, log_error, HistoryLogger

import os
import time
import datetime
import shutil
import sys
from torchsummary import summary

from Elastic_ResNet_Others import Elastic_ResNet18, Elastic_ResNet34, Elastic_ResNet50, Elastic_ResNet101
from utils import measure_model
from data_loader import get_train_valid_loader, get_test_loader
# Init Torch/Cuda
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
best_prec1 = 0
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def get_train_valid_set(data):
    train_set = 0
    valid_set = 0

    return train_set, valid_set

def main(**kwargs):
    global args, best_prec1
    # Override if needed
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)
    print(args)

    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    if args.data in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    program_start_time = time.time()
    instanceName = "Classification_Accuracy"
    folder_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + args.model

    imageStr = {
        "ax0_set_ylabel": "error rate on " + args.data,
        "ax0_title": args.model_name + " test on " + args.data,
        "ax1_set_ylabel": "f1 score on " + args.data,
        "ax1_title": "f1 score " + args.model_name+ " test on" + args.data,
        "save_fig" : args.model_name + "_" + args.data + ".png"
    }


    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + args.model_name + os.sep + ts_str
    
    tensorboard_folder = path + os.sep + "Graph"
    os.makedirs(path)
    args.savedir = path

    global logFile
    logFile = path + os.sep + "log.txt"    
    args.filename = logFile


    # if args.data == "cifar10":
    #     train_val_set = datasets.CIFAR10(data_folder, train=True, download=False,
    #                                  transform=None)
    #     train_set, val_set = get_train_valid_set(train_val_set)

    #     test_set = datasets.CIFAR10(data_folder, train=False, download=False,
    #                                transform=None)
    # elif args.data == "cifar100":
    #     train_val_set = datasets.CIFAR100(data_folder, train=True, download=False,
    #                                  transform=None)
    #     train_set, val_set = get_train_valid_set(train_val_set)

    #     test_set = datasets.CIFAR100(data_folder, train=False, download=False,
    #                                transform=None)
    # else:
    #     print("dataset is CIFAR10, or CIFAR100")
    #     raise NotImplementedError

    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # val_loader = torch.utils.data.DataLoader(
    #     val_set,
    #     batch_size=args.batch_size, shuffle=True, pin_memory=True)  

    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=args.batch_size, shuffle=False, pin_memory=True)


    # save input parameters into log file
    args_str = str(args)
    LOG(args_str, logFile)

    intermediate_outputs = list()

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
        # elasicNN_ResNet50, intermediate_outputs = Elastic_ResNet50()
        elasicNN_ResNet50 = Elastic_ResNet50()
        model = elasicNN_ResNet50
        print("using Elastic_ResNet50 class")

    # elif args.model == "Elastic_ResNet34":
    #     elasicNN_ResNet34 = Elastic_ResNet34(args)
    #     model = elasicNN_ResNet34
    #     print("using Elastic_ResNet34 class")

    # elif args.model == "Elastic_ResNet101":
    #     elasticNN_ResNet101 = Elastic_ResNet101(args)
    #     model = elasticNN_ResNet101
    #     print("using Elastic_ResNet101 class")

    else:
        print("--model parameter should be in [Elastic_ResNet18, Elastic_ResNet34, Elastic_ResNet101]")
        exit()    
    
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True  #这个是干什么的

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)# nesterov set False to keep align with keras default settting


    # Data loading
    data_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/data"

    train_loader, val_loader = get_train_valid_loader(data_dir=data_folder, batch_size=args.batch_size, augment=False,
                                                    random_seed=20180614, valid_size=0.2, shuffle=True,show_sample=True,
                                                    num_workers=1,pin_memory=True)
    test_loader = get_test_loader(data_dir=data_folder, batch_size=args.batch_size, shuffle=True,
                                    num_workers=1,pin_memory=True)






    
    for epoch in range(0, args.epochs):

        # Train for one epoch
        tr_prec1, tr_prec1_x1_out, loss, lr = train(train_loader, model, criterion, optimizer, epoch, intermediate_outputs)
        epoch_str = "==================================== epoch %d ==============================" % epoch
        print(epoch_str)
        epoch_result = "accuracy " + str(tr_prec1) + ", x1_out accuracy " + str(tr_prec1_x1_out) + ", loss " + str(loss) + ", learning rate " + str(lr) 
        print(epoch_result)
        LOG(epoch_str, logFile)
        LOG(epoch_result, logFile)
        
        # Evaluate on validation set
        val_prec1, test_x1_out_prec1 = validate(val_loader, model, criterion)
        val_str = "validation accuracy: " + str(val_prec1)
        print(val_str)
        LOG(val_str, logFile)
        # Remember best prec@1 and save checkpoint
        is_best = val_prec1 < best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        # save_checkpoint({
        #     'epoch': epoch,
        #     'model': args.model,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f\n" %(val_prec1, tr_prec1, loss, lr))

    # TestModel and return
    model = model.cpu().module
    model = nn.DataParallel(model).cuda()
    print(model)

    test_prec1, test_x1_out_prec1 = validate(test_loader, model, criterion)
    test_str = "test accuracy: " + str(test_prec1) + ", test x1_out_accuracy: " + str(test_x1_out_prec1)
    print(test_str)
    LOG(test_str, logFile)
    # n_flops, n_params = measure_model(model, IMAGE_SIZE, IMAGE_SIZE)
    # FLOPS_result = 'Finished training! FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6)
    # LOG(FLOPS_result, logFile)
    # print(FLOPS_result)
    print("============Finish============")
    print('Please run again with --resume --evaluate flags,'
          ' to evaluate the best model.')


def train(train_loader, model, criterion, optimizer, epoch, intermediate_outputs):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_x1_out = AverageMeter()
    model = model.cuda()
    # model = model.cpu()
    ### Switch to train mode
    model.train()
    running_lr = None
    # summary(model, (3, 224, 224))
    end = time.time()
    # LOG("============================================"+epoch+, logFile)
    LOG("=============================== train ===============================", logFile)
    for i, (input, target) in enumerate(train_loader):
        progress = float(epoch * len(train_loader) + i) / \
            (args.epochs * len(train_loader))
        args.progress = progress
        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader))
        if running_lr is None:
            running_lr = lr
        ### Measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        
        ### Compute output
        output, x1_out, inter_clf_x2, inter_clf_x3, inter_clf_x4 = model.forward(input_var)

        model_inter_clf_x1 = torch.nn.Sequential(
                torch.nn.Linear(256, 10)
                ).to(device)
        
        # inter_clf_x1 = inter_clf_x1.cuda()
        # inter_clf_x1 = torch.autograd.Variable(inter_clf_x1)
        # x1_out = torch.autograd.Variable(inter_clf_x1)
        x1_out = nn.AvgPool2d(kernel_size=(56,56))(x1_out)
        x1_out = x1_out.view(16,256)
        # x1_out = nn.AdaptiveAvgPool2d((16,256))(inter_clf_x1)
        
        # x1_out = x1_out.cuda()
        print("x1_out_shape: ", type(x1_out), x1_out.shape)# x1_out_shape:  <class 'torch.Tensor'> torch.Size([16, 256, 1, 1])
        print("1st element: ", x1_out[0][0].item()) #tensor([[ 0.2058]], device='cuda:0')
        # x1_out = x1_out.cuda()
        # x1_out = nn.Linear(256, 10)(x1_out)
        x1_out = model_inter_clf_x1(x1_out)
        print("x1_out_shape: ", type(x1_out), x1_out.shape)# x1_out_shape:  <class 'torch.Tensor'> torch.Size([16, 256, 1, 1])
        prec1_x1_out = accuracy(x1_out, target)
        print("top 1 precision, x1_out: ", prec1_x1_out)
        top1_x1_out.update(prec1_x1_out[0], input.size(0))
        # print("inter_clf_x1", inter_clf_x1)
        # print("inter_clf_x1_shape: ", type(inter_clf_x1), inter_clf_x1.shape) # <class 'torch.Tensor'> torch.Size([16, 256, 56, 56])
        # print("inter_clf_x1", inter_clf_x1)
        # inter_outputs = intermediate_outputs(input_var)
        # for layer in intermediate_outputs:
        # inter_outputs.(layer(input_var))
        print("original final layer classifier: type, shape ", type(output), output.shape)
        print("1st element in final layer clf: ", output[0][0].item()) #tensor([[ 0.2058]], device='cuda:0')
        if args.add_intermediate_layers_number == 2:
            # loss = all_intermediate_layers_losses(output, target_var, criterion)
            loss_final_clf = criterion(x1_out, target_var)
            loss_x1_out = criterion(output, target_var)
            loss = loss_final_clf + loss_x1_out
        elif args.add_intermediate_layers_number == 0:
            loss = criterion(output, target_var)
        
        ### Measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        print("top 1 precision, final output classification: ", prec1)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            temp_result = 'Epoch: [{0}][{1}/{2}]\t' \
            'Time {batch_time.val:.3f}\t' \
            'Data {data_time.val:.3f}\t'  \
            'Loss {loss.val:.4f}\t'  \
            'Prec@1 {top1.val:.3f}\t' \
            'Prec_x1_out@1 {top1_x1_out.val:.3f}\t' \
            'lr {lr: .4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top1_x1_out=top1_x1_out, lr=lr)

            LOG(temp_result, logFile)
            print(temp_result)

    return 100. - top1.avg, 100. - top1_x1_out.avg, losses.avg, running_lr

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    x1_out_losses = AverageMeter()
    top1 = AverageMeter()
    x1_out_top1 = AverageMeter()
    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    LOG("=============================== validate ===============================", logFile)
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        ### Compute output
        # output = model(input_var)

        output, x1_out, inter_clf_x2, inter_clf_x3, inter_clf_x4 = model.forward(input_var)

        model_inter_clf_x1 = torch.nn.Sequential(
                torch.nn.Linear(256, 10)
                ).to(device)
        
        # inter_clf_x1 = inter_clf_x1.cuda()
        # inter_clf_x1 = torch.autograd.Variable(inter_clf_x1)
        # x1_out = torch.autograd.Variable(inter_clf_x1)
        x1_out = nn.AvgPool2d(kernel_size=(56,56))(x1_out)
        x1_out = x1_out.view(16,256)
        # x1_out = nn.AdaptiveAvgPool2d((16,256))(inter_clf_x1)
        
        # x1_out = x1_out.cuda()
        # print("x1_out_shape: ", type(x1_out), x1_out.shape)# x1_out_shape:  <class 'torch.Tensor'> torch.Size([16, 256, 1, 1])
        # print("1st element: ", x1_out[0][0].item()) #tensor([[ 0.2058]], device='cuda:0')
        # x1_out = x1_out.cuda()
        # x1_out = nn.Linear(256, 10)(x1_out)
        x1_out = model_inter_clf_x1(x1_out)
        # print("x1_out_shape: ", type(x1_out), x1_out.shape)# x1_out_shape:  <class 'torch.Tensor'> torch.Size([16, 256, 1, 1])
        
        test_loss_x1_out = criterion(x1_out, target_var)
        test_prec1_x1_out = accuracy(x1_out.data, target)

        x1_out_losses.update(test_loss_x1_out.data[0], input.size(0))
        x1_out_top1.update(test_prec1_x1_out[0], input.size(0))


        loss = criterion(output, target_var)
        

        ### Measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            temp_result = 'Test: [{0}/{1}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'x1_out Loss {x1_out_losses.val:.4f} ({x1_out_losses.avg:.4f})\t' \
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'x1_out_Prec@1 {x1_out_top1.val:.3f} ({x1_out_top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, x1_out_losses=x1_out_losses,
                      top1=top1, x1_out_top1=x1_out_top1)
            LOG(temp_result, logFile)

            print(temp_result)
    prec1_result = ' * Prec@1 {top1.avg:.3f}'.format(top1=top1)
    x1_out_prec1_result = ' * x1_out_Prec@1 {x1_out_top1.avg:.3f}'.format(x1_out_top1=x1_out_top1)
    LOG(prec1_result, logFile)
    LOG(x1_out_prec1_result, logFile)
    print(prec1_result)
    print(x1_out_prec1_result)


    return 100. - top1.avg, x1_out_top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):

    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, 'save_models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
        os.makedirs(model_dir)

    # For mkdir -p when using python3
    # os.makedirs(args.savedir, exist_ok=True)
    # os.makedirs(model_dir, exist_ok=True)

    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def all_intermediate_layers_losses(intermediate_layers_loss, final_output_loss, target_var, criterion):
    losses = []
    print("all intermediate layers loss---the total number of intermediate layer output : ", len(output))
    for out in range(0, len(output)):
        losses.append(criterion(output[out], target_var))
    print("all losses: ", losses)
    loss_weight = [1] * len(output)
    all_loss = sum([a*b for a,b in zip(loss_weight, losses)])
    print("total loss: ", all_loss)
    return all_loss



if __name__ == "__main__":

    main()
