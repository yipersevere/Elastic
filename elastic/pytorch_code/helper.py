from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import scipy

import datetime
from io import StringIO
import shutil
import random
import sys
import time
import sys
import csv


import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator


count_ops = 0
count_params = 0
module_number = 0
modules_flops = []
modules_params = []
to_print = False




label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

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

def LOG(message, logFile):
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(logFile, "a") as fp:
        fp.write(msg + "\n")

    print(msg)

def log_summary(model, logFile):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model.summary()

    sys.stdout = old_stdout
    summary = mystdout.getvalue()

    LOG("Model summary:", logFile)

    for line in summary.split("\n"):
        LOG(line, logFile) 


def log_error(labels, predictions):
    total = 0
    # print(predictions[0])
    # print(labels[0])
    for i in range(len(predictions)):
        if list(predictions[i]).index(max(list(predictions[i]))) ==  list(labels[i]).index(1):
            total += 1
    error = (1- total/len(labels)) * 100
    return error


def log_stats(path, epochs_acc_train, epochs_intermediate_acc_train, epochs_loss_train, epochs_lr, epochs_acc_test, epochs_intermediate_acc_test, epochs_loss_test):

    with open(path + os.sep + "train_accuracies.txt", "a") as fp:
        for a in epochs_acc_train:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "train_intermediate_accuracies.txt", "a") as fp:
        for a in epochs_intermediate_acc_train:
            fp.write("%.4f " % a)
        fp.write("\n")    

    with open(path + os.sep + "train_losses.txt", "a") as fp:
        for loss in epochs_loss_train:
            fp.write("%.4f " % loss)
        fp.write("\n")

    with open(path + os.sep + "epochs_lr.txt", "a") as fp:
        for a in epochs_lr:
            fp.write("%.4f " % a)
        fp.write("\n")    

    with open(path + os.sep + "test_accuracies.txt", "a") as fp:
        for a in epochs_acc_test:
            fp.write("%.4f " % a)
        fp.write("\n")

    with open(path + os.sep + "test_intermediate_accuracies.csv", "a") as fp:
        wr = csv.writer(fp)
        wr.writerows(epochs_intermediate_acc_test)
    
    with open(path + os.sep + "test_losses.txt", "a") as fp:
        for loss in epochs_loss_test:
            fp.write("%.4f " % loss)
        fp.write("\n")    



class Plot():

    def __init__(self, args):
        self.imageStr = args.imageStr
        self.errors = args.errors


    def plot(self):
        """
        plot test
        """
        # fig, ax = plt.subplots(1, sharex=True)
        fig, (ax0, ax1) = plt.subplots(1, 1, sharey=True)
        colormap = plt.cm.tab20

    # plot f1-score
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        for k in range(len(self.errors[0])):
            # Plots
            x = np.arange(len(self.errors)) + 1
            y = np.array(self.errors)[:, k]
            c_label = 'Layer ' + str(k)
            ax0.plot(x, y, label=c_label)

            # Legends
            y = self.errors[-1][k]
            x = len(self.errors)
            ax0.text(x, y, "%d" % k)
        
        ax0.set_ylabel(self.imageStr["ax0_set_ylabel"])
        ax0.set_xlabel('epoch')
        # title = "InceptionV3 test on CIFAR 100"
        # title2 = "ElasticNN-InceptionV3 test on CIFAR 100"
        ax0.set_title(self.imageStr["ax0_title"])
        ax0.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # plot f1-score
        # fig_f_score, ax_f_score = plt.subplots(1, sharex=True)
        # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        # for k in range(len(self.f_score[0])):
        #     # Plots
        #     x = np.arange(len(self.f_score)) + 1
        #     y = np.array(self.f_score)[:, k]
        #     c_label = 'Layer ' + str(k)
        #     ax1.plot(x, y, label=c_label)

        #     # Legends
        #     y = self.f_score[-1][k]
        #     x = len(self.f_score)
        #     ax1.text(x, y, "%d" % k)
        
        # ax1.set_ylabel(self.imageStr["ax1_set_ylabel"])
        # ax1.set_xlabel('epoch')
        # # title = "InceptionV3 test on CIFAR 100"
        # # title2 = "f1-score ElasticNN-InceptionV3 test on CIFAR 100"
        # ax1.set_title(self.imageStr["ax1_title"])
        # ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        fig_size = plt.rcParams["figure.figsize"]

        fig_size[0] = 6.4
        # fig_size[1] = 6.4

        plt.rcParams["figure.figsize"] = fig_size

        #plt.tight_layout()

        # imagecaption = "accuracy_InceptionV3_CIFAR_100.pdf"
        # imagecaption2 = "accuracy_Elastic_InceptionV3_CIFAR_100.pdf"
        plt.savefig(path + os.sep + self.imageStr["save_fig"], bbox_inches="tight")
        plt.close("all")

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params, module_number, modules_flops
    global modules_params, to_print
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    if to_print:
        print("")

    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)
        if hasattr(layer, 'shared'):
            delta_params = delta_params / int(layer.shared)
        module_number += 1
        modules_flops.append(delta_ops)
        modules_params.append(delta_params)
        if to_print:
            print(layer)
            print("Module number: ", module_number)
            print("FLOPS:", delta_ops)
            print("Parameter:", delta_params)

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)
        # module_number += 1
        # modules_flops.append(delta_ops)
        # to_print:
        #   print(layer)
        #   print("Module number: ", module_number)
        #   print("FLOPS:", delta_ops)


    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)
        # module_number += 1
        # modules_flops.append(delta_ops)
        # if to_print:
        #   print("Module number: ", module_number)
        #   print("FLOPS:", delta_ops)
        #   print("##Current params: ", count_params)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    layer.flops = delta_ops
    layer.params = delta_params
    return


def measure_model(model, H, W, debug=False):
    global count_ops, count_params, module_number, modules_flops
    global modules_params, to_print
    count_ops = 0
    count_params = 0
    module_number = 0
    modules_flops = []
    modules_params = []
    to_print = debug

    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    if to_print:
        print("modules flops sum: ", sum(modules_flops[0:2]))
    return count_ops, count_params
