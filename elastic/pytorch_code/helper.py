
import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import scipy

import keras
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, ModelCheckpoint, \
    LearningRateScheduler
from keras.utils import plot_model
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar100, cifar10
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_absolute_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score

import datetime
from io import StringIO
import shutil
import random
import sys
import traceback
import time
import sys



def load_data(data = "cifar100", target_size = (224,224,3), num_class = 100, test_percent=0.2, shuffle=True):
    """"
    @datafile, default, cifar100
    @num_class default, 100
    @target_size: default, (224,224,3)
    return train, val, test
    """
    # x_train shape is (50000, 32,32,3); y_train shape is (50000, 1)
    # (x_data, y_data), (x_test, y_test) = cifar100.load_data()
    if num_class == 10 and data=="cifar10":
        (x_data, y_data), (x_test, y_test) = cifar10.load_data()
    elif num_class == 100 and data=="cifar100":
        (x_data, y_data), (x_test, y_test) = cifar100.load_data()
    else:
        print("num class is not 10 or 100 either, ERROR")
        sys.exit()
    
    # train_data[b'data'][0] 的长度是3072 也就是3072 = 3 × 32 × 32， 也就是说cifar100的照片尺寸是32 × 32；
    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=test_percent)

    if shuffle:
        train_shuffle_indices = np.random.permutation(np.arange(len(y_data)))
        x_data = x_data[train_shuffle_indices]
        y_data = y_data[train_shuffle_indices]
        test_shuffle_indices = np.random.permutation(np.arange(len(y_test)))
        x_test = x_test[test_shuffle_indices]
        y_test = y_test[test_shuffle_indices]


    y_train = np_utils.to_categorical(y_train, num_class)
    y_val = np_utils.to_categorical(y_val, num_class)

    big_x_train = np.array([scipy.misc.imresize(X_train[i], target_size) for i in range(0, len(X_train))]).astype('float32')
    X_train = preprocess_input(big_x_train)
    big_x_val = np.array([scipy.misc.imresize(X_val[i], target_size) for i in range(0, len(X_val))]).astype('float32')
    X_val = preprocess_input(big_x_val)

    big_x_test = np.array([scipy.misc.imresize(x_test[i], target_size) for i in range(0, len(x_test))]).astype('float32')
    x_test = preprocess_input(big_x_test)
    y_test = np_utils.to_categorical(y_test, num_class)

    return X_train, y_train, X_val, y_val, x_test, y_test


def multi_output_generator(generator, num_outputs):
    """ Generator that yields both the age target and the cumulative vector target. """
    while True:  # keras requires all generators to be infinite

        # 这个data的shape是[32*224*224*3],也就是说是32张图片一次批处理操作
        data = generator.__next__()
    # data[0][0] 是一张照片, 也就是 224*224*3
    # data[0] 是32张照片

        image = data[0]
        # age        = np.argmax(data[1], axis = 1)
        # cumulative = np.cumsum(data[1], axis = 1)[:, :-1]
    #data[1][0] 是一张照片的类别，也就是1*100 维度， 其中使用的是 0， 1 编码
    #data[1]是32张照片的分别类别， 也就是 32×101 维度
    #[data[1]] * num_outputs = 12 * 32 * 101
        yield (image, [data[1]] * num_outputs)


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

def get_f1_score(labels, predictions):
    p_indexs = []
    y_indexs = []
    
    for i in range(len(predictions)):
        p_index = list(predictions[i]).index(max(list(predictions[i])))
        y_index = list(labels[i]).index(1)

        p_indexs.append(p_index)
        y_indexs.append(y_index)
    
    score = f1_score(y_indexs, p_indexs, average='micro')
    # scale to [0, 100]
    score = score * 100
    return score

class HistoryLogger(Callback):
    def __init__(self, optimizer, path, batch_size, epoch_size, X, y, num_outputs, train_generator, logFile, imageStr):

        self.path = path
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.X = X
        self.y = y
        self.num_outputs = num_outputs
        self.train_generator = train_generator
        self.logFile = logFile

        self.errors = []
        self.f_score = []
        self.imageStr = imageStr
        super().__init__()


    def plot(self):
        """
        plot test
        """
        # fig, ax = plt.subplots(1, sharex=True)
        fig, (ax0, ax1) = plt.subplots(2, 1, sharey=True)
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

        for k in range(len(self.f_score[0])):
            # Plots
            x = np.arange(len(self.f_score)) + 1
            y = np.array(self.f_score)[:, k]
            c_label = 'Layer ' + str(k)
            ax1.plot(x, y, label=c_label)

            # Legends
            y = self.f_score[-1][k]
            x = len(self.f_score)
            ax1.text(x, y, "%d" % k)
        
        ax1.set_ylabel(self.imageStr["ax1_set_ylabel"])
        ax1.set_xlabel('epoch')
        # title = "InceptionV3 test on CIFAR 100"
        # title2 = "f1-score ElasticNN-InceptionV3 test on CIFAR 100"
        ax1.set_title(self.imageStr["ax1_title"])
        ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        fig_size = plt.rcParams["figure.figsize"]

        fig_size[0] = 6.4
        fig_size[1] = 6.4

        plt.rcParams["figure.figsize"] = fig_size

        #plt.tight_layout()

        # imagecaption = "accuracy_InceptionV3_CIFAR_100.pdf"
        # imagecaption2 = "accuracy_Elastic_InceptionV3_CIFAR_100.pdf"
        plt.savefig(self.path + os.sep + self.imageStr["save_fig"], bbox_inches="tight")
        plt.close("all")

    def on_epoch_end(self, epoch, logs={}):
        LOG ('\n -----------------------------------Epoch: %d------------------------------------------' % epoch, self.logFile)
        layers_error = [] # intermediate layers error
        layers_f1_score = [] 
        preds = self.model.predict(self.X)
        lr = float(K.get_value(self.model.optimizer.lr))
        LOG('\n Learning rate on epoch %d is : ' % epoch + str.format("{:.6f}", lr), self.logFile)

        if self.num_outputs == 1:
            error = log_error(self.y, preds)
            layers_error.append(error)
            score = get_f1_score(self.y, preds)
            layers_f1_score.append(score)

        elif self.num_outputs > 1:
            for k in range(self.num_outputs):
                error = log_error(self.y, preds[k])
                layers_error.append(error)
                
                score = get_f1_score(self.y, preds[k])
                layers_f1_score.append(score)
        
        # record accuracies 
        self.errors.append(layers_error)
        self.f_score.append(layers_f1_score)
        with open(self.path + os.sep + "accuracies.txt", "a") as fp:
            for e in layers_error:
                fp.write("%.4f " % e)
            fp.write("\n")

        with open(self.path + os.sep + "f1-score.txt", "a") as fp:
            for e in layers_f1_score:
                fp.write("%.4f " % e)
            fp.write("\n")

        self.plot()


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

    with open(path + os.sep + "test_intermediate_accuracies.txt", "a") as fp:
        for a in epochs_intermediate_acc_test:
            fp.write("%.4f " % a)
        fp.write("\n")    
    
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
        plt.savefig(self.path + os.sep + self.imageStr["save_fig"], bbox_inches="tight")
        plt.close("all")