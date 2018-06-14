#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:53:25 2018

@author: yue
"""

import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2

import keras
from keras import optimizers
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, ModelCheckpoint, \
    LearningRateScheduler
from keras.utils import plot_model

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_absolute_error
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


import datetime
from io import StringIO
import shutil
import random
import sys
import traceback
import time

np.seterr(divide='ignore', invalid='ignore')
def LOG(message):
    global logFile
    ts = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    msg = "[%s] %s" % (ts, message)

    with open(logFile, "a") as fp:
        fp.write(msg + "\n")

    print(msg)


def step_decay(epoch):
    if epoch <= 150:
        lrate=1e-3
    if 150 < epoch <= 250:
        lrate=1e-4
    if epoch > 250:
        lrate=1e-5
    return lrate


def log_summary(model):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model.summary()

    sys.stdout = old_stdout
    summary = mystdout.getvalue()

    LOG("Model summary:")

    for line in summary.split("\n"):
        LOG(line)


def preprocess(img):
    # RGB -> BGR

    x = img[..., ::-1].astype(np.float32)

    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68

    return x


def load_images(path, target_size, std):
    X = []
    y = []
    std_output = []

    for root, dirs, files in os.walk(path):
        for name in files:
            fullname = root + os.sep + name

            x = cv2.imread(fullname)
            x = cv2.resize(x, target_size)
            x = x[..., ::-1]  # BGR -> RGB (which "preprocess" will invert back)

            x = preprocess(x)
            X.append(x)

            c = int(fullname.split(os.sep)[-2])
            ind = np.zeros(101)
            ind[c] = 1
            y.append(ind)
            std_output.append(std[name])

    y = np.array(y)
    X = np.array(X)
    std_output = np.array(std_output)

    return X, y, std_output


def get_class_weights(path, classes):
    class_sizes = []
    for folder in classes:
        try:
            class_sizes.append(len(os.listdir(path + os.sep + str(folder))))
        except:
            class_sizes.append(0)

    class_sizes = np.array(class_sizes)
    class_weights = np.max(class_sizes) / (class_sizes + 100)
    class_weights = class_weights / np.mean(class_weights)

    class_weights = dict(zip(classes, class_weights))

    return class_weights


def log_error(labels, predictions, std):
    classes = np.arange(0, 101)
    # normalization
    predictions = np.transpose(np.transpose(predictions) / np.sum(predictions, axis=1))
    label = np.matmul(labels, classes)
    predict = np.matmul(predictions, classes)
    err = mean_absolute_error(label, predict)
    epsilon = 1 - np.exp (np.square(predict-label)/(-2*np.square(std)))
    epsilon = np.sum(epsilon)/epsilon.shape[0]
    # print (np.sum(epsilon))
    # print (epsilon.shape[0])
    return err,epsilon

def adapt_model(trained_model, total_outputs, current_layer):
    new_output_layer = []
    if total_outputs == current_layer + 1 :
        new_output_layer = [layer for layer in trained_model.layers if "final_output" in layer.name] 
    else:  
        for layer in trained_model.layers:
            if 'intermediate_' in layer.name:
                num_layer_name = '_' + str(current_layer+1)
                if  num_layer_name in layer.name:
                    print ('\n The current output layer is: ' + layer.name)
                    new_output_layer.append(layer)
                    break
    new_output = new_output_layer[0].output
    inputs = trained_model.inputs
    adapted_model = Model(inputs=inputs, outputs=new_output)
    return adapted_model
    
class HistoryLogger(Callback):
    def __init__(self, optimizer, path, batch_size, epoch_size, X_test, y_test, X_val, y_val,
                 train_generator, num_outputs, std_test, std_val):

        self.path = path
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.train_generator = train_generator
        self.num_outputs = num_outputs
        self.std_test = std_test
        self.std_val = std_val

        self.train_losses = []
        self.train_errors = []
        self.val_losses = []
        self.val_errors = []
        self.val_epsilons = []
        self.test_losses = []
        self.test_errors = []
        self.test_epsilons = []

        self.total_shown = 0


        super().__init__()

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        #        LOG ('\n -----------------------------------Epoch: %d------------------------------------------' % epoch)   
#        test_losses = []
#        test_errors = []
#        test_epsilons = []
#        val_losses = []
#        val_errors = []
#        val_epsilons = []
#        test_predictions = []
#        val_predictions = []
#        
#        for k in range(self.num_outputs):
#            model_adapted = adapt_model(self.model, self.num_outputs, k)
#            START_TIME = time.time()
#            test_prediction = model_adapted.predict(self.X_test)
#            test_cpt_time = time.time()-START_TIME
#            test_speed = test_cpt_time / self.y_test.shape[0]  # time / processed picture
#            LOG ('\n The current speed of output %d on test set is  %f : ' % (k+1, test_speed) + '(seconds / picture)')
#            test_predictions.append(test_prediction)
#        
        LOG ('\n -----------------------------------Epoch: %d------------------------------------------' % epoch)
        test_losses = []
        test_errors = []
        test_epsilons = []
        val_losses = []
        val_errors = []
        val_epsilons = []
        test_predictions = []
        val_predictions = []
        START_TIME = time.time()
        test_predictions = self.model.predict(self.X_test)
        test_cpt_time = time.time()-START_TIME
        test_speed = test_cpt_time / self.y_test.shape[0]  # time / processed picture
        LOG ('\n The current speed on test set is  %f : ' % test_speed + '(seconds / picture)')
        val_predictions = self.model.predict(self.X_val)
        lr = float(K.get_value(self.model.optimizer.lr))
        LOG('\n Learning rate on epoch %d is : ' % epoch + str.format("{:.6f}", lr))
        for k in range(self.num_outputs):
            # test log
            test_loss = log_loss(self.y_test, test_predictions[k])
            test_error, test_epsilon = log_error(self.y_test, test_predictions[k], self.std_test)
            test_losses.append(test_loss)
            test_errors.append(test_error)
            test_epsilons.append(test_epsilon)

            # validate log
            val_loss = log_loss(self.y_val, val_predictions[k])
            val_error, val_epsilon = log_error(self.y_val, val_predictions[k], self.std_val)
            val_losses.append(val_loss)
            val_errors.append(val_error)
            val_epsilons.append(val_epsilon)
        
        # record test 
        self.test_errors.append(test_errors)
        self.test_losses.append(test_losses)
        self.test_epsilons.append(test_epsilons)

        with open(self.path + os.sep + "test_losses.txt", "a") as fp:
            for loss in test_losses:
                fp.write("%.4f " % loss)
            fp.write("\n")

        with open(self.path + os.sep + "test_errors.txt", "a") as fp:
            for error in test_errors:
                fp.write("%.4f " % error)
            fp.write("\n")
        with open(self.path + os.sep + "test_epsilons.txt", "a") as fp:
            for epsilon in test_epsilons:
                fp.write("%.4f " % epsilon)
            fp.write("\n")
        # record val 
        self.val_errors.append(val_errors)
        self.val_losses.append(val_losses)
        self.val_epsilons.append(val_epsilons)

        with open(self.path + os.sep + "val_losses.txt", "a") as fp:
            for loss in val_losses:
                fp.write("%.4f " % loss)
            fp.write("\n")

        with open(self.path + os.sep + "val_errors.txt", "a") as fp:
            for error in val_errors:
                fp.write("%.4f " % error)
            fp.write("\n")
        with open(self.path + os.sep + "val_epsilons.txt", "a") as fp:
            for epsilon in val_epsilons:
                fp.write("%.4f " % epsilon)
            fp.write("\n")
        with open(self.path + os.sep + "timer.txt", "a") as fp:
            fp.write('The average speedn on test set is  %f : ' % test_speed + '(seconds / picture)')
            fp.write("\n")
        
        # plot test
        fig, ax = plt.subplots(3, sharex=True)
        colormap = plt.cm.tab20
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        for k in range(len(self.test_losses[0])):
            # Plots
            x = np.arange(len(self.test_losses)) + 1
            y = np.array(self.test_losses)[:, k]
            c_label = 'Layer ' + str(k)
            ax[0].plot(x, y, label=c_label)

            # Legends
            y = self.test_losses[-1][k]
            x = len(self.test_losses)
            ax[0].text(x, y, "%d" % k)

        # (5-2)*5+1=16
        loss_ticks = np.linspace(2, 5, 16)

        ax[0].set_yticks(loss_ticks)
        ax[0].set_ylabel('Prediction loss')
        ax[0].set_xlabel('Training time')
        ax[0].set_title('Loss evaluation of test set')
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        for k in range(len(self.test_errors[0])):
            # Plots
            x = np.arange(len(self.test_errors)) + 1
            y = np.array(self.test_errors)[:, k]
            c_label = 'Layer ' + str(k)
            ax[1].plot(x, y, label=c_label)

            # Legends

            y = self.test_errors[-1][k]
            x = len(self.test_errors)
            ax[1].text(x, y, "%d" % k)

        # (12-2)*2+1=21
        error_ticks = np.linspace(2, 12, 21)

        ax[1].set_yticks(error_ticks)
        ax[1].set_ylabel('Prediction error (years)')
        ax[1].set_xlabel('Training time')
        ax[1].set_title('Error evaluation of test set')
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        for k in range(len(self.test_epsilons[0])):
            # Plots
            x = np.arange(len(self.test_epsilons)) + 1
            y = np.array(self.test_epsilons)[:, k]
            c_label = 'Layer ' + str(k)
            ax[2].plot(x, y, label=c_label)

            # Legends

            y = self.test_epsilons[-1][k]
            x = len(self.test_epsilons)
            ax[2].text(x, y, "%d" % k)

        # epsilons_sticks = np.linspace(5, 20, 31)

        # ax[2].set_yticks(epsilons_sticks)
        ax[2].set_ylabel('Prediction epsilon (0.30)')
        ax[2].set_xlabel('Training time')
        ax[2].set_title('Epsilon evaluation of test set')
        ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig_size = plt.rcParams["figure.figsize"]
        # print ("Current size:", fig_size)  # Prints: [6.4, 4.8]
        fig_size[0] = 6.4
        fig_size[1] = 14.4
        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, self.num_outputs)])

        plt.rcParams["figure.figsize"] = fig_size

        plt.tight_layout()

        plt.savefig(self.path + os.sep + "loss_MAE_e_test.pdf", bbox_inches="tight")
        plt.close("all")


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


def getstd(csv_path):
    std_output={}
    for ann_file in [csv_path]:
        with open(ann_file, "r") as fp:
            for k, line in enumerate(fp):
                if k == 0:  # skip header
                    continue
                name, age, std = line.split(",")
                std_output[name] = float(std)
    return std_output

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# # def load_cifar(path):
# def get_valid_data(trainData):
    
#     percentage = 0.2
#     return 0
    


def train():
    instanceName = "AGE"
    model_name = "InceptionV3_noshearing"

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    # 这个路径是指生成新的inception v3 的模型的路径
    path = "./Train" + "/CVPR_MODEL" + os.sep + instanceName + os.sep + model_name + "_" + ts_str
    os.makedirs(path)

    global logFile
    logFile = path + os.sep + "log.txt"

    # pretrain_path = "data/noshearing_IMDB_WIKI/CROPS_AGE"
    valid_path = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/data/noshearing_CVPR/VALID"
    train_path = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/data/noshearing_CVPR/TRAIN"
    test_path = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/data/noshearing_CVPR/TEST"
    csv_valid_path = '/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/csv_file/valid_gt.csv'
    csv_test_path = '/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/csv_file/test_gt.csv'


    cifar_train_path = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/data/CIFAR/cifar-100-python/train"
    cifar_valid_path = "" #should split from train set
    cifar_test_path = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/data/CIFAR/cifar-100-python/test"
    # train_data[b'data'][0] 的长度是3072 也就是3072 = 3 × 32 × 32， 也就是说cifar100的照片尺寸是32 × 32；
    # 而CVPR的数据集中的图片大小是： 224 × 224
    train_valid_data = unpickle(cifar_train_path)
    test_data = unpickle(cifar_test_path)
    # valid_data = get_valid_data(train_data)
    # 注意加载的时候我们已经把 32×32*3 的图片变成了 1 × 3072
    X_train, X_valid, y_train, y_valid = train_test_split(train_valid_data[b'data'], train_valid_data[b'fine_labels'], test_size = 0.2)
    X_train = X_train.reshape((len(X_train), 32, 32, 3))
    X_valid = X_valid.reshape((len(X_valid), 32, 32, 3))


    # 生成批处理操作



    plt.imshow(list(X_train[0]))
    plt.show()

    batch_size = 32
    epoch_size = 1024

    LOG("Training starts...")

    target_size = (224, 224)
    # target_size = (32, 32)

    
    # model = load_model('CVPR_MODEL'  + os.sep + 'InceptionV3' + os.sep + 'InceptionV3_noshearing.h5')
    model_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/Pretrained/AGE/InceptionV3_noshearing_2018-04-29-17-47-45/model.h5"
    model = load_model(model_file)


    outputs = model.outputs
    num_outputs = len(outputs)
    print("there are %d outputs.\n" % num_outputs)
    # 这里的output是指模型中的中间层输出， 在inception v3中我们得到了总共12层的输入， 包括final_output
    output_names = [output.name.split("/")[0] for output in outputs]
    losses = {name: 'categorical_crossentropy' for name in output_names}
    loss_weights = [0] * num_outputs
    loss_weights[num_outputs-1] = 1
    loss_weights = {name: w for name, w in zip(output_names, loss_weights)}

    # 这里将inception v3 的模型重新编译一遍， 其中最重要的就是更换了loss_weights, 和 losses 
    model.compile(loss=losses, optimizer=SGD(lr=1e-3), loss_weights=loss_weights)

    # Print the Model summary for the user
    log_summary(model)
    STD_VALID = {}
    STD_TEST = {}
    STD_TEST = getstd(csv_test_path)
    STD_VALID = getstd(csv_valid_path)

    ###### CVPR TRAINING
    # Training data preparation
    train_datagen = ImageDataGenerator(horizontal_flip=False,
                                       data_format=K.image_data_format(),
                                       preprocessing_function=preprocess)

    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        color_mode='rgb',
                                                        classes=[str(c) for c in range(101)],
                                                        class_mode='categorical')
    # cifar_train_generator = “ ”


    multi_generator = multi_output_generator(train_generator, num_outputs)

# 这里的class_weights是一个dict， key是0。。100
    class_weights = get_class_weights(train_path, train_generator.class_indices.values())
    
    # Validating data preparation
    # 这里的X_val的shape是： （1500， 32， 32， 3）， y_val 的shape是(1500, 101), std_val 的shape是(1500,1)
    X_val, y_val, std_val = load_images(valid_path, target_size, STD_VALID)
    
    # Testing data preparation
    # 这里的X_test, y_test 的shape是：（1978， 32， 32， 3）， y_test 的shape是（1978， 101）
    X_test, y_test, std_test = load_images(test_path, target_size, STD_TEST)

    # X_train = Reshape((-1, 32, 32, 3))
    # X_train = 

    # Call back
    history_logger = HistoryLogger(model.optimizer, path, batch_size, epoch_size, X_test, y_test, 
                                   X_val, y_val, multi_generator, num_outputs,  std_test, std_val)
    checkpointer = ModelCheckpoint(filepath=path + os.sep + "model.h5", verbose=0, save_best_only=False)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=5, min_lr=1e-5, cooldown=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=15, verbose=0, mode='auto')

    LOG("Training CVPR...")

    # change_lr = LearningRateScheduler(step_decay)
    # 为什么这里又重新编译了模型一遍
    model.compile(loss=losses, optimizer=SGD(lr=1e-3), loss_weights=loss_weights)
    model.fit_generator(multi_generator,
                        steps_per_epoch=epoch_size,
                        epochs=150,
                        verbose=0,
                        validation_data=(X_val, [y_val] * num_outputs),
                        class_weight=class_weights,
                        callbacks=[history_logger, checkpointer, lr_reducer,early_stop])

#   y_val 是 1500 × 101 维度的数组， X_val 是 1500 * 224 * 224 * 3
# class_weights 是 1 * 101 维度的数组
# 但是这里对valid dataset并没有做批处理的操作




if __name__ == "__main__":
    train()