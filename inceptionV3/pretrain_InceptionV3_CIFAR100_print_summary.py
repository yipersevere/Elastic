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
from keras.layers import Dense, Activation, Flatten
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
# from keras.applications.xception import Xception
# from keras.applications.mobilenet import MobileNet
# from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import np_utils

from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, ModelCheckpoint, \
    LearningRateScheduler
from keras.utils import plot_model

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_absolute_error
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
import datetime
from io import StringIO
import shutil
import random
import sys
import traceback
import time
from keras.applications.inception_v3 import preprocess_input
import scipy
from keras.datasets import cifar100
# from keras.applications.inception_v3_matt import InceptionV3, preprocess_input
from helper import load_data, multi_output_generator, LOG, log_summary, log_error, HistoryLogger
import math

def pretrain():

    # ===============parameters================

    program_start_time = time.time()
    instanceName = "Accuracy"
    model_name = "InceptionV3_CIFAR100"
    folder_path = "./Train_InceptionV3" + "/CIFAR100"
    batch_size = 32
    epoch_size = 1
    num_outputs = 1
    num_classes = 100
    target_size = (224,224,3)
    imageStr = {
        "ax0_set_ylabel": "error rate on CIFAR-100",
        "ax0_title": "InceptionV3 test on CIFAR 100",
        "ax1_set_ylabel": "f1 score on CIFAR-100",
        "ax1_title": "f1 score InceptionV3 test on CIFAR 100",
        "save_fig" : "InceptionV3_CIFAR_100.pdf"
    }
    # ==============Done parameters============
    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + model_name + "_" + ts_str
    os.makedirs(path)

    logFile = path + os.sep + "log.txt"
    
    # train_data[b'data'][0] 的长度是3072 也就是3072 = 3 × 32 × 32， 也就是说cifar100的照片尺寸是32 × 32；



    X_train, y_train, X_val, y_val, x_test, y_test= load_data(datafile = "cifar100", target_size = target_size, num_class = num_classes, test_percent=0.2)
    

    LOG("Pre-training CIFAR-100 on InceptionV3...", logFile)

    # 从keras中下载inceptionv3 model, 然后根据ElasticNN论文中的描述, 对inceptionV3模型修改
    # "3" 是指 三通道, 即 RGB
    
    
    base_model = InceptionV3(include_top=False, input_shape=target_size)

    # 冻结base_model原本模型的神经网络层，只有新添加的层才能训练
    for layer in base_model.layers:
        layer.trainable = False
        
    w = base_model.outputs[0]
    w = Flatten()(w)

    final_output = Dense(num_classes, activation='softmax', name='final_output')(w)

    inputs = base_model.inputs

    output = final_output

    # 开始定义自己的神经网络模型
    model = Model(inputs=inputs, outputs=output)

    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-3), metrics=['accuracy'])
    log_summary(model, logFile)
    
    # train_datagen = ImageDataGenerator(horizontal_flip=False,
    #                                    data_format=K.image_data_format())
    
    # train_generator = train_datagen.flow(X_train, y_train,
    #                                     batch_size=batch_size)

    

    # checkpointer = ModelCheckpoint(filepath= path + 'model.best.hdf5', verbose=1, save_best_only=True)
    # lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5, cooldown=1)
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=0, mode='auto')
    # history_logger = HistoryLogger(model.optimizer, path, batch_size, epoch_size, x_test, y_test, num_outputs, train_generator, logFile, imageStr)

    # steps = math.ceil(len(X_train) / batch_size)
    
    
    # LOG("===================Pretraining only the new layer for %d epochs==================" % epoch_size, logFile)
    # # 这里的意思是训练最后一层全连接层的权重，但是没有包括之前forze
    # model.fit_generator(train_generator,
    #                     epochs=epoch_size,
    #                     steps_per_epoch=steps,
    #                     verbose=0,
    #                     validation_data=(X_val, [y_val]),
    #                     callbacks=[checkpointer,lr_reducer, early_stop])

    # # train all model layers which means including all previsous-frozed layers before.
    # for layer in model.layers:
    #     layer.trainable = True
    
    # # 经过训练之后， 再次重新编译神经网络模型
    # model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-3), metrics=['accuracy'])
    
    # log_summary(model, logFile)

    # LOG("====================Pretraining all layers, with including all previours frozened layers====================", logFile)
    # # model.fit_generator(train_generator,
    # #                     epochs=epoch_size,
    # #                     steps_per_epoch=steps,
    # #                     verbose=0,
    # #                     validation_data=(X_val, [y_val]),
    # #                     callbacks=[history_logger, checkpointer, lr_reducer, early_stop])
                        

    # print("program elapse: ", time.time() - program_start_time)

if __name__ == "__main__":
    pretrain()
