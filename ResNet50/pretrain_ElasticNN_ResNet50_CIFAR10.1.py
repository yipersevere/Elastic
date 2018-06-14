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
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
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
    model_name = "Elastic-ResNet50_CIFAR10_all_intermediate_resblock"
    folder_path = "./Train_ElasticNN-ResNet50" + "/CIFAR10"
    num_classes = 10

    batch_size = 32
    epoch_size = 1
    target_size = (224, 224, 3)
    dropout_rate = 0.2
    imageStr = {
        "ax0_set_ylabel": "error rate on CIFAR-10",
        "ax0_title": "ElasticNN-ResNet50 test on CIFAR 10",
        "ax1_set_ylabel": "f1 score on CIFAR-10",
        "ax1_title": "f1 score ElasticNN-ResNet50 test on CIFAR 10",
        "save_fig" : "ElasticNN-ResNet50_CIFAR_10.pdf"
    }
   # ==============Done parameters============
    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + model_name + "_" + ts_str
    os.makedirs(path)

    global logFile
    logFile = path + os.sep + "log.txt"

    # x_train shape is (50000, 32,32,3); y_train shape is (50000, 1)
    X_train, y_train, X_val, y_val, x_test, y_test= load_data(datafile = "cifar10", target_size = target_size, num_class = num_classes, test_percent=0.2)

    LOG("Pre-training CIFAR-10 on ElasticNN-ResNet50...", logFile)

    base_model = ResNet50(include_top=False, input_shape=target_size)

    for layer in base_model.layers:
        layer.trainable = False

    w = base_model.outputs[0]
    w = Flatten()(w)
    
    # add dropout before softmax to avoid overfit
    w = Dropout(dropout_rate)(w)

    final_output = Dense(num_classes, activation='softmax', name='final_output')(w)

    add_layers = [layer for layer in base_model.layers if "add_" in layer.name]

    intermediate_outputs = []

    for layer in add_layers:
        w = layer.output
        name = layer.name
        w = GlobalAveragePooling2D()(w)

        # add dropout before softmax to avoid overfit
        w = Dropout(dropout_rate)(w)
        
        w = Dense(num_classes, activation='softmax', name='intermediate_' + name)(w)
        intermediate_outputs.append(w)

    inputs = base_model.inputs
    outputs = intermediate_outputs + [final_output]

    model = Model(inputs=inputs, outputs=outputs)

    output_names = [output.name.split("/")[0] for output in outputs]
    losses = {name: 'categorical_crossentropy' for name in output_names}

    num_outputs = len(outputs)
    print("there are %d outputs.\n" % num_outputs)

    loss_weights = [0] * num_outputs


    # loss_weights[8:num_outputs] = [1] * len(loss_weights[8:num_outputs])
    """
    here we get all 16 resblock and we add them all to the network, which means the weight for resblock is 1; before we only use layers
    from 8th to 16th.
    """
    loss_weights[0:num_outputs] = [1] * len(loss_weights[0:num_outputs])





    loss_weights = {name: w for name, w in zip(output_names, loss_weights)}

    model.compile(loss=losses, optimizer=SGD(lr=1e-3), loss_weights=loss_weights)

    # Print the Model summary
    log_summary(model, logFile)
    
    train_datagen = ImageDataGenerator(horizontal_flip=False,
                                       data_format=K.image_data_format())
    
    train_generator = train_datagen.flow(X_train, y_train,
                                        batch_size=batch_size)

    multi_generator = multi_output_generator(train_generator, num_outputs)
    
    history_logger = HistoryLogger(model.optimizer, path, batch_size, epoch_size, x_test, y_test, num_outputs, multi_generator, logFile, imageStr)

    checkpointer = ModelCheckpoint(filepath= path + 'model.best.hdf5', verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5, cooldown=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    steps = math.ceil(len(X_train) / batch_size)

    LOG("===================Pretraining the new layer for %d epochs==================" % epoch_size, logFile)
    # 这里的意思是训练最后一层全连接层的权重，但是没有包括之前forze
    model.fit_generator(multi_generator,
                        epochs=epoch_size,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]*num_outputs),
                        callbacks=[checkpointer,lr_reducer, early_stop])

    # load the weights that yielded the best validation accuracy
    model.load_weights(path+'model.best.hdf5')

    # train all model layers which means including all previsous-frozed layers before.
    for layer in model.layers:
        layer.trainable = True
    # checkpointer2 = ModelCheckpoint(filepath= path + 'model2.best.hdf5', verbose=1, save_best_only=True)
    # 经过训练之后， 再次重新编译神经网络模型
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-2), metrics=['accuracy'])

    LOG("====================Pretraining all layers, with including all previours frozened layers====================", logFile)

        # Print the Model summary
    log_summary(model, logFile)
    model.fit_generator(multi_generator,
                        epochs=epoch_size,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]*num_outputs),
                        callbacks=[history_logger, checkpointer, lr_reducer, early_stop])
                        
    program_elapse = time.time() - program_start_time
    print("program elapse: ", program_elapse)
    # log_summary("program elapse: "+str(program_elapse), logFile)
if __name__ == "__main__":

    pretrain()
