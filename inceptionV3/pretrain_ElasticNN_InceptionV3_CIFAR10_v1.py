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
import math
from helper import load_data, multi_output_generator, LOG, log_summary, log_error, HistoryLogger


def pretrain():
    
    # ===============parameters================
    
    program_start_time = time.time()
    instanceName = "Accuracy"
    model_name = "Elastic-InceptionV3_CIFAR10"


    folder_path = "./Pretrained_ElasticNN" + "/CIFAR10"

    num_classes = 10
    batch_size = 32
    epoch_size = 1000
    dropout_rate = 0.2
    target_size = (224,224,3)
    imageStr = {
        "ax0_set_ylabel": "error rate on CIFAR-10",
        "ax0_title": "Elastic-InceptionV3 test on CIFAR 10",
        "ax1_set_ylabel": "f1 score on CIFAR-10",
        "ax1_title": "f1 score Elastic-InceptionV3 test on CIFAR 10",
        "save_fig" : "Elastic-InceptionV3_CIFAR_10.pdf"
    }

    # ==============Done parameters============

    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + model_name + "_" + ts_str
    os.makedirs(path)

    logFile = path + os.sep + "log.txt"

    X_train, y_train, X_val, y_val, x_test, y_test= load_data(datafile = "cifar10", target_size = (224,224,3), num_class = 10, test_percent=0.2)

    LOG("Pre-training CIFAR-10 on ElasticNN-InceptionV3...", logFile)

    # # 从keras中下载inceptionv3 model, 然后根据ElasticNN论文中的描述, 对inceptionV3模型修改
    # # "3" 是指 三通道, 即 RGB
    # # target_size = (139,139,3)
    
    # base_model = InceptionV3(include_top=False, input_shape=target_size)

    # # 冻结base_model原本模型的神经网络层，只有新添加的层才能训练
    # for layer in base_model.layers:
    #     layer.trainable = False
        
    # w = base_model.outputs[0]
    # w = Flatten()(w)

    # # add dropout before softmax to avoid overfit
    # w = Dropout(dropout_rate)(w)

    # final_output = Dense(num_classes, activation='softmax', name='final_output')(w)


    # # 把 mixed_* 中间层包含进去
    # add_layers = []
    # add_layers = [layer for layer in base_model.layers if "mixed" in layer.name and ('mixed9_0'!= layer.name and 'mixed9_1'!= layer.name )]

    # intermediate_outputs = []



    # for layer, no_layer in zip(add_layers, range(len(add_layers))):
    #     w = layer.output
    #     name = layer.name
    #     w = GlobalAveragePooling2D()(w) #是指 三通道, 即 RGB
        
    #     # add dropout to avoid overfit, since for the last layers, it has overfit problem
    #     if no_layer in [8, 9, 10, 11]:
    #         w = Dropout(dropout_rate)(w)

    #     w = Dense(num_classes, activation='softmax', name='intermediate_' + name)(w)
    #     # w = Dense(101, activation='sigmoid', name='intermediate_' + name)(w)
    #     intermediate_outputs.append(w)    
    

    # inputs = base_model.inputs

    # outputs = intermediate_outputs + [final_output]
    # # output = final_output

    # # 开始定义自己的神经网络模型
    # model = Model(inputs=inputs, outputs=outputs)

    # output_names = [output.name.split("/")[0] for output in outputs]
    # losses = {name: 'categorical_crossentropy' for name in output_names}

    # num_outputs = len(outputs)
    # print("there are %d outputs, (# of outputs) == (# of intermediate) + (# of final output.)\n" % num_outputs)

    # # 对前面几个中间层的输出的权重设置为1, default weight for each intermediate is 0, and for the final output is 1
    # loss_weights = [0] * num_outputs
    # loss_weights[num_outputs-1] = 1
    # loss_weights[:num_outputs] = [1]*len(loss_weights)
    # loss_weights = {name: w for name, w in zip(output_names, loss_weights)}

    # model.compile(loss=losses, optimizer=SGD(lr=1e-3), loss_weights=loss_weights, metrics=['accuracy'])

    # # Print the Model summary
    
    
    model = load_model("./Pretrained_ElasticNN/CIFAR10/Accuracy/Elastic-InceptionV3_CIFAR10_2018-05-16-09-01-10model.best.hdf5")


    outputs = model.outputs

    num_outputs = len(outputs)
    print("there are %d outputs.\n" % num_outputs)

    output_names = [output.name.split("/")[0] for output in outputs]
    losses = {name: 'categorical_crossentropy' for name in output_names}

    loss_weights = [0] * num_outputs
    loss_weights[num_outputs-1] = 1
    loss_weights[:num_outputs] = [1]*len(loss_weights)
    loss_weights = {name: w for name, w in zip(output_names, loss_weights)}
        
    for layer in model.layers:
            layer.trainable = True
        # 经过训练之后， 再次重新编译神经网络模型

    model.compile(loss=losses, optimizer=SGD(lr=1e-3), loss_weights=loss_weights)





    train_datagen = ImageDataGenerator(horizontal_flip=False,
                                       data_format=K.image_data_format())
    
    train_generator = train_datagen.flow(X_train, y_train,
                                        batch_size=batch_size)

    multi_generator = multi_output_generator(train_generator, num_outputs)

    history_logger = HistoryLogger(model.optimizer, path, batch_size, epoch_size, x_test, y_test, num_outputs, multi_generator, logFile, imageStr)

    checkpointer = ModelCheckpoint(filepath= path + 'model.best.hdf5', verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5, cooldown=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.003, patience=10, verbose=0, mode='auto')
    
    steps = math.ceil(len(X_train) / batch_size)

    LOG("===================Pretraining the new layer for %d epochs==================" % epoch_size, logFile)
    # 这里的意思是训练最后一层全连接层的权重，但是没有包括之前forze
    model.fit_generator(multi_generator,
                        epochs=epoch_size,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]*num_outputs),
                        callbacks=[checkpointer, lr_reducer, early_stop])





    
    # # load the weights that yielded the best validation accuracy
    # model.load_weights(path+'model.best.hdf5')

    # # # evaluate test accuracy
    # # score = model.evaluate(x_test, y_test, verbose=0)
    # # accuracy = 100*score[1]

    # # print test accuracy
    # # print('previous layers are not trainable, Test accuracy: %.4f%%' % accuracy)





    # train all model layers which means including all previsous-frozed layers before.
    for layer in model.layers:
        layer.trainable = True
    # checkpointer2 = ModelCheckpoint(filepath= path + 'model2.best.hdf5', verbose=1, save_best_only=True)
    # # 经过训练之后， 再次重新编译神经网络模型
    model.compile(loss=losses, optimizer=SGD(lr=1e-3), loss_weights=loss_weights, metrics=['accuracy'])
    
    log_summary(model, logFile)
    LOG("====================Pretraining all layers, with including all previours frozened layers====================", logFile)

    model.fit_generator(multi_generator,
                        epochs=epoch_size,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]*num_outputs),
                        callbacks=[history_logger, checkpointer,lr_reducer, early_stop])
                        

    # print("program elapse: ", time.time() - program_start_time)

    # # load the weights that yielded the best validation accuracy
    # model.load_weights(path+'model2.best.hdf5')

    # # # evaluate test accuracy
    # # score = model.evaluate(x_test, y_test, verbose=0)
    # # accuracy = 100*score[1]

    # # print test accuracy
    # print('all layers are trainable, Test accuracy: %.4f%%' % accuracy)




if __name__ == "__main__":
    pretrain()

    # model = load_model("./Pretrained_ElasticNN_CIFAR100/AGE/InceptionV3_CIFAR100_2018-05-08-08-53-56model.best.hdf5")
    # # x_train shape is (50000, 32,32,3); y_train shape is (50000, 1)
    # (x_data, y_data), (x_test, y_test) = cifar100.load_data()

    # # train_data[b'data'][0] 的长度是3072 也就是3072 = 3 × 32 × 32， 也就是说cifar100的照片尺寸是32 × 32；
    # num_classes = 100
    # X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    # y_train = np_utils.to_categorical(y_train, num_classes)
    # y_val = np_utils.to_categorical(y_val, num_classes)num_classes

    # big_x_train = np.array([scipy.misc.imresize(X_train[i], (139, 139, 3)) for i in range(0, len(X_train))]).astype('float32')
    # X_train = preprocess_input(big_x_train)

    # big_x_val = np.array([scipy.misc.imresize(X_val[i], (139, 139, 3)) for i in range(0, len(X_val))]).astype('float32')
    # X_val = preprocess_input(big_x_val)

    # big_x_test = np.array([scipy.misc.imresize(x_test[i], (139, 139, 3)) for i in range(0, len(x_test))]).astype('float32')
    # x_test = preprocess_input(big_x_test)
    
    # y_test = np_utils.to_categorical(y_test, num_classes)

    # score = model.predict(x_test)
    # on_epoch_end(score, y_test)
    # print(len(score))
    # print(len(score[0]))
    # print(len(score[0][0]))
    # print(score[0][0])
    # # accuracy = 100*score[1]
    # print("y true label : ", y_test[0])
    # print(list(y_test[0]).index(1))
    # print(list(score[0][0]).index(max(list(score[0][0]))))
    # print test accuracy
    # print('all layers are trainable, Test accuracy: %.4f%%' % accuracy)
    # 我需要去计算每一层的accuracy
