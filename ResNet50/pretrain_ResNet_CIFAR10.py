# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Mon 2017-11-13 19:09:54

@author: yue
"""

import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2

from keras.layers import Dense, Activation, Flatten
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
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, ModelCheckpoint
from keras.utils import plot_model

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_absolute_error
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

import datetime
from io import StringIO
import shutil
import random
import sys
import traceback
import time
from helper import load_data, multi_output_generator, LOG, log_summary, log_error, HistoryLogger
np.seterr(divide='ignore', invalid='ignore')



def pretrain():
    

    # ===============parameters================
    program_start_time = time.time()
    instanceName = "Accuracy"
    model_name = "ResNet_CIFAR10"
    folder_path = "./Train_ResNet50" + "/CIFAR10"
    num_classes = 10

    batch_size = 32
    epoch_size = 100
    num_outputs = 1
    target_size = (224, 224, 3)
    imageStr = {
        "ax0_set_ylabel": "error rate on CIFAR-10",
        "ax0_title": "ResNet-50 test on CIFAR 10",
        "ax1_set_ylabel": "f1 score on CIFAR-10",
        "ax1_title": "f1 score ResNet-50 test on CIFAR 10",
        "save_fig" : "ResNet-50_CIFAR_10.pdf"
    }
   # ==============Done parameters============



    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = folder_path + os.sep + instanceName + os.sep + model_name + "_" + ts_str
    os.makedirs(path)

    global logFile
    logFile = path + os.sep + "log.txt"

    
    X_train, y_train, X_val, y_val, x_test, y_test= load_data(datafile = "cifar10", target_size = (224,224,3), num_class = 10, test_percent=0.2)

    LOG("Pre-training on noshearing data (ResNet50)...", logFile)
    
    base_model = ResNet50(include_top=False, input_shape=target_size)

    for layer in base_model.layers:
        layer.trainable = False

    w = base_model.outputs[0]
    w = Flatten()(w)
    final_output = Dense(num_classes, activation='softmax', name='final_output')(w)

    inputs = base_model.inputs
    outputs = final_output

    model = Model(inputs=inputs, outputs=outputs)

    
    print("there are %d outputs.\n" % num_outputs)
    
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-3), metrics=['accuracy'])

    # Print the Model summary for the user

    log_summary(model, logFile)
  

    train_datagen = ImageDataGenerator(horizontal_flip=False,
                                       data_format=K.image_data_format())
    
    train_generator = train_datagen.flow(X_train, y_train,
                                        batch_size=batch_size)

    checkpointer = ModelCheckpoint(filepath= path + 'model.best.hdf5', verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5, cooldown=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.003, patience=10, verbose=0, mode='auto')
    history_logger = HistoryLogger(model.optimizer, path, batch_size, epoch_size, x_test, y_test, num_outputs, train_generator, logFile, imageStr)


    
    ###### IMDB_WIKI NEW LAYER TRAINING

   
    LOG("Pretraining the new layer for %d epochs..." % epoch_size, logFile)

    model.fit_generator(train_generator,
                        steps_per_epoch=1024,
                        epochs=epoch_size,
                        verbose=0,
                        validation_data=(X_val, [y_val]*num_outputs),
                        callbacks=[checkpointer, lr_reducer, early_stop])

    ###### IMDB_WIKI FULL TRAINING

    for layer in model.layers:
        layer.trainable = True

    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-3), metrics=['accuracy'])

    LOG("Pretraining all layers...", logFile)

    model.fit_generator(train_generator,
                        steps_per_epoch=1024,
                        epochs=epoch_size,
                        verbose=0,
                        validation_data=(X_val, [y_val] * num_outputs),
                        callbacks=[history_logger, checkpointer, lr_reducer, early_stop])


if __name__ == "__main__":
    pretrain()
