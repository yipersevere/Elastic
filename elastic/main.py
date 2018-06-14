
# Example run:
# ./main.py --model ElasticNN_ResNet50 --data cifar10 --num_classes 10 --target_size (224,224,3) --epoch 1 \
#  --add_intermediate_layers_number 2 --model_name Elastic-ResNet50_CIFAR10_all_intermediate_resblock --dropout_rate 0.2 \
# --batch_size 32 --learning_rate 1e-3

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import matplotlib

matplotlib.use("PDF")
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import np_utils

from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar100, cifar10

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, mean_absolute_error

import datetime
from io import StringIO
import shutil
import random
import sys
import traceback
import time
import math


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.allow_growth = True
#set_session(tf.Session(config=config))


from helper import load_data, multi_output_generator, LOG, log_summary, log_error, HistoryLogger
from opts import args
from ElasticNN import ElasicNN_ResNet50, ElasicNN_Inception, ElasticNN_MobileNets_alpha_0_75
from ElasticNN_Others_ResNet import ElasicNN_ResNet152

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

def main(**kwargs):
    global args
    # Override if needed
    for arg, v in kwargs.items():
        args.__setattr__(arg, v)
    print(args)
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

    global logFile
    logFile = path + os.sep + "log.txt"

    LOG("Pre-training " + args.data + " on " + args.model_name+ "...", logFile)

    if args.layers_weight_change == 1:
        LOG("weights for intermediate layers: 1/(152-Depth), giving different weights for different intermediate layers output, using the formula weigh = 1/(152-Depth)", logFile)
    elif args.layers_weight_change == 0:
        LOG("weights for intermediate layers: 1, giving same weights for different intermediate layers output as  1", logFile)
    else:
        print("Parameter --layers_weight_change, Error")
        sys.exit()

    if args.model == "Elastic_InceptionV3":
        elasicNN_inceptionV3 = ElasicNN_Inception(args)
        model = elasicNN_inceptionV3.model
        num_outputs = elasicNN_inceptionV3.num_outputs
        print("using inceptionv3 class")

    elif args.model == "Elastic_ResNet":
        elasicNN_resnet50 = ElasicNN_ResNet50(args)
        model = elasicNN_resnet50.model
        num_outputs = elasicNN_resnet50.num_outputs
        print("using resnet class")

    elif args.model == "Elastic_MobileNets_alpha_0_75":
        elasicNN_mobileNets_alpha_0_75 = ElasticNN_MobileNets_alpha_0_75(args)
        model = elasicNN_mobileNets_alpha_0_75.model
        num_outputs = elasicNN_mobileNets_alpha_0_75.num_outputs
        print("using mobilenets class")
    
    elif args.model == "Elastic_ResNet152":
        elasicNN_resnet152 = ElasicNN_ResNet152(args)
        model = elasicNN_resnet152.model
        num_outputs = elasicNN_resnet152.num_outputs
        print("using resnet 152 class")

    else:
        print("--model parameter should be in [inception, resnet, mobilenet]")
        exit()

    # Print the Model summary
    log_summary(model, logFile)

    datagen = ImageDataGenerator(horizontal_flip=False, data_format=K.image_data_format())
    
    global checkpointer
    checkpointer = ModelCheckpoint(filepath= path + 'model.best.hdf5', verbose=1, save_best_only=True)
    global lr_reducer
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5, cooldown=1)
    global early_stop
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=0, mode='auto')
    
    global tbCallBack
    tbCallBack = keras.callbacks.TensorBoard(log_dir=tensorboard_folder, histogram_freq=0, write_graph=True, write_images=True)

    global csv_logger
    csv_logger = CSVLogger(path + os.sep + 'log.csv', append=True, separator=';')

    X_train, y_train, X_val, y_val, x_test, y_test= load_data(data = args.data, target_size = args.target_size, num_class = args.num_classes, test_percent=0.2)
    
    train_generator = datagen.flow(X_train, y_train, batch_size=args.batch_size)
    multi_generator = multi_output_generator(train_generator, num_outputs)
    
    steps = math.ceil(len(X_train) / args.batch_size)
    # train ElasticNN-ResNet50
    model = train(model, X_val, y_val, multi_generator, args.batch_size, num_outputs, args.epoch, steps)

    model = eval(model, X_val, y_val, x_test, y_test, multi_generator, args.batch_size, num_outputs, args.epoch, path, imageStr, steps)
    # save model picture, this can't be run since gpu memory is currently used by above model
    # keras.utils.plot_model(model, to_file=path + os.sep + args.model_name +".png", show_shapes=True)
    return

def train(model, X_val, y_val, multi_generator, batch_size, num_outputs, epoch, steps):


    LOG("=====================Pretraining model -- %s ==============================" % args.model_name, logFile)
    LOG("===================training the intermediate layers, not including frozen layers for %d epochs==================" % epoch, logFile)
    # 这里的意思是训练最后一层全连接层的权重，但是没有包括之前forze
    model.fit_generator(multi_generator,
                        epochs=epoch,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]*num_outputs),
                        callbacks=[checkpointer,lr_reducer, early_stop, csv_logger])
    return model

def eval(model, X_val, y_val, x_test, y_test, multi_generator, batch_size, num_outputs, epoch, path, imageStr, steps):
    
    history_logger = HistoryLogger(model.optimizer, path, batch_size, epoch, x_test, y_test, num_outputs, multi_generator, logFile, imageStr)
   # load the weights that yielded the best validation accuracy
    model.load_weights(path+'model.best.hdf5')

    # train all model layers which means including all previsous-frozed layers before.
    for layer in model.layers:
        layer.trainable = True
    # checkpointer2 = ModelCheckpoint(filepath= path + 'model2.best.hdf5', verbose=1, save_best_only=True)
    # 经过训练之后， 再次重新编译神经网络模型
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=args.learning_rate), metrics=['accuracy'])

    LOG("==================== trainingg previous-frozen layers and the intermedate layers ====================", logFile)

        # Print the Model summary
    log_summary(model, logFile)
    model.fit_generator(multi_generator,
                        epochs=epoch,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]*num_outputs),
                        callbacks=[history_logger, checkpointer, lr_reducer, early_stop, tbCallBack, csv_logger])

    return model



if __name__ == '__main__':
    main()
