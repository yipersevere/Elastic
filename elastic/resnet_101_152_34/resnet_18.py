from __future__ import division
from __future__ import print_function
import six
import sys
sys.path.append("../")
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from helper import log_summary
import numpy as np


import sys
sys.path.append("../")
import cv2
import numpy as np
import copy

import keras
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras import backend as K
from helper import log_summary

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])


if __name__ == "__main__":
    """
    Adapted from keras example cifar10_cnn.py
    Train ResNet-18 on the CIFAR10 small images dataset.
    GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
    """
    log_file = "resnet_18_summary.txt"

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger('resnet18_cifar10.csv')

    batch_size = 32
    nb_classes = 10
    nb_epoch = 1
    data_augmentation = True

    # input image dimensions
    img_rows, img_cols = 32, 32
    # The CIFAR10 images are RGB.
    img_channels = 3

    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train /= 128.
    X_test /= 128.

    model = ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    log_summary(model, log_file)

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                validation_data=(X_test, Y_test),
                shuffle=True,
                callbacks=[lr_reducer, early_stopper, csv_logger])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_data=(X_test, Y_test),
                            epochs=nb_epoch, verbose=1, max_q_size=100,
                            callbacks=[lr_reducer, early_stopper, csv_logger])


class ElasicNN_ResNet18():
    '''
    modify ResNet152 model with elastic idea, adding intermediate layers into end of this model
    '''
    def __init__(self, args):
        self.data = args.data
        self.num_classes = args.num_classes
        self.target_size = args.target_size
        self.epoch = args.epoch
        self.model_name = args.model_name
        self.dropout_rate = args.dropout_rate
        self.batch_size = args.batch_size
        self.add_intermediate_layers_number = args.add_intermediate_layers_number
        self.learning_rate = args.learning_rate
        # # build neural network
        self.model = self.build_model()

    def add_intermediate_layers(self, flag_num_intermediate_layers, base_model):

        intermediate_layers_name = [
                                    'res2a', 'res2b', 'res2c', 'res3a', 'res3b1', 'res3b2',
                                    'res3b3','res3b4', 'res3b5', 'res3b6', 'res3b7', 'res4a',
                                    'res4b1','res4b2', 'res4b3', 'res4b4', 'res4b5','res4b6',
                                    'res4b7','res4b8', 'res4b9', 'res4b10', 'res4b11', 'res4b12',
                                    'res4b13','res4b14', 'res4b15', 'res4b16', 'res4b17', 'res4b18',
                                    'res4b19','res4b20', 'res4b21', 'res4b22', 'res4b23', 'res4b24',
                                    'res4b25','res4b26', 'res4b27', 'res4b28', 'res4b29', 'res4b30',
                                    'res4b31','res4b32', 'res4b33', 'res4b34', 'res4b35', 'res5a',
                                    'res5b','res5c'
                                    ]
        intermediate_outputs = []
        add_layers = [layer for layer in base_model.layers if layer.name in intermediate_layers_name]
        print("===================================================base_model layers==============================================")
        # layer_names = []
        # for l in base_model.layers:
        #     layer_names.append(l.name)
        # print(layer_names)
        print("intermediate layers outputs length: ", len(add_layers))
        print("===================================================base_model layers==============================================")

        for layer in add_layers:
            w = layer.output
            name = layer.name
            w = GlobalAveragePooling2D()(w)
            # add dropout before softmax to avoid overfit
            w = Dropout(self.dropout_rate)(w)
            w = Dense(self.num_classes, activation='softmax', name='intermediate_' + name)(w)
            intermediate_outputs.append(w)

        if flag_num_intermediate_layers == 0:
            # not add any intermediate layers, only final output,
            intermediate_outputs = []

        elif flag_num_intermediate_layers == 1:
            # there are total 16 resblock, and we pick resblock starting from 8th to 16th.
            start_layer_index = 8
            intermediate_outputs = intermediate_outputs[start_layer_index:]
        
        elif flag_num_intermediate_layers == 2:
            # add all intermediate layers output
            intermediate_outputs = intermediate_outputs
        else:
            print("Elastic-ResNet152-model, add_intermediate_layers_number parameter error, it should be in [0, 1, 2]")
            exit()
        print("intermediate layers outputs number: ", len(intermediate_outputs))
        self.num_outputs = len(intermediate_outputs) + 1
        return intermediate_outputs

    def build_model(self):
        # base_model = resnet152_model(include_top=False, input_shape=self.target_size)
        base_model = resnet152_model(include_top=False, input_shape=self.target_size, pooling=None, weights_path="/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/resnet_101_152_34/resnet152_weights_tf.h5")
        model = ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)

        for layer in base_model.layers:
            layer.trainable = False

        # change adding intermediate layers according to users input parameter
        intermediate_outputs = self.add_intermediate_layers(self.add_intermediate_layers_number, base_model)

        # add original final output
        w = base_model.outputs[0]
        w = Flatten()(w)
        w = Dropout(self.dropout_rate)(w)
        final_output = Dense(self.num_classes, activation='softmax', name='final_output')(w)

        outputs = intermediate_outputs + [final_output]
        output_names = [output.name.split("/")[0] for output in outputs]
        losses = {name: 'categorical_crossentropy' for name in output_names}

        # 这里的outputs既包括了intermediate层也包括了之前final_output层
        num_outputs = len(outputs)
        print("Elastic-ResNet152-model, there are %d outputs, including intermediate layers and original final output\n" % num_outputs)

        xl = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148, 151]
        layers_weight = [1/(152-x) for x in xl]
        
        loss_layers_weights = [1] * num_outputs
        loss_weights = {name: w for name, w in zip(output_names, loss_layers_weights)}

        inputs = base_model.inputs
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=losses, optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss_weights=loss_weights, metrics=['accuracy'])
        
        return model