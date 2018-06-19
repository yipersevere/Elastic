# -*- coding: utf-8 -*-
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


sys.setrecursionlimit(3000)

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet152_model(include_top=False, input_shape=(224, 224, 3), pooling=None, weights_path=None):
    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=input_shape, name='data')
        # img_input = Input(shape=(224, 224, 3), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(3, 224, 224), name='data')
            
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)

    # x_fc = Flatten()(x_fc)
    # x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    if include_top:
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    else:
        if pooling == 'avg':
            x_fc = GlobalAveragePooling2D()(x_fc)
        elif pooling == 'max':
            x_fc = GlobalMaxPooling2D()(x_fc)
    
    
    model = Model(img_input, x_fc)
    
    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model


def resnet101_model(include_top=False, input_shape=(224, 224, 3), pooling=None, weights_path=None):
    '''Instantiate the ResNet101 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(224, 224, 3), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,3):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)

    # x_fc = Flatten()(x_fc)
    # x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    if include_top:
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    else:
        if pooling == 'avg':
            x_fc = GlobalAveragePooling2D()(x_fc)
        elif pooling == 'max':
            x_fc = GlobalMaxPooling2D()(x_fc)
    model = Model(img_input, x_fc)
    # load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model


class ElasicNN_ResNet152():
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
        self.layers_weight_change = args.layers_weight_change
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
                                    'res5b'
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
        base_model = resnet152_model(include_top=False, input_shape=self.target_size, pooling=None, weights_path="./resnet_101_152_34/resnet152_weights_tf.h5")

        # base_model = resnet152_model(include_top=False, input_shape=self.target_size, pooling=None, weights_path="/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/resnet_101_152_34/resnet152_weights_tf.h5")

        for layer in base_model.layers:
            layer.trainable = False

        # change adding intermediate layers according to users input parameter
        intermediate_outputs = self.add_intermediate_layers(self.add_intermediate_layers_number, base_model)
        # since final output classifier and last intermediate layers output classifier are basically same, so when we use weight chanaging policy, we just use final output and drop last intermediate layers
        #intermediate_outputs = intermediate_outputs[:-2]
        #self.num_outputs = self.num_outputs - 1

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

        if self.add_intermediate_layers_number == 2:
            if self.layers_weight_change == 1:
                # loss weight changing for different depth intermediate layer output classifiers
                xl = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148]
                intermediate_layers_weight = [round(1/(152-x), 3) for x in xl]
                loss_layers_weights = intermediate_layers_weight + [1]    #[1] is for the final output layer                
            elif self.layers_weight_change == 0:
                # loss weights are 1 for all intermediate layers output classifiers
                loss_layers_weights = [1] * num_outputs
            else:
                print("Parameter --layers_weight_change, Error")
                sys.exit()
        
        else:
            loss_layers_weights = [1] * num_outputs

        if len(loss_layers_weights) == num_outputs and num_outputs == self.num_outputs:
            print("# of classifier == # of weight list == self.num_outputs")
        else:
            print("# of classifier != # of weight list, ERROR")
            sys.exit()
        
        # loss_layers_weights = [1] * num_outputs
        loss_weights = {name: w for name, w in zip(output_names, loss_layers_weights)}

        inputs = base_model.inputs
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=losses, optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss_weights=loss_weights, metrics=['accuracy'])
        
        return model


class ElasicNN_ResNet101():
    '''
    modify ResNet101 model with elastic idea, adding intermediate layers into end of this model
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
        self.layers_weight_change = args.layers_weight_change
        # # build neural network
        self.model = self.build_model()

    def add_intermediate_layers(self, flag_num_intermediate_layers, base_model):

        intermediate_layers_name = []
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
            print("Elastic-ResNet101-model, add_intermediate_layers_number parameter error, it should be in [0, 1, 2]")
            exit()
        print("intermediate layers outputs number: ", len(intermediate_outputs))
        self.num_outputs = len(intermediate_outputs) + 1
        return intermediate_outputs

    def build_model(self):
        # base_model = resnet152_model(include_top=False, input_shape=self.target_size)
        base_model = resnet101_model(include_top=False, input_shape=self.target_size, pooling=None, weights_path="/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/resnet_101_152_34/resnet101_weights_tf.h5")

        for layer in base_model.layers:
            layer.trainable = False

        # change adding intermediate layers according to users input parameter
        intermediate_outputs = self.add_intermediate_layers(self.add_intermediate_layers_number, base_model)
        # since final output classifier and last intermediate layers output classifier are basically same, so when we use weight chanaging policy, we just use final output and drop last intermediate layers
        #intermediate_outputs = intermediate_outputs[:-2]
        #self.num_outputs = self.num_outputs - 1

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
        print("Elastic-ResNet101-model, there are %d outputs, including intermediate layers and original final output\n" % num_outputs)

        if self.add_intermediate_layers_number == 2:
            if self.layers_weight_change == 1:
                # loss weight changing for different depth intermediate layer output classifiers
                xl = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139, 142, 145, 148]
                intermediate_layers_weight = [round(1/(152-x), 3) for x in xl]
                loss_layers_weights = intermediate_layers_weight + [1]    #[1] is for the final output layer                
            elif self.layers_weight_change == 0:
                # loss weights are 1 for all intermediate layers output classifiers
                loss_layers_weights = [1] * num_outputs
            else:
                print("Parameter --layers_weight_change, Error")
                sys.exit()
        
        else:
            loss_layers_weights = [1] * num_outputs

        if len(loss_layers_weights) == num_outputs and num_outputs == self.num_outputs:
            print("# of classifier == # of weight list == self.num_outputs")
        else:
            print("# of classifier != # of weight list, ERROR")
            sys.exit()
        
        # loss_layers_weights = [1] * num_outputs
        loss_weights = {name: w for name, w in zip(output_names, loss_layers_weights)}

        inputs = base_model.inputs
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=losses, optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss_weights=loss_weights, metrics=['accuracy'])
        
        return model