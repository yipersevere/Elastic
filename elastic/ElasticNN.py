import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam

class ElasicNN_ResNet50():
    '''
    modify ResNet50 model with elastic idea, adding intermediate layers into end of this model
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

        intermediate_outputs = []
        add_layers = [layer for layer in base_model.layers if "add_" in layer.name]

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
            print("Elastic-ResNet50-model, add_intermediate_layers_number parameter error, it should be in [0, 1, 2]")
            exit()
        self.num_outputs = len(intermediate_outputs) + 1
        return intermediate_outputs

    def build_model(self):
        base_model = ResNet50(include_top=False, input_shape=self.target_size)
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
        print("Elastic-ResNet50-model, there are %d outputs, including intermediate layers and original final output\n" % num_outputs)

        loss_layers_weights = [1] * num_outputs
        loss_weights = {name: w for name, w in zip(output_names, loss_layers_weights)}

        inputs = base_model.inputs
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=losses, optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss_weights=loss_weights, metrics=['accuracy'])
        
        return model


class ElasicNN_Inception():
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

        intermediate_outputs = []
        add_layers = [layer for layer in base_model.layers if "mixed" in layer.name and ('mixed9_0'!= layer.name and 'mixed9_1'!= layer.name )]

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
            # there are total 11 inception block, and we pick block starting from 8th to 16th.
            start_layer_index = 8
            intermediate_outputs = intermediate_outputs[start_layer_index:]
        
        elif flag_num_intermediate_layers == 2:
            # add all intermediate layers output
            intermediate_outputs = intermediate_outputs
        else:
            print("Elastic-InceptionV3 model, add_intermediate_layers_number parameter error, it should be in [0, 1, 2]")
            exit()
        self.num_outputs = len(intermediate_outputs) + 1
        return intermediate_outputs

    def build_model(self):
        base_model = InceptionV3(include_top=False, input_shape=self.target_size)
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
        print("Elastic-InceptionV3 model, there are %d outputs, including intermediate layers and original final output\n" % num_outputs)

        loss_layers_weights = [1] * num_outputs
        loss_weights = {name: w for name, w in zip(output_names, loss_layers_weights)}

        inputs = base_model.inputs
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=losses, optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss_weights=loss_weights, metrics=['accuracy'])
        
        return model


class ElasticNN_MobileNets_alpha_0_75():
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

        intermediate_outputs = []
        add_layers = []
        for layer in base_model.layers:
            if '_pw_' in layer.name:
                if 'relu' in layer.name:
                    add_layers.append(layer)

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
            # there are total 11 inception block, and we pick block starting from 8th to 16th.
            start_layer_index = 8
            intermediate_outputs = intermediate_outputs[start_layer_index:]
        
        elif flag_num_intermediate_layers == 2:
            # add all intermediate layers output
            intermediate_outputs = intermediate_outputs
        else:
            print("Elastic-MobileNets-alpha-0-75 model, add_intermediate_layers_number parameter error, it should be in [0, 1, 2]")
            exit()
        self.num_outputs = len(intermediate_outputs) + 1
        return intermediate_outputs

    def build_model(self):
        base_model = MobileNet(include_top=False, alpha=0.75, input_shape=self.target_size)
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
        print("Elastic-MobileNets-alpha-0-75 model, there are %d outputs, including intermediate layers and original final output\n" % num_outputs)

        loss_layers_weights = [1] * num_outputs
        loss_weights = {name: w for name, w in zip(output_names, loss_layers_weights)}

        inputs = base_model.inputs
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=losses, optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss_weights=loss_weights, metrics=['accuracy'])
        
        return model
    

# class ElasticNN_VGG16():
#     def __init__(self, args):
#         self.data = args.data
#         self.num_classes = args.num_classes
#         self.target_size = args.target_size
#         self.epoch = args.epoch
#         self.model_name = args.model_name
#         self.dropout_rate = args.dropout_rate
#         self.batch_size = args.batch_size
#         self.add_intermediate_layers_number = args.add_intermediate_layers_number
#         self.learning_rate = args.learning_rate
#         # # build neural network
#         self.model = self.build_model()

#     def add_intermediate_layers(self, flag_num_intermediate_layers, base_model):

#         intermediate_outputs = []
#         add_layers = []
#         add_layers = [layer for layer in base_model.layers if "_pool" in layer.name]

#         for layer in add_layers:
#             w = layer.output
#             name = layer.name
#             w = GlobalAveragePooling2D()(w)
#             # add dropout before softmax to avoid overfit
#             w = Dropout(self.dropout_rate)(w)
#             w = Dense(self.num_classes, activation='softmax', name='intermediate_' + name)(w)
#             intermediate_outputs.append(w)

#         if flag_num_intermediate_layers == 0:
#             # not add any intermediate layers, only final output,
#             intermediate_outputs = []

#         elif flag_num_intermediate_layers == 1:
#             # there are total 11 inception block, and we pick block starting from 8th to 16th.
#             start_layer_index = 8
#             intermediate_outputs = intermediate_outputs[start_layer_index:]
        
#         elif flag_num_intermediate_layers == 2:
#             # add all intermediate layers output
#             intermediate_outputs = intermediate_outputs
#         else:
#             print("Elastic-VGG16 model, add_intermediate_layers_number parameter error, it should be in [0, 1, 2]")
#             exit()
#         self.num_outputs = len(intermediate_outputs) + 1
#         return intermediate_outputs

#     def build_model(self):
#         base_model = VGG16(include_top=False, input_shape=self.target_size)
#         for layer in base_model.layers:
#             layer.trainable = False

#         # change adding intermediate layers according to users input parameter
#         intermediate_outputs = self.add_intermediate_layers(self.add_intermediate_layers_number, base_model)

#         # add original final output
#         w = base_model.outputs[0]
#         w = Flatten()(w)
#         w = Dropout(self.dropout_rate)(w)
#         final_output = Dense(self.num_classes, activation='softmax', name='final_output')(w)

#         outputs = intermediate_outputs + [final_output]
#         output_names = [output.name.split("/")[0] for output in outputs]
#         losses = {name: 'categorical_crossentropy' for name in output_names}

#         # 这里的outputs既包括了intermediate层也包括了之前final_output层
#         num_outputs = len(outputs)
#         print("Elastic-VGG16 model, there are %d outputs, including intermediate layers and original final output\n" % num_outputs)

#         loss_layers_weights = [1] * num_outputs
#         loss_weights = {name: w for name, w in zip(output_names, loss_layers_weights)}

#         inputs = base_model.inputs
#         model = Model(inputs=inputs, outputs=outputs)
#         model.compile(loss=losses, optimizer=SGD(lr=self.learning_rate, momentum=0.9), loss_weights=loss_weights, metrics=['accuracy'])
        
#         return model    