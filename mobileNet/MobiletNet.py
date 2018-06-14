from helper import load_data, multi_output_generator, log_summary, log_error
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, ModelCheckpoint, \
    LearningRateScheduler
import os
import time
import datetime
from keras.layers.pooling import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import math


# global variables
batch_size = 32
epoch_size = 100
init_epochs = 100 #数据迭代的轮数
logFile = ""
steps = math.ceil(len(X_train) / batch_size)

X_train, y_train, X_val, y_val, x_test, y_test = load_data()

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5, cooldown=1)
early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=15, verbose=0, mode='auto')



def build_model():
    """
    build model
    """
    # 从keras中下载MobileNet model, 然后根据ElasticNN论文中的描述, 对MobileNet模型修改
    # "3" 是指 三通道, 即 RGB
    base_model = MobileNet(include_top=False, alpha=0.75, input_shape=target_size)

    # 冻结base_model原本模型的神经网络层，只有新添加的层才能训练
    for layer in base_model.layers:
        layer.trainable = False
        
    w = base_model.outputs[0]
    w = Flatten()(w)

    final_output = Dense(num_classes, activation='softmax', name='final_output')(w)

    add_layers = []
    for layer in base_model.layers:
        if '_pw_' in layer.name:
            if 'relu' in layer.name:
                add_layers.append(layer)

    intermediate_outputs = []

    for layer in add_layers:
        w = layer.output
        name = layer.name
        w = GlobalAveragePooling2D()(w)
        w = Dense(101, activation='sigmoid', name='intermediate_' + name)(w)
        intermediate_outputs.append(w)
    

    inputs = base_model.inputs

    outputs = intermediate_outputs + [final_output]

    # 开始定义自己的神经网络模型
    model = Model(inputs=inputs, outputs=outputs)

    output_names = [output.name.split("/")[0] for output in outputs]
    losses = {name: 'categorical_crossentropy' for name in output_names}

    num_outputs = len(outputs)
    print("there are %d outputs.\n" % num_outputs)
    loss_weights = [0] * num_outputs
    loss_weights[6:num_outputs] = [1]*len(loss_weights[6:num_outputs])
    loss_weights = {name: w for name, w in zip(output_names, loss_weights)}

    model.compile(loss=losses, optimizer=SGD(lr=1e-3), loss_weights=loss_weights, metrics=['accuracy'])
    
    return model


def rebuild_model():
    # train all model layers which means including all previsous-frozed layers before.
    for layer in model.layers:
        layer.trainable = True
    
    # 经过训练之后， 再次重新编译神经网络模型
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-2), metrics=['accuracy'])
    
    LOG("====================Pretraining all layers, with including all previours frozened layers====================")
    checkpointer2 = ModelCheckpoint(filepath= path + 'model2.best.hdf5', verbose=1, save_best_only=True)
    model.fit_generator(train_generator,
                        epochs=init_epochs,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]),
                        callbacks=[checkpointer2,lr_reducer, early_stop])


def train(path):
    logFile = path + os.sep + "log.txt"

    model = build_model()

    # Print the Model summary
    log_summary(model)
    
    train_datagen = ImageDataGenerator(horizontal_flip=False, data_format=K.image_data_format())
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    multi_generator = multi_output_generator(train_generator, num_outputs)

    checkpoint_file_path = path + 'model.best.hdf5'
    checkpointer = ModelCheckpoint(filepath= checkpoint_file_path , verbose=1, save_best_only=True)
    
    
    LOG("===================Pretraining the new layer for %d epochs==================" % init_epochs)
    # 这里的意思是训练最后一层全连接层的权重，但是没有包括之前forze
    model.fit_generator(multi_output_generator,
                        epochs=init_epochs,
                        steps_per_epoch=steps,
                        verbose=0,
                        validation_data=(X_val, [y_val]),
                        callbacks=[checkpointer,lr_reducer, early_stop])
    return 0


def eval():
    # load the weights that yielded the best validation accuracy
    model.load_weights(path+'model.best.hdf5')

    # evaluate test accuracy
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = 100*score[1]

    # print test accuracy
    print('previous layers are not trainable, Test accuracy: %.4f%%' % accuracy)

    # print test accuracy
    print('all layers are trainable, Test accuracy: %.4f%%' % accuracy)
    return 0


if __name__ == "__main__":

    instanceName = "AGE"
    model_name = "hl_MOBILE_0.75_"
    timestamp = datetime.datetime.now()
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    path = "./Pretrained_ElasticNN_MobileNet_CIFAR100" + os.sep + instanceName + os.sep + model_name + "_" + ts_str
    os.makedirs(path)
    start_time = time.time()
    train(path)
    print("train with frozening all layers nontrainable, time : " time.time()-start_time, " s")
    eval()

    # here, train again, but with all layers are set trainable 
    start_time = time.time()
    train()
    print("train, all layers are set trainable, time : " time.time()-start_time, " s")
    # then again, eval again.
    eval()