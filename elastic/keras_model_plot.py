#encoding=utf-8

from __future__ import absolute_import 
from __future__ import print_function

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model
import os
from keras.utils.generic_utils import CustomObjectScope
import keras



folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/Elastic_InceptionV3/Classification_Accuracy/CIFAR10_all_intermediate__mixedLayers_Elastic_InceptionV3"



model = load_model(folder + os.sep + "2018-05-31-22-48-15model.best.hdf5")

plot_model(model, to_file=folder + os.sep + 'CIFAR10_all_intermediate__mixedLayers_Elastic_InceptionV3.png',show_shapes=True)



# # only for mobilenet since mobilenet has "MobileNet uses several custom functions."-ValueError: Unknown activation function:relu6
# with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#     model = load_model(folder + os.sep + "2018-06-03-16-41-53model.best.hdf5")
#     plot_model(model, to_file=folder + os.sep + 'CIFAR10_0_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75.png',show_shapes=True)


