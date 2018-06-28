import torch
import torch.nn as nn

import  csv
import pandas as pd
import os
# checkpoint = torch.load("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/Elastic_ResNet50/Classification_Accuracy/pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet50/2018-06-24-20-56-30/save_models/model_best.pth.tar")

# model = checkpoint["model"]

# epochs_intermediate_acc_test = [[1,2,3],[4,5,6]]



# with open("./test_intermediate_accuracies.csv", "a") as fp:
#     wr = csv.writer(fp)
#     wr.writerows(epochs_intermediate_acc_test)


origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/Elastic_ResNet50/Classification_Accuracy/pytorch_CIFAR10_3_intermediate_classifiers_Elastic_ResNet50/2018-06-27-00-35-10/epochs_lr.txt"
elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/Elastic_ResNet50/Classification_Accuracy/pytorch_CIFAR10_3_intermediate_classifiers_Elastic_ResNet50/2018-06-27-00-35-10/train_intermediate_accuracies.txt"
error_origin = pd.read_table(origin_file, sep=" ", header=None)
error_elastic = pd.read_table(elastic_file, sep=" ", header=None) 

folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/temp"

error_origin.to_csv(folder + os.sep + 'epochs_lr.txt', header=None, index=None, sep='\n', mode='a')
error_elastic.to_csv(folder + os.sep + 'train_intermediate_accuracies.txt', header=None, index=None, sep='\n', mode='a')
