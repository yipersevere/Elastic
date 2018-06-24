import torch
import torch.nn as nn


checkpoint = torch.load("/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/Elastic_ResNet50/Classification_Accuracy/pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet50/2018-06-24-20-56-30/save_models/model_best.pth.tar")

model = checkpoint["model"]
