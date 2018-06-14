import pandas as pd


def mobileNet_cifar100():
    # plot mobilenet error rate
    error_origin = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/mobileNet/Train_MobileNet/CIFAR100/Accuracy/MobileNet_CIFAR100_2018-05-16-15-42-26/accuracies.txt', delim_whitespace=True, header=None)
    error_elastic = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/mobileNet/Train_MobileNet/CIFAR100/Accuracy/Elastic-MobileNet_alpha_0.75_CIFAR100_2018-05-16-22-47-32/accuracies.txt', delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,12,13]
    folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/plot"
    captionStrDict = {
        "save_file_name" : folder + os.sep + "accuracy_ElasticNN_MobileNet_&_Original_MobileNet_CIFAR_100.pdf",
        "fig_title" : "ElasticNN-MobileNet & Original-MobileNet test on CIFAR 100",
        "x_label" : "epoch",
        "y_label" : "error rate on CIFAR-100",
        "elastic_layer_label" : "Elastic_MobileNet_Layer_",
        "original_layer_label" : "original_MobiletNet"
    }

    return error_origin, error_elastic, layer_plot_index, captionStrDict


def resetnet50_cifar10():
    origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/ResNet50/Train_ResNet50/CIFAR10/Accuracy/ResNet_CIFAR10_2018-05-17-16-35-14/accuracies.txt"
    elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/ResNet50/Train_ElasticNN-ResNet50/CIFAR10/Accuracy/Elastic-ResNet50_CIFAR10_2018-05-17-22-54-24/accuracies.txt"
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,12,15,16]
    folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/plot"
    captionStrDict = {
        "save_file_name" : folder + os.sep + "error_rate_ResNet50_Compare_CIFAR_10.pdf",
        "fig_title" : "ElasticNN-ResNet50 & Original-ResNet50 test on CIFAR 10",
        "x_label" : "epoch",
        "y_label" : "error rate on CIFAR-10",
        "elastic_layer_label" : "Elastic_ResNet50_Layer_",
        "original_layer_label" : "original_ResNet50"
    }

    return error_origin, error_elastic, layer_plot_index, captionStrDict


def inceptionv3_cifar10():
    origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Train_InceptionV3/CIFAR10/Accuracy/InceptionV3_CIFAR10_2018-05-15-21-23-44/accuracies.txt"
    elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Pretrained_ElasticNN/CIFAR10/Accuracy/Elastic-InceptionV3_CIFAR10_2018-05-16-13-26-15/accuracies.txt"
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,11]
    folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/plot"
    captionStrDict = {
        "save_file_name" : folder + os.sep + "error_rate_InceptionV3_Compare_CIFAR_10.pdf",
        "fig_title" : "ElasticNN-InceptionV3 & Original-InceptionV3 test on CIFAR 10",
        "x_label" : "epoch",
        "y_label" : "error rate on CIFAR-10",
        "elastic_layer_label" : "Elastic_InceptionV3_Layer_",
        "original_layer_label" : "original_InceptionV3"
    }
    return error_origin, error_elastic, layer_plot_index, captionStrDict

def inceptionv3_cifar100():
    origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Train_InceptionV3/CIFAR100/Accuracy/InceptionV3_CIFAR100_2018-05-08-23-21-28/accuracies.txt"
    elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Pretrained_ElasticNN/CIFAR100/AGE/InceptionV3_CIFAR100_2018-05-09-01-31-06/accuracies.txt"
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,11]
    folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/plot"
    captionStrDict = {
        "save_file_name" : folder + os.sep + "error_rate_InceptionV3_Compare_CIFAR_100.pdf",
        "fig_title" : "ElasticNN-InceptionV3 & Original-InceptionV3 test on CIFAR 100",
        "x_label" : "epoch",
        "y_label" : "error rate on CIFAR-100",
        "elastic_layer_label" : "Elastic_InceptionV3_Layer_",
        "original_layer_label" : "original_InceptionV3"
    }
    return error_origin, error_elastic, layer_plot_index, captionStrDict


def mobileNets_alpha_0_75_F1_cifar100():
    origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/mobileNet/Train_MobileNet/CIFAR100/Accuracy/MobileNet_CIFAR100_2018-05-16-15-42-26/f1-score.txt"
    elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/mobileNet/Train_MobileNet/CIFAR100/Accuracy/Elastic-MobileNet_alpha_0.75_CIFAR100_2018-05-16-22-47-32/f1-score.txt"
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,12,13]
    folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/plot"
    captionStrDict = {
        "save_file_name" : folder + os.sep + "accuracy_ElasticNN_MobileNet_&_Original_MobileNet_CIFAR_100.pdf",
        "fig_title" : "ElasticNN-MobileNet & Original-MobileNet test on CIFAR 100",
        "x_label" : "epoch",
        "y_label" : "error rate on CIFAR-100",
        "elastic_layer_label" : "Elastic_MobileNet_Layer_",
        "original_layer_label" : "original_MobiletNet"
    }

    return error_origin, error_elastic, layer_plot_index, captionStrDict
