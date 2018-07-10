import pandas as pd
import os

note_comment = "* start counting intermediate layer from 0"
plot_save_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/plot"
y_label_str = "test error (%)"
fig_title_str_cifar10 = "classification error on CIFAR-10"
fig_title_str_cifar100 = "classification error on CIFAR-100"
x_label_str = "epoch"


def mobileNet_cifar100():
    # plot mobilenet error rate
    error_origin = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/mobileNet/Train_MobileNet/CIFAR100/Accuracy/MobileNet_CIFAR100_2018-05-16-15-42-26/accuracies.txt', delim_whitespace=True, header=None)
    error_elastic = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/mobileNet/Train_MobileNet/CIFAR100/Accuracy/Elastic-MobileNet_alpha_0.75_CIFAR100_2018-05-16-22-47-32/accuracies.txt', delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,12,13]
    folder = plot_save_folder
    captionStrDict = {
        "save_file_name" : folder + os.sep + "CIFAR_100_accuracy_Elastic&Original_MobileNet.pdf",
        "fig_title" : fig_title_str_cifar100,
        "x_label" : x_label_str,
        "y_label" : y_label_str,
        'elastic_final_layer_label': "Elastic_MobileNet_Final_Layer_Output_Classifier",
        "elastic_intermediate_layer_label" : "Elastic_MobileNet_Intermediate_Layer_Classifier_",
        "original_layer_label" : "Original_MobileNet"        
    }

    return error_origin, error_elastic, layer_plot_index, captionStrDict

def mobileNet_cifar10():
    # plot mobilenet error rate
    error_origin = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/Elastic_MobileNets_alpha_0_75/Classification_Accuracy/CIFAR10_0_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75/2018-06-03-16-41-53/accuracies.txt', delim_whitespace=True, header=None)
    error_elastic = pd.read_table('/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/Elastic_MobileNets_alpha_0_75/Classification_Accuracy/CIFAR10_all_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75/2018-06-02-22-21-37/accuracies.txt', delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,12,13]#13th layer is the final output layer, 12th is the last intermediate layer in mobilenet
    folder = plot_save_folder
    captionStrDict = {
        "save_file_name" : folder + os.sep + "CIFAR_10_accuracy_Elastic&Original_MobileNet.pdf",
        "fig_title" : fig_title_str_cifar10,
        "x_label" : x_label_str,
        "y_label" : y_label_str,
        'elastic_final_layer_label': "Elastic_MobileNet_Final_Layer_Output_Classifier",
        "elastic_intermediate_layer_label" : "Elastic_MobileNet_Intermediate_Layer_Classifier_",
        "original_layer_label" : "Original_MobileNet"        
    }

    return error_origin, error_elastic, layer_plot_index, captionStrDict


def inceptionv3_cifar100():
    origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Train_InceptionV3/CIFAR100/Accuracy/InceptionV3_CIFAR100_2018-05-08-23-21-28/accuracies.txt"
    elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Pretrained_ElasticNN/CIFAR100/AGE/InceptionV3_CIFAR100_2018-05-09-01-31-06/accuracies.txt"
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,11]#11th layer is the final output layer, 10th is the last intermediate layer in mobilenet
    folder = plot_save_folder
    captionStrDict = {
        "save_file_name" : folder + os.sep + "CIFAR_100_accuracy_Elastic&Original_InceptionV3.pdf",
        "fig_title" : fig_title_str_cifar100,
        "x_label" : x_label_str,
        "y_label" : y_label_str,
        'elastic_final_layer_label': "Elastic_InceptionV3_Final_Layer_Output_Classifier",
        "elastic_intermediate_layer_label" : "Elastic_InceptionV3_Intermediate_Layer_Classifier_",
        "original_layer_label" : "Original_InceptionV3"
    }
    return error_origin, error_elastic, layer_plot_index, captionStrDict

def inceptionv3_cifar10():
    origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/inceptionV3/Train_InceptionV3/CIFAR10/Accuracy/InceptionV3_CIFAR10_2018-05-15-21-23-44/accuracies.txt"
    elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/ElasticNN/trainModel_withCIFAR/elastic/Elastic_InceptionV3/Classification_Accuracy/CIFAR10_all_intermediate__mixedLayers_Elastic_InceptionV3/2018-06-20-16-16-29/accuracies.txt"
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    layer_plot_index = [0,2,4,6,8,10,11]
    folder = plot_save_folder
    captionStrDict = {
        "save_file_name" : folder + os.sep + "CIFAR_10_accuracy_Elastic&Original_InceptionV3.pdf",
        "fig_title" : fig_title_str_cifar10,
        "x_label" : x_label_str,
        "y_label" : y_label_str,
        'elastic_final_layer_label': "Elastic_InceptionV3_Final_Layer_Output_Classifier",
        "elastic_intermediate_layer_label" : "Elastic_InceptionV3_Intermediate_Layer_Classifier_",
        "original_layer_label" : "Original_InceptionV3"
    }
    return error_origin, error_elastic, layer_plot_index, captionStrDict



def pytorch_cifar10_loss():
    origin_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/temp/narvi-ResNet50-pytorch-CIFAR10-log-temp/test_losses.txt"
    elastic_file = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/temp/narvi-ResNet50-pytorch-CIFAR10-log-temp/train_losses.txt"
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    layer_plot_index = [0] 
    folder = plot_save_folder
    captionStrDict = {
        "save_file_name" : folder + os.sep + "Pytorch_CIFAR_10_train_test_epoch_Loss_Original_ResNet50.pdf",
        "fig_title" : "Pytorch_CIFAR_10_Original_ResNet50",
        "x_label" : "epochs",
        "y_label" : "sum loss",
        'elastic_final_layer_label': "train_loss",
        "elastic_intermediate_layer_label" : "Elastic_ResNet-152_Intermediate_Layer_Classifier_",
        "original_layer_label" : "test_loss"
    }
    return error_origin, error_elastic, layer_plot_index, captionStrDict


def pytorch_ResNet(model, data):
    if data == "CIFAR10" or data == "cifar10":
        fig_title_str = fig_title_str_cifar10

        if model == "ResNet18" or model == "resnet18":
            NotImplementedError
        elif model == "ResNet34" or model == "resnet34":
            NotImplementedError
        elif model == "ResNet50" or model == "resnet50":
            NotImplementedError
        elif model == "ResNet101" or model == "resnet101":
            NotImplementedError
        elif model == "ResNet152" or model == "resnet152":
            NotImplementedError
        else:
            NotImplementedError

    elif data == "CIFAR100" or data == "cifar100":
        fig_title_str = fig_title_str_cifar100

        if model == "ResNet18" or model == "resnet18":
            # CIFAR 100, ResNet 18
            folder = "/home/yi/narvi/elastic/pytorch_code/Elastic_ResNet18/Classification_Accuracy/"
            origin_file = folder + os.sep + "pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet18_include_pretrain/2018-07-10-01-10-19/test_errors.txt"
            elastic_file = folder + os.sep + "pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet18_include_pretrain/2018-07-10-01-10-53/test_errors.txt"
            fig_title_prefix = "Elastic ResNet 18 "
            save_file_name = "Pytorch_CIFAR_100_accuracy_Elastic&Original_ResNet18"
            layer_plot_index = [0,1,2,3,4,5,6,7,8]
            original_layer_label = "Original_ResNet-18"
            
        elif model == "ResNet34" or model == "resnet34":
            folder = "/home/yi/narvi/elastic/pytorch_code/Elastic_ResNet34/Classification_Accuracy/"
            origin_file = folder + os.sep + ""
            elastic_file = folder + os.sep + "pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet34_include_pretrain/2018-07-09-11-40-40/test_errors.txt"
            fig_title_prefix = "Elastic ResNet 34 "
            save_file_name = "Pytorch_CIFAR_100_accuracy_Elastic&Original_ResNet34"
            layer_plot_index = [0,4,6,8,10,12,15,16]
            original_layer_label = "Original_ResNet-34"            
            
        elif model == "ResNet50" or model == "resnet50":
            folder = "/home/yi/narvi/elastic/pytorch_code/Elastic_ResNet50/Classification_Accuracy/"
            origin_file = folder + os.sep + "pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet50_include_pretrain/2018-07-09-10-06-37/test_errors.txt"
            elastic_file = folder + os.sep + "pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet50_include_pretrain/2018-07-09-10-06-41/test_errors.txt"
            fig_title_prefix = "Elastic ResNet 50 "
            save_file_name = "Pytorch_CIFAR_100_accuracy_Elastic&Original_ResNet50"
            layer_plot_index = [0,4,6,8,10,12,15,16]
            original_layer_label = "Original_ResNet-50"
        elif model == "ResNet101" or model == "resnet101":
            folder = "/home/yi/narvi/elastic/pytorch_code/Elastic_ResNet101/Classification_Accuracy/"
            origin_file = folder + os.sep + "pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet101_include_pretrain/2018-07-09-11-37-55/test_errors.txt"
            elastic_file = folder + os.sep + "pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet101_include_pretrain/2018-07-09-11-38-00/test_errors.txt"
            fig_title_prefix = "Elastic ResNet 101 "
            save_file_name = "Pytorch_CIFAR_100_accuracy_Elastic&Original_ResNet101"
            layer_plot_index = [0,5,10,15,20,25,30,32,33]
            original_layer_label = "Original_ResNet-101"

        elif model == "ResNet152" or model == "resnet152":
            NotImplementedError            
        else:
            NotImplementedError
    else:
        print("data should be CIFAR 10 or CIFAR 100")
        NotImplementedError
    
    error_origin = pd.read_table(origin_file, delim_whitespace=True, header=None)
    error_elastic = pd.read_table(elastic_file, delim_whitespace=True, header=None) 
    
    save_folder = "/media/yi/e7036176-287c-4b18-9609-9811b8e33769/Elastic/elastic/pytorch_code/plot"

    captionStrDict = {
        "save_file_name" : save_folder + os.sep + save_file_name +".pdf",
        "save_png_file_name" : save_folder + os.sep + save_file_name +".png",
        "fig_title" : fig_title_prefix + fig_title_str,
        "x_label" : x_label_str,
        "y_label" : y_label_str,
        'elastic_final_layer_label': "Final_Output_Classifier",
        "elastic_intermediate_layer_label" : "Intermediate_Classifier_",
        "original_layer_label" : original_layer_label
    }
    return error_origin, error_elastic, layer_plot_index, captionStrDict    