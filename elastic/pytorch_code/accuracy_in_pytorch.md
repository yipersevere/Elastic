# Accuracy in Pytorch  

# classification on **CIFAR-10**

model                                                                     | error (best)     
--------------------------------------------------------------------------| ------------------- 
ResNet-18-include-pretrain                                                | 7.6900
Elastic_ResNet-18-include-pretrain                                        | 12.0300
ResNet-34-No-include-pretrain                                             | 4.9100
Elastic_ResNet-34-No-include-pretrain                                     | 11.0700  
ResNet-50-No-include-pretrain                                             | 5.0700  
Elastic_ResNet-50-No-include-pretrain                                     | 9.1000  
ResNet-101-include-pretrain                                               | 8.5900
Elastic_ResNet-101-include-pretrain                                       | 10.4900
ResNet-101-include-pretrain                                               | 8.5900
Elastic_ResNet-101-include-pretrain                                       | 10.4900

### SqueezeNet
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
SqueezeNet-include-pretrain                                               | 
Elastic_SqueezeNet-include-pretrain                                       |

### DenseNet
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
DenseNet-121-include-pretrain                                             | 10.6800
Elastic_DenseNet-121-include-pretrain                                     | 8.5500
Elastic_DenseNet-121-include-pretrain_BP_two_loop                         | 8.5200
Elastic_DenseNet-121-include_pretrain_retain_graph                        | 


### VGG16_bn
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
VGG16_include_pretrain                                                    | 33.2800
Elastic_VGG16_include_pretrrain                                           | 
Elastic_VGG16_include_pretrain_BP_two_loop                                | 33.2800(until 70 epoches)
Elastic_VGG16_include_pretrain_retain_graph                               | 32.3500

### MobileNet
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
MobileNet_include_pretrain
Elastic_MobileNet_include_pretrain                                        |
Elastic_MobileNet_include_pretrain_BP_two_loop                            |
Elastic_MobileNet_include_pretrain_retain_graph                           |


reference  
**pretrain** means using 10 epochs to train **only** final classifier and intermediate classifiers first  


*[1] second to last classifier gets better result than the last classifier
* SqueezeNet, initial learning_rate  = 1e-3, not 1e-2

暂时的结论：
在ResNet18-No-include-pretrain 中，精确度会比 ResNet18-include-pretrain 中会高 1%, 这个在
Elastic_ResNet-18-include-pretrain-No-skip-last-interCLF 和 Elastic_ResNet-18-No-include-pretrain-No-skip-last-interCLF 也发生了，即高 2%


# classification on **CIFAR-100**

model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
ResNet-18-No-include-pretrain                                             | 25.1800  
Elastic_ResNet-18-No-include-pretrain                                     | 30.9500 
ResNet-34-No-include-pretrain                                             | 22.4600 
Elastic_ResNet-34-No-include-pretrain                                     | 33.1300  
ResNet-50-No-include-pretrain                                             | 23.6600   
Elastic_ResNet-50-No-include-pretrain                                     | 29.1100 
ResNet-101-include-pretrain                                               | 22.3300   
Elastic_ResNet-101-include-pretrain                                       | 38.1000  


### SqueezeNet
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
SqueezeNet-include-pretrain                                               | 29.1700 
Elastic_SqueezeNet-include-pretrain                                       | 38.4400
Elastic_SqueezeNet-include-pretrain_BP_two_loop                           | 32.5900(End with 20 epoches, anormal)


### DenseNet
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
DenseNet-121-include-pretrain                                             | 10.6800
Elastic_DenseNet-121-include-pretrain                                     | 8.5500
Elastic_DenseNet-121-include-pretrain_BP_two_loop                         | 8.5200
Elastic_DenseNet-121-include_pretrain_retain_graph                        | 



### VGG16_bn
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
VGG16_include_pretrain                                                    | 33.2800
Elastic_VGG16_include_pretrrain                                           | 
Elastic_VGG16_include_pretrain_BP_two_loop                                | 33.2800(until 70 epoches)
Elastic_VGG16_include_pretrain_retain_graph                               | 32.3500

### MobileNet
model                                                                     | error (best)              
--------------------------------------------------------------------------| ------------------- 
MobileNet_include_pretrain
Elastic_MobileNet_include_pretrain                                        |
Elastic_MobileNet_include_pretrain_BP_two_loop                            |
Elastic_MobileNet_include_pretrain_retain_graph                           |



[1] **pretrain** means using 10 epochs to train **only** final classifier and intermediate classifiers first 




