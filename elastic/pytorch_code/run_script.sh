# ResNet-152
    ## cifar10, 
        ### imagenet pretrained weight
python main.py --model Elastic_ResNet152 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet152_include_pretrain

python main.py --model Elastic_ResNet152 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet152_include_pretrain
    
        ### No imagenet pretrained weight

    # cifar100
python main.py --model Elastic_ResNet152 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet152

python main.py --model Elastic_ResNet152 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet152


# ResNet-101
    ## cifar10
        ### imagenet pretrained weight
python main.py --model Elastic_ResNet101 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet101

python main.py --model Elastic_ResNet101 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet101
    
        ### No imagenet pretrained weight

    # cifar100
python main.py --model Elastic_ResNet101 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet101

python main.py --model Elastic_ResNet101 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet101



# ResNet-50
    ## cifar10, 
        ### imagenet pretrained weight
python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet50

python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet50
    
        ### No imagenet pretrained weight

    # cifar100
python main.py --model Elastic_ResNet50 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet50_diff_BP_two_loops

python main.py --model Elastic_ResNet50 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet50




# ResNet-34
    ## cifar10, 
        ### imagenet pretrained weight
python main.py --model Elastic_ResNet34 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet34

python main.py --model Elastic_ResNet34 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet34
    
        ### No imagenet pretrained weight

    # cifar100
python main.py --model Elastic_ResNet34 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet34

python main.py --model Elastic_ResNet34 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet34


# ResNet-18
    ## cifar10, 
        ### imagenet pretrained weight
python main.py --model Elastic_ResNet18 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet18

python main.py --model Elastic_ResNet18 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet18
    
        ### No imagenet pretrained weight

    # cifar100
python main.py --model Elastic_ResNet18 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet18

python main.py --model Elastic_ResNet18 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_classifiers_Elastic_ResNet18



# VGG16_bn
python main.py --model Elastic_VGG16 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_Elastic_VGG16_diff_BP

python main.py --model Elastic_VGG16 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_Elastic_VGG16_diff_BP

python main.py --model Elastic_VGG16 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_Elastic_VGG16_include_pretrain_skip_last_interCLF

python main.py --model Elastic_VGG16 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_Elastic_VGG16_include_pretrain_skip_last_interCLF


# InceptionV3
    #CIFAR100
python main.py --model Elastic_InceptionV3 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_Elastic_InceptionV3_diff_BP_two_loop

python main.py --model Elastic_InceptionV3 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_Elastic_InceptionV3_include_pretrain_skip_last_interCLF

    #CIFAR10
python main.py --model Elastic_InceptionV3 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_Elastic_InceptionV3_include_pretrain_skip_last_interCLF

python main.py --model Elastic_InceptionV3 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_Elastic_InceptionV3_include_pretrain_skip_last_interCLF



#SqueezeNet
    #CIFAR100
python main.py --model Elastic_SqueezeNet --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --learning_rate　1e-3 --add_intermediate_layers 2 --layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_Elastic_SqueezeNet_include_pretrain_skip_last_interCLF_LR

python main.py --model Elastic_SqueezeNet --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --learning_rate　1e-3 --add_intermediate_layers 2 --layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_Elastic_SqueezeNet_diff_BP_two_loop

python main.py --model Elastic_SqueezeNet --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_Elastic_SqueezeNet_include_pretrain_skip_last_interCLF
    
    #CIFAR10
python main.py --model Elastic_SqueezeNet --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_Elastic_SqueezeNet_include_pretrain_skip_last_interCLF

python main.py --model Elastic_SqueezeNet --data cifar10 --num_classes 10 --batch_size 32 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_Elastic_SqueezeNet_include_pretrain_skip_last_interCLF_batch32

#MobileNet
    #CIFAR10
python main.py --model Elastic_MobileNet --data cifar10 --num_classes 10 --batch_size 16 --epochs 1 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_Elastic_MobileNet_include_pretrain_skip_last_interCLF

python main.py --model Elastic_MobileNet --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_Elastic_MobileNet_include_pretrain_skip_last_interCLF

    #CIFAR 100
python main.py --model Elastic_MobileNet --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_Elastic_MobileNet_include_pretrain_skip_last_interCLF

python main.py --model Elastic_MobileNet --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_0_intermediate_Elastic_MobileNet_include_pretrain_skip_last_interCLF

#DenseNet121
    #CIFAR10
python main.py --model Elastic_DenseNet121 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_Elastic_DenseNet121_include_pretrain_skip_last_interCLF

python main.py --model Elastic_DenseNet121 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_Elastic_DenseNet121_include_pretrain_skip_last_interCLF

    #CIFAR100
......
......

#DenseNet169
    #CIFAR10
python main.py --model Elastic_DenseNet169 --data cifar10 --num_classes 10 --batch_size 16 --epochs 1 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_Elastic_DenseNet169_include_pretrain_skip_last_interCLF

python main.py --model Elastic_DenseNet169 --data cifar10 --num_classes 10 --batch_size 16 --epochs 100 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_Elastic_DenseNet169_include_pretrain_skip_last_interCLF