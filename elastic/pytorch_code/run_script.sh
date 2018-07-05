# cifar10, ResNet-50
python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --batch_size 16 --epochs 1 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet50

python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --batch_size 16 --epochs 1 --add_intermediate_layers 0 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet50

python main.py --model Elastic_ResNet101 --data cifar10 --num_classes 10 --batch_size 16 --epochs 1 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet101


python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --epochs 100 --add_intermediate_layers_number 2 \
--layers_weight_change 0 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet50

# cifar100， ResNet-50
python main.py --model Elastic_ResNet50 --data cifar100 --num_classes 100 --batch_size 16 --epochs 100 --add_intermediate_layers 2 \
--layers_weight_change 0 --pretrained_weight 1 --model_name pytorch_CIFAR100_all_intermediate_classifiers_Elastic_ResNet50


# demo to run 

