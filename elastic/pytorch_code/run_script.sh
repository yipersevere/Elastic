# cifar10, ResNet-50
python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --batch_size 16 --epochs 1 --add_intermediate_layers_number 0 \
--layers_weight_change 0 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet50


python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --epochs 100 --add_intermediate_layers_number 2 \
--layers_weight_change 0 --model_name pytorch_CIFAR10_all_intermediate_classifiers_Elastic_ResNet50

# cifar100， ResNet-50



# demo to run 
python main.py --model Elastic_ResNet50 --data cifar10 --num_classes 10 --batch_size 16 --epochs 2 --add_intermediate_layers_number 2 --layers_weight_change 0 --model_name pytorch_CIFAR10_0_intermediate_classifiers_Elastic_ResNet50
