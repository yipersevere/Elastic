# cifar10
python main.py --model Elastic_ResNet101 --data cifar10 --num_classes 10 --batch_size 32 --epoch 1 --add_intermediate_layers_number 0 \
--layers_weight_change 0 --model_name CIFAR10_from_0_intermediate_resblock_Elastic_ResNet101

python main.py --model Elastic_ResNet101 --data cifar10 --num_classes 10 --epoch 100 --add_intermediate_layers_number 2 \
--layers_weight_change 1 --model_name CIFAR10_all_intermediate_resblock_Elastic_ResNet101_weight_101-Depth

python main.py --model Elastic_ResNet101 --data cifar10 --num_classes 10 --epoch 100 --add_intermediate_layers_number 2 \
--layers_weight_change 0 --model_name CIFAR10_all_intermediate_resblock_Elastic_ResNet101_weights_1

# cifar100

