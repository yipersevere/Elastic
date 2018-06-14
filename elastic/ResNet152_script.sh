#!/bin/bash
#
#SBATCH -J resnet
#
#SBATCH --output=log-output.txt
#SBATCH --error=log-error.txt
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40960
#SBATCH --time=3-23:59:00
#SBATCH --partition=gpu

module load CUDA
source /home/opt/anaconda3/bin/activate dl





# cifar10
python main.py --model Elastic_ResNet152 --data cifar10 --num_classes 10 --epoch 100 --add_intermediate_layers_number 0 \
--layers_weight_change 0 --model_name CIFAR10_from_0_intermediate_resblock_Elastic_ResNet152


python main.py --model Elastic_ResNet152 --data cifar10 --num_classes 10 --epoch 100 --add_intermediate_layers_number 2 \
--layers_weight_change 1 --model_name CIFAR10_all_intermediate_resblock_Elastic_ResNet152_weight_152-Depth

python main.py --model Elastic_ResNet152 --data cifar10 --num_classes 10 --epoch 100 --add_intermediate_layers_number 2 \
--layers_weight_change 0 --model_name CIFAR10_all_intermediate_resblock_Elastic_ResNet152_weights_1

# cifar100
python main.py --model Elastic_ResNet152 --data cifar100 --num_classes 100 --epoch 100 --add_intermediate_layers_number 0 \
--layers_weight_change 0 --model_name CIFAR100_from_0_intermediate_resblock_Elastic_ResNet152


python main.py --model Elastic_ResNet152 --data cifar100 --num_classes 100 --epoch 100 --add_intermediate_layers_number 2 \
--layers_weight_change 1 --model_name CIFAR100_all_intermediate_resblock_Elastic_ResNet152_weight_152-Depth

python main.py --model Elastic_ResNet152 --data cifar100 --num_classes 100 --epoch 100 --add_intermediate_layers_number 2 \
--layers_weight_change 0 --model_name CIFAR100_all_intermediate_resblock_Elastic_ResNet152_weights_1





