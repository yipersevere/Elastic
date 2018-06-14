#!/bin/bash
#
#SBATCH -J resnet
#
#SBATCH --output=log/
#SBATCH --error=log/
#
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20480
#SBATCH --time=3:59:00
#SBATCH --partition=gpu

module load CUDA
source /home/opt/anaconda3/bin/activate learning

python main.py --model Elastic_MobileNets_alpha_0_75 --data cifar10 --num_classes 10 --epoch 1 --add_intermediate_layers_number 2 --model_name CIFAR10_all_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75

# to run ElasticNN-ResNet50 cifar10, add all intermediate layers， 1 epoch testing
# python main.py --model Elastic_ResNet --data cifar10 --num_classes 10 \
# --epoch 1 --add_intermediate_layers_number 2 --model_name CIFAR10_all_intermediate_resblock_Elastic_ResNet50



# to run ElasticNN-ResNet50 cifar10, add all intermediate layers， 1 epoch testing
#python main.py --model Elastic_ResNet --data cifar10 --num_classes 10 \
#--epoch 1 --add_intermediate_layers_number 1 --model_name CIFAR10_from_8th_intermediate_resblock_Elastic_ResNet50 \


# to run ElasticNN-ResNet50 cifar10, add 0 intermediate layers， 1 epoch testing
#python main.py --model Elastic_ResNet --data cifar10 --num_classes 10 \
#--epoch 1 --add_intermediate_layers_number 0 --model_name CIFAR10_0_intermediate_resblock_Elastic_ResNet50



# to run ElasticNN-ResNet50 cifar100, add all intermediate layers， 1 epoch testing
#python main.py --model Elastic_ResNet --data cifar100 --num_classes 100 \
#--epoch 100 --add_intermediate_layers_number 2 --model_name CIFAR100_all_intermediate_resblock_Elastic_ResNet50






