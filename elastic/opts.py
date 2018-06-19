import argparse

# General model args
parser = argparse.ArgumentParser(description='keras Elastic-ResNet implementation')

parser.add_argument('--data', type=str, help="training and testing data, data=cifar10; data=cifar100", default="cifar10")
parser.add_argument('--num_classes', type=int, help="classification number, 10 or 100", default=10)
parser.add_argument('--target_size', type=tuple, help='default target size is (224,224,3)', default=(224,224,3))
parser.add_argument('--epoch', type=int, help="epoch number, default 1, set 100 or 1000", default=1)
parser.add_argument('--add_intermediate_layers_number', type=int, 
                    help="add intermediate layers, 2: all intermediate layers; "
                                                    "1: skip early intermediate layers output;"
                                                    "0 : not any intermediate layers. (default: 0)", default=0)
# parser.add_argument('--model', type=str, help="model folder, like ElasticNN-ResNet50", default="Elastic_ResNet")
# parser.add_argument('--model_name', type=str, help="exact model name", default="CIFAR10_all_intermediate_resblock_Elastic_ResNet50")

parser.add_argument('--model', type=str, help="model folder, like ElasticNN-ResNet50", default="Elastic_ResNet101")
parser.add_argument('--model_name', type=str, help="exact model name", default="CIFAR10_from_0_intermediate_resblock_Elastic_ResNet101")

parser.add_argument('--dropout_rate', type=float, help="dropout rate, (default: 0.2)", default=0.2)
parser.add_argument('--batch_size', type=int, help="batch size for training and testing, (default: 32)", default=16)

parser.add_argument('--learning_rate', type=float, help="initial learning rate (default: 1e-3)", default=1e-3)
parser.add_argument('--layers_weight_change', type=int, default=0, 
                    help="1 for giving different weights for different intermediate layers output classifiers, 0 for setting all weights are 1")


# Init Environment
args = parser.parse_args()



