# to run ElasticNN-InceptionV3 cifar10, add all intermediate layers， 1 epoch testing
python main.py --model Elastic_MobileNets_alpha_0_75 --data cifar10 --num_classes 10 \
--epoch 1 --add_intermediate_layers_number 2 --model_name CIFAR10_all_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75


# to run ElasticNN-InceptionV3 cifar10, add 0 intermediate layers， 1 epoch testing
#python main.py --model Elastic_MobileNets_alpha_0_75 --data cifar10 --num_classes 10 \
#--epoch 1 --add_intermediate_layers_number 0 --model_name CIFAR10_0_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75


# to run ElasticNN-InceptionV3 cifar100, add all intermediate layers， 1 epoch testing
#python main.py --model Elastic_MobileNets_alpha_0_75 --data cifar100 --num_classes 100 \
#--epoch 1 --add_intermediate_layers_number 2 --model_name CIFAR100_all_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75


# to run ElasticNN-InceptionV3 cifar100, add 0 intermediate layers，1 epoch testing
#python main.py --model Elastic_MobileNets_alpha_0_75 --data cifar100 --num_classes 100 \
#--epoch 1 --add_intermediate_layers_number 0 --model_name CIFAR100_0_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75


