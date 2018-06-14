#!/bin/bash
#
# At first let's give job some descriptive name to distinct
# from other jobs running on cluster
#SBATCH -J elasticNN
#
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=log/output.txt
#SBATCH --error=log/error.txt
#
# We'll want to allocate one CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#
# We'll want to reserve 2GB memory for the job
# and 3 days of compute time to finish.
# Also define to use the GPU partition.
#SBATCH --mem=40960
#SBATCH --time=5:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#

#SBATCH --mail-user=yi.zhou@tut.fi
#SBTACH --mail-type=BEGIN, END, FAIL
# These commands will be executed on the compute node:

# Load all modules you need below. Edit these to your needs.

module load CUDA
source activate learning

# Finally run your job. Here's an example of a python script.
python main.py --model Elastic_MobileNets_alpha_0_75 --data cifar10 --num_classes 10 --epoch 1 --add_intermediate_layers_number 2 --model_name CIFAR10_all_intermediate_pw_relu_Elastic_MobileNets_alpha_0_75
