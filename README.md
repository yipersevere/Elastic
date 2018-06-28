# TUT-lab ThinkStation computer environment
ubuntu 16.04
anaconda with python 3.6.4  

cuda 9.0.176  
cudnn 7.0.5.15  

tensorflow 1.8.0  
pytorch 0.4.0  
keras 2.1.5, with tensorflow-GPU as backend  
mxnet 1.2.0  

torch 7  
Lua version: lua 5.3.3  
gcc --version  6.4.0
gfortran-6 (6.4.0-17ubuntu1~16.04)

GPU Nvidia TITAN Xp 12GB  


## conda virtual envs
caffe  
source activate caffe  


# Narvi zhouy computer environment
user name: zhouy  
ssh narvi.tut.fi  

module load CUDA or module use CUDA  

source activate dl  
Python 3.6.5  
cudatoolkit  9.0  -- install with conda-forge community version
cudnn 7.1.2
tensorflow-gpu 1.5.0
tensorflow 1.8.0

* even though tensorflow is 1.5 or pillow, or opencv but when I change my python from 3.6 to 3.5, then in conda env, all my third packages need to be reinstalled with -pyt35, since before it is with -py36
keras 2.1.5 with modify version to display # of ops in terminal

# mount remote folder into local folder
mkdir narvi  
sshfs zhouy@narvi.tut.fi:/home/zhouy/local_elasticnn narvi/
## umount folder
sudo umount narvi/
rm -r narvi

# frequent command used in this project based on TUT's lab computer
tensorboard folder path:  
/home/yi/anaconda3/lib/python3.6/site-packages/tensorboard

## run tensorboard:  
python /home/yi/anaconda3/lib/python3.6/site-packages/tensorboard/main.py --port=8008 --logdir=Graph/ 


## test whether tensorflow-gpu works
'''
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
'''
# request allocation resource
## srun
srun --pty -J torch --gres=gpu:1 --partition=gpu --time=3-23:59:00 --mem=40960 --ntasks=1 --cpus-per-task=16 /bin/bash -i
module load CUDA
source activate dl
/bin/bash MobileNets_alpha_0_75.sh

# background run
screen  
ctrl + z (stop current job, but not kill it)  
bg (restore suspend job and run it background)  

## submit own job
sbatch -J resnet --partition=gpu --gres=gpu:1 --mem=40960 --ntasks=1 --cpus-per-task=4 --time=6-23:59:00 ResNet_script.sh



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
#SBATCH --time=7:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
# These commands will be executed on the compute node:

# Load all modules you need below. Edit these to your needs.

#module load matlab
module load CUDA
source activate learning

# Finally run your job. Here's an example of a python script.
python train.py





## check job status
squeue
squeue -u zhouy

## cancel job
scancel 2865293  

## send file or folder to Narvi
scp -r keras-2.1.5 zhouy@narvi.tut.fi:~  

# Jakko's template code folder on narvi
/sgn-data/VisionGrp/Landmarks/Landmark_recognition  

#slurm  
https://wiki.eduuni.fi/display/tutsgn/TUT+Narvi+Cluster    