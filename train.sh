#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --account=class
#SBATCH --partition=class
#SBATCH --qos high
#SBATCH -t 1-00:00:00
#SBATCH --signal=SIGUSR1@90

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_IB_DISABLE=1

cd ~/
source /fs/classhomes/fall2024/cmsc848k/c848k032/miniconda3/bin/activate obitonet
cd /fs/classhomes/fall2024/cmsc848k/c848k032/ObitoNet
module add cuda/11.8.0
module add gcc/11.2.0
srun -u python main.py --config configs/config.yaml --exp_name nexustest
echo "ran successfully"