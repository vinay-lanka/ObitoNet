#!/bin/bash

#SBATCH --job-name=obitonet_distributed  # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gres=gpu:rtxa5000:4            # GPUs per node
#SBATCH --ntasks=1                       # Number of tasks per node
#SBATCH --mem-per-cpu=64gb               # Memory per CPU
#SBATCH --account=class                  # Account name
#SBATCH --partition=class                # Partition name
#SBATCH --qos=high                       # Quality of Service
#SBATCH --time=1-00:00:00                # Time limit (days-hours:minutes:seconds)
#SBATCH --signal=SIGUSR1@90              # Signal for preemption (optional)
#SBATCH --output=job_%j.out              # Standard output log
#SBATCH --error=job_%j.err               # Standard error log

# Debugging and optimization flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_IB_DISABLE=1

# Activate the conda environment
cd ~/
source /fs/classhomes/fall2024/cmsc848k/c848k032/miniconda3/bin/activate obitonet

# Navigate to the project directory
cd /fs/classhomes/fall2024/cmsc848k/c848k032/ObitoNet

# Load necessary modules
module add cuda/11.8.0
module add gcc/11.2.0

# Setup environment variables for distributed training
export MASTER_ADDR=$(hostname)           # Master address (current node)
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)  # Random port to avoid conflicts
export WORLD_SIZE=$SLURM_NTASKS          # Total number of processes
export NODE_RANK=$SLURM_NODEID           # Node rank
export OMP_NUM_THREADS=10                # Threads per process

# Disable wandb logging
# export WANDB_MODE=disabled

# Optimize CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32

# Run distributed training with torchrun
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /fs/classhomes/fall2024/cmsc848k/c848k032/ObitoNet/main.py \
    --config configs/config.yaml \
    --exp_name CA_Train_1_DISTRIBUTED

echo "ran successfully"
