#!/bin/bash

# Activate the Conda environment
source /fs/classhomes/fall2024/cmsc848k/c848k062/miniconda3/bin/activate obito_env

# Navigate to the project directory
cd /fs/classhomes/fall2024/cmsc848k/c848k062/ObitoNet

# Load required modules
module add cuda/11.8.0
module add gcc/11.2.0

# Run the main script with the specified configuration
python main.py --config configs/config.yaml --start_ckpt_epoch 2 --exp_name nexustest

# Print success message
echo "ran successfully"
