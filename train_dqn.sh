#!/bin/bash
#SBATCH -n 4
#SBATCH --mem 24G
#SBATCH -p mlow
#SBATCH --gres gpu:1
#SBATCH -o logs/train_dqn_%u_%j.out
#SBATCH -e logs/train_dqn_%u_%j.err

set -e

# Automatically move to the correct directory
cd /ghome/group01/C5/vali/OPTIONAL/pong_dqn_activity

# Activate the correct virtual environment
source /ghome/group01/C5/vali/.venv/bin/activate

# Create necessary directories
mkdir -p logs models runs

echo "Starting DQN Pong training on node: $HOSTNAME"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# --- RUN 1 (BASELINE) ---
python train.py \
    --run_name "dqn_pong_1M_baseline" \
    --timesteps 1000000 \
    --learning_rate 0.0001 \
    --exploration_fraction 0.1 \
    --batch_size 32

echo "Training Run 1 Complete!"

# --- RUN 2 (Low Learning Rate & Slower Decay) ---
python train.py \
    --run_name "dqn_pong_1M_low_lr" \
    --timesteps 1000000 \
    --learning_rate 0.00005 \
    --exploration_fraction 0.2 \
    --batch_size 32

echo "Training Run 2 Complete!"

# --- RUN 3 (High Learning Rate) ---
python train.py \
    --run_name "dqn_pong_1M_high_lr" \
    --timesteps 1000000 \
    --learning_rate 0.0002 \
    --exploration_fraction 0.1 \
    --batch_size 32

echo "Training Run 3 Complete!"

# --- RUN 4 (Large Batch Size) ---
python train.py \
    --run_name "dqn_pong_1M_large_batch" \
    --timesteps 1000000 \
    --learning_rate 0.0001 \
    --exploration_fraction 0.1 \
    --batch_size 128

echo "Training Run 4 Complete!"
