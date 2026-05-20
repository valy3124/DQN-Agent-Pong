# RL-Activity 1: DQN on Pong

This repository contains the training and evaluation code for solving the `PongNoFrameskip-v4` environment using Stable Baselines3.

## Structure 

- `train.py`: The core script implementing the SB3 DQN with `AtariWrapper` and `Wandb` logging (Part 1).
- `train_dqn.sh`: A SLURM script to dispatch the hyperparameter experiments over a compute node.
- `eval.py`: The script to evaluate a trained model, report rewards, and export GIFs (Part 2).
- `requirements.txt`: Environment dependencies.

## Installation & Setup

1. Ensure you have the proper virtual environment active, or install the required packages:
```bash
pip install -r requirements.txt
```

2. Authentication with Weights & Biases (WandB) is required for logging:
```bash
wandb login
```

## Part 1: Training a DQN agent on Pong

To run the training for the DQN agent, you can execute the Python script `train.py` directly. This script sets up the `AtariWrapper` and environment, defines the `DQN` model, and executes the training loop. 

**Run a custom training locally:**
```bash
python train.py \
    --run_name "dqn_pong_test" \
    --timesteps 1000000 \
    --learning_rate 0.0001 \
    --exploration_fraction 0.1 \
    --batch_size 32
```
All hyperparameters are adjustable via command-line arguments. Metrics, models, and graphs will automatically sync to your WandB account, and checkpoints will be saved locally inside the `models/` directory.

**Run the hyperparameter search on SLURM:**
To run the configured hyperparameter search via jobs, submit the bash script:
```bash
sbatch train_dqn.sh
```

## Part 2: Testing the trained agent

To evaluate the best trained model (or any provided model) and generate the requested gameplay animations (GIFs), use `eval.py`.

The script will:
1. Load the model and run 100 independent episodes.
2. Output the empirical mean reward and standard deviation.
3. Automatically render and save the **best** and **worst** performing episodes as GIFs in the `gifs/` folder.

**Run the evaluation script:**
```bash
python eval.py --model_path models/dqn_pong_1M_baseline/dqn_pong_final.zip --episodes 100
```
