import argparse
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
import os

def main():
    parser = argparse.ArgumentParser(description="Train DQN on Pong")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--exploration_fraction", type=float, default=0.1, help="Exploration decay fraction")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--run_name", type=str, default="dqn_pong_default", help="Name of the run")
    args = parser.parse_args()

    # Initialize WandB
    run = wandb.init(
        project="pong_dqn_activity",
        name=args.run_name,
        sync_tensorboard=True,  # Auto-upload tensorboard metrics
        config=vars(args),
        save_code=True,
    )

    # 1. Environment Wrappers (Exercise 1.1)
    # Using the standard deepmind wrappers suggested via stable-baselines3
    def make_env():
        env = gym.make("ALE/Pong-v5")
        env = Monitor(env)
        env = AtariWrapper(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            clip_reward=True,
            action_repeat_probability=0.0
        )
        return env

    env = DummyVecEnv([make_env])
    # Stack 4 frames so the model can perceive velocity/direction of the ball
    env = VecFrameStack(env, n_stack=4)

    # 2. DQN Implementation (Exercise 1.2)
    model = DQN(
        "CnnPolicy", 
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=0.01,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        learning_starts=10000,
        optimize_memory_usage=False,
        verbose=1,
        tensorboard_log=f"runs/{args.run_name}"
    )

    # Train Model (Exercise 1.2 & 1.3)
    print(f"Starting training for {args.timesteps} timesteps...")
    
    # WandbCallback automatically logs all metrics and saves models
    wandb_callback = WandbCallback(
        model_save_path=f"models/{args.run_name}",
        verbose=2,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=wandb_callback
    )

    # Save final model
    print("Saving final model...")
    model.save(f"models/{args.run_name}/dqn_pong_final.zip")
    
    run.finish()

if __name__ == "__main__":
    main()
