import argparse
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import numpy as np
import imageio
import os

def render_episode(env_name, model, max_steps=5000):
    """
    Runs a single episode and returns the cumulative reward and the frames.
    """
    # Create the identical environment to the training one but render rgb_array
    env = gym.make(env_name, render_mode="rgb_array")
    env = AtariWrapper(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False, # Standard eval doesn't terminate on life loss like training does
        clip_reward=False, # We want actual real-world episodic score 
        action_repeat_probability=0.0
    )
    
    # We must use dummy vec for frame stacking
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    obs = env.reset()
    
    frames = []
    total_reward = 0.0
    done = False
    step = 0
    
    # Extract inner unwrapped environment to render frames properly
    # Stable baselines wraps it tightly, so we call env.envs[0].render()
    
    while not done and step < max_steps:
        # Use underlying environment purely for RGB frame capturing
        frames.append(env.envs[0].unwrapped.render())
        
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        total_reward += rewards[0]
        step += 1
        done = dones[0]

    env.close()
    return total_reward, frames

def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN Model on Pong")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .zip model file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = DQN.load(args.model_path)

    print(f"Evaluating for {args.episodes} episodes...")
    rewards = []
    
    best_reward = -float('inf')
    worst_reward = float('inf')
    best_frames = []
    worst_frames = []

    for i in range(args.episodes):
        reward, frames = render_episode("ALE/Pong-v5", model)
        rewards.append(reward)
        
        print(f"Episode {i+1}/{args.episodes} - Reward: {reward}")

        # Track the best episode
        if reward > best_reward:
            best_reward = reward
            best_frames = frames
            
        # Track the worst episode
        if reward < worst_reward:
            worst_reward = reward
            worst_frames = frames

    # Compute Statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print("\n" + "="*40)
    print(f"Evaluation Results over {args.episodes} episodes:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Worst Reward: {worst_reward:.2f}")
    print("="*40 + "\n")

    # Export GIFs (Exercise 2.3)
    os.makedirs("gifs", exist_ok=True)
    
    print(f"Saving best episode GIF (Reward: {best_reward})...")
    imageio.mimwrite("gifs/best_episode.gif", best_frames, duration=1000/30) # ~30 FPS
    
    print(f"Saving worst episode GIF (Reward: {worst_reward})...")
    imageio.mimwrite("gifs/worst_episode.gif", worst_frames, duration=1000/30)
    
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()
