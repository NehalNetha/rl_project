import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import os
import time
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train RL agents on MuJoCo environments")
parser.add_argument("--algo", type=str.upper, default="PPO", choices=["PPO", "SAC", "TD3"],
                    help="RL Algorithm to use (PPO, SAC, TD3)")
parser.add_argument("--env", type=str, default="Humanoid-v5", help="Environment ID")
parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
args = parser.parse_args()
# --- End Argument Parsing ---


# --- Configuration ---
env_name = args.env
algorithm = args.algo
total_timesteps = args.timesteps
seed = args.seed
log_dir = "/Users/nehal/SixthSemester/ganLab/rl_project/logs/"
model_dir = "/Users/nehal/SixthSemester/ganLab/rl_project/models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# --- End Configuration ---

env = gym.make(env_name)
eval_env = gym.make(env_name)

if seed is not None:
    env.reset(seed=seed)
    eval_env.reset(seed=seed+1) # Use a different seed for eval env
# --- End Environment Setup ---


vis_env_random = gym.make(env_name, render_mode="human")
print("Visualizing environment with random actions before training...")
obs, _ = vis_env_random.reset()
for _ in range(200):
    action = vis_env_random.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = vis_env_random.step(action)
    time.sleep(0.01)
    if terminated or truncated:
        obs, _ = vis_env_random.reset()
vis_env_random.close() # Close the random visualization window

user_input = input(f"About to train {algorithm} on {env_name} for {total_timesteps} steps. Press Enter to start, or type 'q' to quit: ")
if user_input.lower() == 'q':
    env.close()
    eval_env.close()
    exit()
# --- End Visualize Random Actions ---


# --- Agent Initialization ---
common_params = {
    "policy": "MlpPolicy",
    "env": env,
    "verbose": 1,
    "tensorboard_log": log_dir,
    "seed": seed,
    # Add other common parameters like learning rate if desired,
    # but defaults might differ significantly between algos.
    # We'll start with defaults.
}

if algorithm == "PPO":
    # PPO specific params (example, using previous values)
    model = PPO(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        **common_params # Add common params
    )
elif algorithm == "SAC":
    # SAC uses a replay buffer, defaults are often good starting points
    model = SAC(
         # learning_rate=3e-4, # Often default works well
         buffer_size=total_timesteps, # Store most experiences
         learning_starts=10000, # Start learning after 10k steps
         batch_size=256, # Common default
         gamma=0.99,
         # train_freq=(1, "step"), # Default
         # gradient_steps=-1, # Default
         **common_params
    )
elif algorithm == "TD3":
    # TD3 also uses a replay buffer
     model = TD3(
         # learning_rate=1e-3, # Often default works well
         buffer_size=total_timesteps,
         learning_starts=10000,
         batch_size=100, # Common default
         gamma=0.99,
         # train_freq=(1, "episode"), # Default for TD3 is episode
         # gradient_steps=-1, # Default
         **common_params
     )
else:
    raise ValueError(f"Algorithm {algorithm} not supported.")

print(f"Initialized {algorithm} model.")
# --- End Agent Initialization ---


# --- Training ---
print(f"Starting training for {total_timesteps} timesteps...")
model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=f"{algorithm}_{env_name}")
print("Training finished.")
# --- End Training ---


# --- Saving Model ---
model_filename = f"{algorithm.lower()}_{env_name}_{total_timesteps}.zip"
model_path = os.path.join(model_dir, model_filename)
model.save(model_path)
print(f"Model saved to {model_path}")
# --- End Saving Model ---


# --- Evaluation ---
print("Evaluating trained agent...")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Algorithm: {algorithm}")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
# --- End Evaluation ---


# --- Visualization ---
print("Visualizing trained agent performance...")
vis_env = gym.make(env_name, render_mode="human")
obs, _ = vis_env.reset()
for _ in range(1000): # Visualize for 1000 steps
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vis_env.step(action)
    if terminated or truncated:
        obs, _ = vis_env.reset()
vis_env.close()
# --- End Visualization ---

# Clean up training environment
env.close()
eval_env.close()

print("Script finished.")