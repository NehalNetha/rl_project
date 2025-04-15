# <cell>
# --- Imports ---
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
# Note: make_vec_env is not used in the original script, so removed for simplicity
import os
import time
# Removed argparse as we'll define args directly

# <cell>
# --- Configuration ---
# --- Set your parameters here ---
algorithm_str = "PPO" # Choose "PPO", "SAC", or "TD3"
env_name = "Humanoid-v5" # Specify the MuJoCo Environment ID
total_timesteps = 1_000_000 # Total training timesteps
seed = 42 # Set to None for no specific seed, or an integer for reproducibility
# --- End Parameter Setting ---

# --- Paths (Colab specific) ---
# Using relative paths or /content/ is common in Colab
log_dir = "logs/"
model_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# --- End Configuration ---

# <cell>
# --- Environment Setup ---
print(f"Setting up environment: {env_name}")
env = gym.make(env_name)
# For evaluation, use a separate instance
# Render mode is usually not needed for eval, omit for efficiency
eval_env = gym.make(env_name)

if seed is not None:
    print(f"Setting random seed: {seed}")
    # Seed the environment reset for reproducibility
    # Note: Stable Baselines3 models handle their own seeding internally when seed is passed
    env.reset(seed=seed)
    eval_env.reset(seed=seed+1) # Use a different seed for eval env
# --- End Environment Setup ---

# <cell>
# --- (Optional) Visualize Random Actions ---
# Note: Direct rendering with 'human' mode doesn't work in standard Colab.
# You might need alternative methods like saving videos or using specific Colab libraries if visualization is crucial.
# This section is commented out by default.

# print("Visualizing environment with random actions (if possible)...")
# try:
#     vis_env_random = gym.make(env_name, render_mode="human") # This line will likely fail in Colab
#     obs, _ = vis_env_random.reset()
#     for _ in range(200):
#         action = vis_env_random.action_space.sample()
#         obs, reward, terminated, truncated, info = vis_env_random.step(action)
#         time.sleep(0.01) # May not be effective without visual rendering
#         if terminated or truncated:
#             obs, _ = vis_env_random.reset()
#     vis_env_random.close()
# except Exception as e:
#     print(f"Could not create visualization environment: {e}")
#     print("Skipping random action visualization.")

# --- End Visualize Random Actions ---

# <cell>
# --- Agent Initialization ---
print(f"Initializing {algorithm_str} agent...")

# Map string to class
ALGOS = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
if algorithm_str not in ALGOS:
     raise ValueError(f"Algorithm {algorithm_str} not supported. Choose from {list(ALGOS.keys())}")
algorithm_class = ALGOS[algorithm_str]

common_params = {
    "policy": "MlpPolicy",
    "env": env,
    "verbose": 1,
    "tensorboard_log": log_dir,
    "seed": seed,
    # Add other common parameters like learning rate if desired,
    # but defaults might differ significantly between algos.
    # We'll start with defaults and algorithm-specific adjustments below.
}

model_params = common_params.copy() # Start with common params

if algorithm_str == "PPO":
    model_params.update({
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
    })
elif algorithm_str == "SAC":
    model_params.update({
         # "learning_rate": 3e-4, # Often default works well
         "buffer_size": total_timesteps, # Store experiences up to total steps
         "learning_starts": 10000, # Start learning after 10k steps
         "batch_size": 256, # Common default
         "gamma": 0.99,
         # "train_freq": (1, "step"), # Default
         # "gradient_steps": -1, # Default
    })
elif algorithm_str == "TD3":
     model_params.update({
         # "learning_rate": 1e-3, # Often default works well
         "buffer_size": total_timesteps,
         "learning_starts": 10000,
         "batch_size": 100, # Common default
         "gamma": 0.99,
         # "train_freq": (1, "episode"), # Default for TD3 is episode
         # "gradient_steps": -1, # Default
     })

model = algorithm_class(**model_params)

print(f"Initialized {algorithm_str} model.")
# --- End Agent Initialization ---

# <cell>
# --- Training ---
# The input prompt is removed for non-interactive Colab execution
print(f"Starting training for {total_timesteps} timesteps...")
# Define TensorBoard log name based on algo and env
tb_log_name = f"{algorithm_str}_{env_name}"
model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=tb_log_name)
print("Training finished.")
# --- End Training ---

# <cell>
# --- Saving Model ---
model_filename = f"{algorithm_str.lower()}_{env_name}_{total_timesteps}.zip"
model_path = os.path.join(model_dir, model_filename)
model.save(model_path)
print(f"Model saved to {model_path}")
# --- End Saving Model ---

# <cell>
# --- Evaluation ---
print("Evaluating trained agent...")
# Use the separate evaluation environment
# deterministic=True makes the agent pick the best action, not sample
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"--- Evaluation Results ---")
print(f"Algorithm: {algorithm_str}")
print(f"Environment: {env_name}")
print(f"Mean reward over 10 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"-------------------------")
# --- End Evaluation ---

# <cell>
# --- (Optional) Visualization of Trained Agent ---
# As with random visualization, 'human' mode is problematic in Colab.
# This section is commented out. Consider alternatives like saving videos.

# print("Visualizing trained agent performance (if possible)...")
# try:
#     vis_env = gym.make(env_name, render_mode="human") # This line will likely fail in Colab
#     obs, _ = vis_env.reset()
#     for _ in range(1000): # Visualize for 1000 steps
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = vis_env.step(action)
#         if terminated or truncated:
#             obs, _ = vis_env.reset()
#     vis_env.close()
# except Exception as e:
#     print(f"Could not create visualization environment: {e}")
#     print("Skipping trained agent visualization.")

# --- End Visualization ---

# <cell>
# --- Cleanup ---
print("Closing environments...")
env.close()
eval_env.close()
print("Script finished.")
# --- End Cleanup ---
