import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import os
import time
import argparse

# --- Argument Parsing ---python c:\Users\cl502_21\Downloads\rl_new\rl_project\mujoco_rl\train_mujoco_agent.py --algo PPO --env Humanoid-v5 --timesteps 500000 --checkpoint c:\Users\cl502_21\Downloads\rl_new\rl_project\mujoco_rl\models\ppo_Humanoid-v5_1000000.zip
parser = argparse.ArgumentParser(description="Train RL agents on MuJoCo environments")
parser.add_argument("--algo", type=str.upper, default="PPO", choices=["PPO", "SAC", "TD3"],
                    help="RL Algorithm to use (PPO, SAC, TD3)")
parser.add_argument("--env", type=str, default="Humanoid-v5", help="Environment ID")
parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps for this session")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint to load and continue training") # Add this line
args = parser.parse_args()
# --- End Argument Parsing ---


# --- Configuration ---
env_name = args.env
algorithm = args.algo
total_timesteps = args.timesteps
seed = args.seed
checkpoint_path = args.checkpoint # Store the checkpoint path
log_dir = os.path.join(os.path.dirname(__file__), "logs")
model_dir = os.path.join(os.path.dirname(__file__), "models")
# Path to the existing model # Remove this line, it's replaced by the argument
# existing_model_path = os.path.join(model_dir, "ppo_Humanoid-v5_5000000.zip") # Remove this line
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# --- End Configuration ---

env = gym.make(env_name)
eval_env = gym.make(env_name)

if seed is not None:
    env.reset(seed=seed)
    eval_env.reset(seed=seed+1) # Use a different seed for eval env
# --- End Environment Setup ---

# Replace random visualization with checkpoint model visualization
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading model from {checkpoint_path} for visualization...")
    vis_model = eval(algorithm).load(checkpoint_path, device="cuda")
    
    vis_env = gym.make(env_name, render_mode="human")
    print("Visualizing agent with loaded checkpoint before additional training...")
    obs, _ = vis_env.reset()
    for _ in range(1000):  # Show 200 steps of the pre-trained agent
        action, _states = vis_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = vis_env.step(action)
        time.sleep(0.01)
        if terminated or truncated:
            obs, _ = vis_env.reset()
    vis_env.close()
else:
    # If no checkpoint, still show random actions
    vis_env_random = gym.make(env_name, render_mode="human")
    print("No checkpoint found. Visualizing environment with random actions before training...")
    obs, _ = vis_env_random.reset()
    for _ in range(200):
        action = vis_env_random.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = vis_env_random.step(action)
        time.sleep(0.01)
        if terminated or truncated:
            obs, _ = vis_env_random.reset()
    vis_env_random.close()

user_input = input(f"About to train {algorithm} on {env_name} for {total_timesteps} steps. Press Enter to start, or type 'q' to quit: ")
if user_input.lower() == 'q':
    env.close()
    eval_env.close()
    exit()
# --- End Visualization Before Training ---


# --- Agent Initialization ---
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading existing model from {checkpoint_path}")
    model = eval(algorithm).load(checkpoint_path, env=env, device="cuda")
    # Optionally reset the learning rate for continued training
    model.learning_rate = 3e-4
else:
    print("No existing model found. Starting fresh training.")
    common_params = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "seed": seed,
 "device": device,
}
common_params = {
    "policy": "MlpPolicy",
    "env": env,
    "verbose": 1,
    "tensorboard_log": log_dir,
    "seed": seed,
    "device": "cuda",  # Add this line to enable CUDA
    # Add other common parameters like learning rate if desired,
    # but defaults might differ significantly between algos.
    # We'll start with defaults.
}

if algorithm == "PPO":
    # PPO specific params with enhanced network and learning settings
    model = PPO(
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 256],  # Deeper policy network
                vf=[256, 256, 256]   # Deeper value network
            )
        ),
        learning_rate=1e-4,  # Smaller learning rate for fine-tuning
        n_steps=4096,        # Increased steps per update
        batch_size=128,      # Larger batch size
        n_epochs=15,         # More epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,      # Slightly increased exploration
        **common_params
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
if checkpoint_path and os.path.exists(checkpoint_path):
    # Extract the original timesteps from the checkpoint filename
    checkpoint_filename = os.path.basename(checkpoint_path)
    original_timesteps = 0

    # Try to extract the original timesteps from the filename
    try:
        # Assuming format like "ppo_Humanoid-v5_1000000.zip"
        parts = checkpoint_filename.split('_')
        if len(parts) >= 3:
            # Handle potential "continued" keyword in the filename if loading a continued model
            if parts[-2].startswith("continued"):
                 # e.g., ppo_Humanoid-v5_continued_1000000+500000.zip
                 time_part = parts[-1].split('.')[0] # "1000000+500000"
                 original_timesteps = sum(map(int, time_part.split('+')))
            else:
                 # e.g., ppo_Humanoid-v5_1000000.zip
                 original_timesteps = int(parts[-1].split('.')[0])
    except ValueError:
        print(f"Warning: Could not parse original timesteps from checkpoint filename: {checkpoint_filename}. Using 0.")
        original_timesteps = 0 # Default if parsing fails

    # Create a filename that shows the cumulative timesteps
    cumulative_timesteps = original_timesteps + total_timesteps
    model_filename = f"{algorithm.lower()}_{env_name}_continued_{cumulative_timesteps}.zip"
    print(f"Saving continued model with cumulative timesteps: {cumulative_timesteps}")
else:
    # Standard naming for fresh training
    model_filename = f"{algorithm.lower()}_{env_name}_{total_timesteps}.zip"
    print(f"Saving new model with total timesteps: {total_timesteps}")


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