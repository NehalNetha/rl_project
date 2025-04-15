import gymnasium as gym
import numpy as np # For evaluation stats
import torch # For evaluation device check
import os
import time
import argparse
from custom_ppo import CustomPPO # Import our custom agent

# --- Helper Function for Evaluation ---
def evaluate_custom_agent(agent, env, n_eval_episodes=10, deterministic=True):
    """
    Evaluates the custom PPO agent.
    :param agent: The custom PPO agent to evaluate.
    :param env: The environment to evaluate the agent on.
    :param n_eval_episodes: Number of episodes to use for evaluation.
    :param deterministic: Whether the agent should use deterministic actions.
    :return: (mean_reward, std_reward)
    """
    episode_rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train Custom PPO agent on MuJoCo environments")
# Keep similar args for consistency
parser.add_argument("--env", type=str, default="Humanoid-v5", help="Environment ID")
parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
parser.add_argument("--seed", type=int, default=None, help="Random seed (Note: Pytorch seeding might need more setup for full reproducibility)")
# Add CustomPPO specific hyperparameters as args if desired
parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda factor")
parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip range epsilon")
parser.add_argument("--update_epochs", type=int, default=10, help="Number of PPO update epochs per rollout")
parser.add_argument("--minibatch_size", type=int, default=64, help="PPO minibatch size")
parser.add_argument("--steps_per_epoch", type=int, default=2048, help="Steps collected per PPO update cycle")
parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension for actor/critic")

args = parser.parse_args()
# --- End Argument Parsing ---


# --- Configuration ---
env_name = args.env
total_timesteps = args.timesteps
seed = args.seed
# Note: Proper seeding involves torch, numpy, and env seeding
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # If using CUDA

log_dir = "/Users/nehal/SixthSemester/ganLab/rl_project/logs/custom_ppo/" # Separate log dir
model_dir = "/Users/nehal/SixthSemester/ganLab/rl_project/models/custom_ppo/" # Separate model dir
os.makedirs(log_dir, exist_ok=True) # Logs aren't implemented in CustomPPO yet
os.makedirs(model_dir, exist_ok=True)
# --- End Configuration ---

# --- Environment Setup ---
# Create environment for training (NO rendering needed here)
print(f"Creating training environment '{env_name}' (no rendering)...")
env = gym.make(env_name)

# Create separate environment for evaluation (render_mode doesn't matter here)
eval_env = gym.make(env_name)

# Optional: Seed environments
if seed is not None:
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)
# --- End Environment Setup ---

# --- Visualize Random Actions (Optional) ---
try:
    vis_env_random = gym.make(env_name, render_mode="human")
    print("Visualizing environment with random actions before training...")
    obs, _ = vis_env_random.reset()
    for _ in range(200):
        action = vis_env_random.action_space.sample()
        obs, reward, terminated, truncated, info = vis_env_random.step(action)
        time.sleep(0.01)
        if terminated or truncated:
            obs, _ = vis_env_random.reset()
    vis_env_random.close()
except Exception as e:
    print(f"Could not create visualization environment: {e}")

user_input = input(f"About to train CustomPPO on {env_name} for {total_timesteps} steps. Press Enter to start, or type 'q' to quit: ")
if user_input.lower() == 'q':
    env.close()
    eval_env.close()
    exit()
# --- End Visualize Random Actions ---


# --- Agent Initialization ---
print("Initializing CustomPPO agent...")
agent = CustomPPO(
    env=env, # Pass the training env
    hidden_dim=args.hidden_dim,
    lr_actor=args.lr_actor,
    lr_critic=args.lr_critic,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_epsilon=args.clip_epsilon,
    update_epochs=args.update_epochs,
    minibatch_size=args.minibatch_size,
    steps_per_epoch=args.steps_per_epoch
)
print("CustomPPO agent initialized.")
# --- End Agent Initialization ---


# --- Training ---
agent.learn(total_timesteps=total_timesteps, progress_bar=True)
# --- End Training ---


# --- Saving Model ---
model_filename = f"custom_ppo_{env_name}_{total_timesteps}.pth" # Use .pth extension
model_path = os.path.join(model_dir, model_filename)
agent.save(model_path)
# --- End Saving Model ---


# --- Evaluation ---
print("Evaluating trained agent...")
# Use our custom evaluation function
mean_reward, std_reward = evaluate_custom_agent(agent, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Algorithm: CustomPPO")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
# --- End Evaluation ---


# --- Visualization ---
print("Visualizing trained agent performance...")
try:
    vis_env = gym.make(env_name, render_mode="human")
    obs, _ = vis_env.reset()
    for _ in range(1000): # Visualize for 1000 steps
        action, _states = agent.predict(obs, deterministic=True) # Use agent.predict
        obs, reward, terminated, truncated, info = vis_env.step(action)
        # time.sleep(0.01) # Optional slow down
        if terminated or truncated:
            obs, _ = vis_env.reset()
    vis_env.close()
except Exception as e:
    print(f"Could not create visualization environment: {e}")
# --- End Visualization ---

# Clean up environments
env.close()
eval_env.close()

print("Script finished.") 