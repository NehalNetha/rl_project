import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

# --- Parameters ---
# Make sure this matches the environment the model was trained on
env_name = "Hopper-v5"
# Path to the saved model file
model_path = os.path.join(os.path.dirname(__file__), "models", "ppo_Hopper-v5_100000.zip")
# --- End Parameters C:\Users\cl502_21\Downloads\rl_new\rl_project\mujoco_rl\models\ppo_Hopper-v5_100000.zip

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

# Create the environment with rendering enabled
# Make sure MuJoCo rendering prerequisites are met
try:
    vis_env = gym.make(env_name, render_mode="human")
except Exception as e:
    print(f"Error creating environment: {e}")
    print("Ensure you have a display environment set up (e.g., an X server on Linux or appropriate setup on macOS/Windows).")
    exit()


# Load the trained agent
# Note: You might need to pass the env or device explicitly if loading fails
try:
    model = PPO.load(model_path, env=vis_env)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error loading the model: {e}")
    vis_env.close()
    exit()

# Visualize the trained agent
print("Visualizing trained agent performance...")
obs, _ = vis_env.reset()
episodes = 0
max_episodes = 50 # Visualize for 5 episodes

while episodes < max_episodes:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = vis_env.step(action)

    # Optional: add a small delay to make visualization slower
    # time.sleep(0.01)

    if terminated or truncated:
        print(f"Episode {episodes + 1} finished.")
        episodes += 1
        if episodes < max_episodes:
            obs, _ = vis_env.reset()
        else:
            print("Max episodes reached.")

print("Visualization finished.")
vis_env.close()