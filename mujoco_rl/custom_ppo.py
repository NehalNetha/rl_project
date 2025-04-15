import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import os
import platform # To check OS
import copy # For deep copying state potentially
import mujoco # <--- ADD THIS IMPORT

# --- Imports for Non-Blocking Input ---
import sys
if platform.system() == "Windows":
    import msvcrt
else: # Linux/macOS
    import select
    import tty
    import termios
# --- End Non-Blocking Input Imports ---

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim)) # Learnable log std deviation

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        action_mean = self.fc_mean(x)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        dist = Normal(action_mean, action_std)
        return dist

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        value = self.fc_value(x)
        return value

# --- PPO Agent ---
class CustomPPO:
    def __init__(self, env, hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, update_epochs=10, minibatch_size=64, steps_per_epoch=2048):
        # self.env is the main (non-rendering) environment
        self.env = env
        self.env_id = env.spec.id # Store env ID for creating vis_env later

        # --- Visualization attributes ---
        self.vis_env = None # Separate environment for visualization
        self.rendering_enabled = False # Start with rendering off
        self.render_toggle_key = 'v' # Key to toggle visualization
        # ---------------------------------

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        # Use tensors for actions limits directly if needed elsewhere, numpy here is fine
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.steps_per_epoch = steps_per_epoch

        # Initialize Actor and Critic Networks
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
        self.critic = Critic(self.state_dim, hidden_dim)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout buffer
        self.buffer = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [],
            'log_probs': [], 'values': [], 'dones': [], 'advantages': [], 'returns': []
        }
        self._clear_buffer()
        self.total_steps = 0

    def _clear_buffer(self):
        for key in self.buffer:
            self.buffer[key] = []

    def _store_transition(self, state, action, reward, next_state, log_prob, value, done):
        self.buffer['states'].append(torch.tensor(state, dtype=torch.float32))
        self.buffer['actions'].append(torch.tensor(action, dtype=torch.float32))
        self.buffer['rewards'].append(torch.tensor(reward, dtype=torch.float32))
        self.buffer['next_states'].append(torch.tensor(next_state, dtype=torch.float32))
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(torch.tensor(done, dtype=torch.float32))

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            dist = self.actor(state_tensor)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            value = self.critic(state_tensor)
            # Clip action using numpy arrays from env space
            action_np = action.squeeze(0).numpy()
            action_np = np.clip(action_np, self.action_low, self.action_high)
        # Return numpy action, tensor log_prob, tensor value
        return action_np, log_prob.squeeze(0), value.squeeze(0)

    def predict(self, obs, deterministic=True):
        action, _, _ = self.select_action(obs, deterministic=deterministic)
        return action, None

    def _compute_gae_and_returns(self, last_value, last_done):
        states = torch.stack(self.buffer['states'])
        rewards = torch.stack(self.buffer['rewards'])
        values = torch.stack(self.buffer['values']).squeeze(-1) # Ensure value is squeezed correctly
        dones = torch.stack(self.buffer['dones'])

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        num_steps = len(rewards)

        last_value_tensor = torch.tensor(last_value, dtype=torch.float32)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - torch.tensor(last_done, dtype=torch.float32) # Ensure last_done is tensor
                next_value = last_value_tensor
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value = values[t+1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + values
        self.buffer['advantages'] = list(advantages.detach().clone())
        self.buffer['returns'] = list(returns.detach().clone())

    def _update(self):
        states = torch.stack(self.buffer['states']).detach()
        actions = torch.stack(self.buffer['actions']).detach()
        old_log_probs = torch.stack(self.buffer['log_probs']).detach()
        advantages = torch.stack(self.buffer['advantages']).detach()
        returns = torch.stack(self.buffer['returns']).detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(states)

        for _ in range(self.update_epochs):
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                dist = self.actor(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                current_values = self.critic(mb_states).squeeze(-1) # Ensure value is squeezed correctly
                critic_loss = nn.MSELoss()(current_values, mb_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    def _check_keypress(self):
        if platform.system() == "Windows":
            if msvcrt.kbhit():
                try:
                    return msvcrt.getch().decode('utf-8').lower()
                except UnicodeDecodeError:
                    return None
            return None
        else:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return sys.stdin.read(1).lower()
            return None

    def learn(self, total_timesteps, progress_bar=True):
        """Main training loop."""
        print(f"Starting training with CustomPPO for {total_timesteps} timesteps...")
        print(f"Press '{self.render_toggle_key}' in the terminal to toggle live visualization.") # Always show msg
        start_time = time.time()
        obs, _ = self.env.reset()
        current_epoch_steps = 0

        old_settings_unix = None
        if platform.system() != "Windows":
            try:
                old_settings_unix = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except termios.error as e:
                 print(f"Warning: Could not set terminal to cbreak mode ({e}). Keypress toggle might not work correctly.")
                 old_settings_unix = None

        try:
            while self.total_steps < total_timesteps:

                # --- Check for visualization toggle key ---
                key = self._check_keypress()
                if key == self.render_toggle_key:
                    self.rendering_enabled = not self.rendering_enabled
                    status = "ON" if self.rendering_enabled else "OFF"
                    print(f"\nVisualization toggled {status}")

                    if self.rendering_enabled and self.vis_env is None:
                        # Create visualization environment only when needed
                        print("Creating visualization window...")
                        try:
                            self.vis_env = gym.make(self.env_id, render_mode="human")
                            self.vis_env.reset() # Reset the env before first render/state set
                            print("Visualization window created.")
                        except Exception as e:
                            print(f"\nError creating visualization environment: {e}")
                            self.rendering_enabled = False # Turn off if creation failed
                            if self.vis_env:
                                self.vis_env.close()
                                self.vis_env = None
                    elif not self.rendering_enabled and self.vis_env is not None:
                        # Destroy visualization environment when toggled off
                        print("Closing visualization window...")
                        self.vis_env.close()
                        self.vis_env = None
                        print("Visualization window closed.")
                # ------------------------------------------

                # Collect rollouts (using self.env)
                state_tensor = torch.tensor(obs, dtype=torch.float32)
                action, log_prob, value = self.select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action) # Step the main env
                done = terminated or truncated

                # --- Conditional Rendering using self.vis_env ---
                if self.rendering_enabled and self.vis_env is not None:
                    try:
                        # --- State Synchronization (MuJoCo specific) ---
                        main_data = self.env.unwrapped.data
                        vis_data = self.vis_env.unwrapped.data
                        vis_model = self.vis_env.unwrapped.model # Get the model for mj_forward

                        # Copy the essential simulation state fields
                        vis_data.qpos[:] = main_data.qpos[:]
                        vis_data.qvel[:] = main_data.qvel[:]
                        vis_data.time = main_data.time
                        # Optionally copy control signals if they affect visualization significantly
                        # vis_data.ctrl[:] = main_data.ctrl[:]

                        # --- Recalculate derived quantities ---
                        mujoco.mj_forward(vis_model, vis_data)
                        # ------------------------------------

                        # Render the synchronized and recalculated state
                        self.vis_env.render()

                        # Optional sleep to make it watchable
                        # time.sleep(0.01)
                    except Exception as e:
                        print(f"\nError during rendering or state sync: {e}")
                        self.rendering_enabled = False # Disable rendering on error
                        if self.vis_env:
                            self.vis_env.close()
                            self.vis_env = None
                # ------------------------------------------------

                self._store_transition(obs, action, reward, next_obs, log_prob, value, done)
                obs = next_obs
                current_epoch_steps += 1
                self.total_steps += 1

                epoch_finished = current_epoch_steps == self.steps_per_epoch
                terminal = done

                if epoch_finished or terminal:
                    if terminal:
                        last_value = 0.0
                        last_done = True
                        obs, _ = self.env.reset() # Reset main env
                    else:
                        with torch.no_grad():
                            last_value_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                            last_value = self.critic(last_value_tensor).item()
                        last_done = False

                    self._compute_gae_and_returns(last_value, last_done)
                    self._update()
                    self._clear_buffer()
                    current_epoch_steps = 0

                    # --- Progress Logging ---
                    if progress_bar:
                        elapsed_time = time.time() - start_time
                        steps_per_second = self.total_steps / elapsed_time if elapsed_time > 0 else 0
                        remaining_steps = total_timesteps - self.total_steps
                        eta_seconds = (remaining_steps / steps_per_second) if steps_per_second > 0 else 0
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                        print(f"Steps: {self.total_steps}/{total_timesteps} | SPS: {steps_per_second:.0f} | Vis: {'ON ' if self.rendering_enabled else 'OFF'} | ETA: {eta_str}      ", end='\r')

        finally: # Ensure terminal settings and vis_env are cleaned up
             if old_settings_unix is not None and platform.system() != "Windows":
                 termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings_unix)
             # --- Close vis_env if it's still open on exit ---
             if self.vis_env is not None:
                 print("\nClosing visualization window on exit...")
                 self.vis_env.close()
                 self.vis_env = None
             # -------------------------------------------------

        print("\nTraining finished.")

    def close(self):
        """Closes the visualization environment if it exists."""
        if self.vis_env is not None:
            print("Closing visualization environment.")
            self.vis_env.close()
            self.vis_env = None

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_path = filepath if filepath.endswith('.pth') else f"{filepath}.pth"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, save_path)
        print(f"CustomPPO model saved to {save_path}")

    def load(self, filepath):
        load_path = filepath if filepath.endswith('.pth') else f"{filepath}.pth"
        if not os.path.exists(load_path):
            print(f"Error: Model file not found at {load_path}")
            return False
        checkpoint = torch.load(load_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor.train()
        self.critic.train()
        print(f"CustomPPO model loaded from {load_path}")
        return True 