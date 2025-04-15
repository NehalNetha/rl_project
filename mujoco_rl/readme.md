# Reinforcement Learning for Continuous Control in MuJoCo Environments

**Authors:**
*   Name (PRN)
*   Name (PRN)
*   Name (PRN)

## 1. Description of the Problem

This report details the process of training and evaluating various Reinforcement Learning (RL) agents to solve continuous control tasks within simulated physics environments powered by MuJoCo and accessed via the Gymnasium interface. The primary goal is to enable an agent to learn an optimal policy for interacting with its environment to maximize a cumulative reward signal over time, tackling challenges inherent in high-dimensional state and action spaces.

## 2. Description of the Environment and Agent

### 2.1. Environment (MuJoCo via Gymnasium)

*   **Platform**: MuJoCo (Multi-Joint dynamics with Contact) physics engine.
*   **Interface**: Gymnasium API (`gymnasium.make(env_name)`).
*   **Tasks**: Standard continuous control benchmark tasks (e.g., `Humanoid-v5`, `Walker2d-v5`, `Ant-v5`, etc., specified via the `--env` argument). These environments simulate complex physical systems like legged robots.

### 2.2. Agent Characteristics

*   **States (\(S\))**: The state space is typically high-dimensional and continuous. It usually includes information like joint positions, joint velocities, readings from inertial measurement units (IMUs), contact forces, and potentially target locations or other task-specific variables. The exact composition is defined by the specific Gymnasium environment (e.g., `env.observation_space`).
*   **Actions (\(A\))**: The action space is also continuous, representing forces or torques applied to the agent's actuators (joints). The dimension and bounds of the action space are defined by the environment (e.g., `env.action_space`). Actions are often clipped to stay within valid physical limits.
*   **Objective**: The agent's objective is to learn a policy \( \pi(a|s) \) (mapping states to actions) that maximizes the expected discounted cumulative reward \( G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \), where \( R_t \) is the reward received at timestep \( t \) and \( \gamma \) is the discount factor. The reward function \( R(s, a, s') \) is designed by the environment creators to encourage desired behaviors (e.g., moving forward quickly, maintaining balance, reaching a target) and penalize undesired outcomes (e.g., falling, using excessive energy).
*   **Model**: The RL methods employed (PPO, SAC, TD3, Custom PPO) are **model-free**. They do not explicitly learn a model of the environment's dynamics \( P(s'|s, a) \) or reward function \( R(s, a, s') \). Instead, they learn the policy and/or value functions directly through trial-and-error interaction with the environment, which is provided by the MuJoCo simulation.
*   **Discount Factor (\(\gamma\))**: A discount factor (e.g., `gamma=0.99` as used in the configurations) is applied to future rewards. This hyperparameter balances the importance of immediate versus future rewards. A value close to 1 encourages far-sighted behavior, while a value closer to 0 prioritizes immediate gains.

### 2.3. Markov Decision Process (MDP) Formulation

The interaction between the agent and the MuJoCo environment can be formulated as an MDP, defined by the tuple \( (S, A, P, R, \gamma) \):

*   \( S \): The continuous state space (joint angles, velocities, etc.).
*   \( A \): The continuous action space (joint torques/forces).
*   \( P(s'|s, a) \): The state transition probability function, implicitly defined by the MuJoCo physics engine. Given the current state \( s \) and action \( a \), it determines the probability distribution over the next state \( s' \). Due to the deterministic nature of the physics simulation (given a fixed seed), this is often treated as a deterministic transition \( s' = f(s, a) \).
*   \( R(s, a, s') \): The reward function provided by the Gymnasium environment, mapping a transition to a scalar reward signal.
*   \( \gamma \): The discount factor (e.g., 0.99).

The goal is to find the optimal policy \( \pi^* \) that maximizes the expected return from any starting state.

### 2.4. Visualization

The scripts include functionality to visualize the agent's behavior using `render_mode="human"`:
1.  **Random Actions**: Before training, the environment is rendered with random actions to show the baseline behavior.
2.  **Trained Agent**: After training (or optionally during training for `CustomPPO`), the environment is rendered with the learned policy taking deterministic actions to qualitatively assess performance. The `CustomPPO` implementation includes a mechanism to toggle live rendering during training by pressing the 'v' key, synchronizing the state from the non-rendered training environment to a separate visualization environment.

*(Self-note: Actual screenshots would be inserted here in a full report showing the MuJoCo simulation window for a specific environment before and after training.)*

## 3. Method Description

Two primary approaches were used for training agents: leveraging the Stable Baselines3 library and implementing a custom PPO algorithm.

### 3.1. Stable Baselines3 (SB3) Algorithms

The `train_mujoco_agent.py` script utilizes the robust implementations provided by the Stable Baselines3 library. The following algorithms were configured:

*   **PPO (Proximal Policy Optimization)**: An on-policy, actor-critic algorithm known for its stability and performance across many benchmark tasks. It uses a clipped surrogate objective function and Generalized Advantage Estimation (GAE) to constrain policy updates and reduce variance. It collects rollouts of experience and updates the policy and value function using this data.
*   **SAC (Soft Actor-Critic)**: An off-policy, actor-critic algorithm based on the maximum entropy RL framework. It aims to maximize both the expected reward and the policy's entropy, encouraging exploration. It uses a replay buffer to store and sample past experiences, making it generally more sample-efficient than on-policy methods like PPO.
*   **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**: An off-policy, actor-critic algorithm designed for continuous action spaces. It builds upon DDPG by introducing clipped double Q-learning (using two critic networks and taking the minimum Q-value), delayed policy updates, and target policy smoothing (adding noise to target actions) to improve stability and performance. It also utilizes a replay buffer.

These algorithms were configured using default or common hyperparameters specified within the script, leveraging the `MlpPolicy` (Multi-Layer Perceptron) for both actor and critic networks.

### 3.2. Custom PPO Implementation (`custom_ppo.py`)

A Proximal Policy Optimization (PPO) algorithm was implemented from scratch in `custom_ppo.py` and trained using `train_custom_ppo.py`. Key features include:

*   **Actor-Critic Architecture**: Separate neural networks for the actor (policy) and the critic (value function). Both are MLPs with configurable hidden dimensions (`hidden_dim`).
    *   **Actor**: Outputs the mean of a Gaussian distribution for the continuous actions. A learnable parameter (`log_std`) defines the standard deviation, allowing the agent to control its exploration level.
    *   **Critic**: Outputs a single value representing the expected return from a given state.
*   **Rollout Buffer**: Collects trajectories of `(state, action, reward, next_state, done, log_prob, value)` tuples over a fixed number of steps (`steps_per_epoch`).
*   **Generalized Advantage Estimation (GAE)**: Calculates advantages \( \hat{A}_t \) using the collected rewards and value estimates, incorporating the \( \gamma \) and \( \lambda \) (`gae_lambda`) parameters to balance bias and variance.
*   **PPO Objective Function**: Updates the actor network by maximizing the PPO-clip surrogate objective:
    \[ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) \right] \]
    where \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \) is the probability ratio and \( \epsilon \) is the clipping hyperparameter (`clip_epsilon`).
*   **Value Function Loss**: Updates the critic network by minimizing the Mean Squared Error (MSE) between its predictions and the calculated returns (target values) derived from GAE.
*   **Optimization**: Uses Adam optimizer for both actor and critic networks with separate learning rates (`lr_actor`, `lr_critic`). Updates are performed over multiple epochs (`update_epochs`) using minibatches (`minibatch_size`) sampled from the collected rollout data.
*   **Live Visualization Toggle**: Incorporates a non-blocking keypress check (`_check_keypress`) allowing the user to toggle (`'v'`) a separate visualization window (`vis_env`) during the training loop. When enabled, the state from the primary (non-rendered) training environment is copied to the visualization environment at each step for rendering. This involves synchronizing MuJoCo's `qpos`, `qvel`, and `time` fields and recalculating derived quantities using `mujoco.mj_forward`.

## 4. Results

### 4.1. Training Process

*   Agents were trained using the respective scripts (`train_mujoco_agent.py` for SB3, `train_custom_ppo.py` for Custom PPO).
*   Key parameters like the algorithm (`--algo`), environment (`--env`), total timesteps (`--timesteps`), and random seed (`--seed`) were configurable via command-line arguments.
*   SB3 models were saved as `.zip` files, while the custom PPO model was saved using `torch.save` into a `.pth` file containing state dictionaries for actors, critics, and optimizers.
*   Training progress was monitored via console output, showing steps completed, steps per second (SPS), and ETA. The custom PPO implementation also showed the status of the live visualization toggle.
*   SB3 training logs could be optionally saved to TensorBoard (`tensorboard_log` parameter), allowing for visualization of learning curves (e.g., episode rewards, loss functions). (Note: TensorBoard logging was not explicitly implemented for the `CustomPPO` class in the provided code).

### 4.2. Evaluation

*   After training, agents were evaluated quantitatively in a separate, non-rendered evaluation environment (`eval_env`).
*   For SB3 agents, `stable_baselines3.common.evaluation.evaluate_policy` was used.
*   For the `CustomPPO` agent, a dedicated `evaluate_custom_agent` function was implemented.
*   Both evaluation methods run the agent's learned policy (in deterministic mode) for a fixed number of episodes (`n_eval_episodes=10`) and calculate the mean and standard deviation of the total reward achieved per episode.
*   These metrics (Mean Reward ± Std Reward) provide a quantitative measure of the agent's performance and consistency on the task.

### 4.3. Qualitative Assessment

*   Visualizations before training (random agent) and after training (learned policy) provided a qualitative understanding of the agent's learned behavior and proficiency in solving the task (e.g., observing if the Humanoid learns to walk stably).

## 5. Graphs and Further Analysis

*   **Quantitative Results**: The primary numerical results are the Mean ± Std Reward values printed at the end of each training script run. Comparing these values across different algorithms (PPO, SAC, TD3, CustomPPO) on the same environment and timestep budget provides insight into their relative performance and sample efficiency.
*   **TensorBoard Logs (SB3)**: If TensorBoard logging was enabled for SB3 runs, graphs visualizing episode reward, value loss, policy loss, entropy, etc., over training steps would be available. These graphs are crucial for diagnosing training stability and convergence.
*   *(Self-note: In a full report, tables comparing Mean/Std rewards across algorithms/environments and example TensorBoard graphs would be included here.)*

This study demonstrates the application of standard and custom RL algorithms to challenging continuous control problems, highlighting the training procedures, evaluation methodologies, and visualization techniques used to develop and assess agent performance in MuJoCo environments.
