import pybullet as p
import time
import pybullet_data
import numpy as np # Import numpy for calculations

print("Starting PyBullet Quadcopter Simulation...")

# Connect to the physics server
physicsClient = p.connect(p.GUI)
print(f"Connected to physics server with ID: {physicsClient}")

# Add the PyBullet data path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set simulation parameters
p.setGravity(0, 0, -9.81)
# You might need a smaller time step for stability with drone control
# timeStep = 1./240.
# p.setTimeStep(timeStep)

# Load ground plane
print("Loading ground plane...")
planeId = p.loadURDF("plane.urdf")
print(f"Loaded plane with ID: {planeId}")

# --- Load the Quadcopter ---
print("Loading quadcopter...")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
try:
    # Note: Some versions of pybullet_data might have husky/quadrotor.urdf
    # Check your pybullet_data directory if "quadrotor.urdf" fails
    quadcopterId = p.loadURDF("quadrotor.urdf", startPos, startOrientation)
    print(f"Loaded quadcopter with ID: {quadcopterId}")
except p.error as e:
    print(f"Error loading quadrotor.urdf: {e}")
    print("Please ensure 'quadrotor.urdf' is available in the pybullet_data path or common repositories.")
    print("Search Path:", pybullet_data.getDataPath())
    p.disconnect()
    exit()

# --- Quadcopter Physics and Control Setup ---

# Get mass (needed for hover thrust calculation)
# Assuming mass is primarily in the base link (-1)
quad_mass = p.getDynamicsInfo(quadcopterId, -1)[0]
print(f"Quadcopter mass: {quad_mass:.3f} kg")

# Calculate approximate hover thrust (total force needed to counteract gravity)
hover_thrust = quad_mass * 9.81
print(f"Approximate hover thrust: {hover_thrust:.3f} N")

# Define approximate rotor positions relative to the center of mass (in the quadcopter's frame)
# These values depend on the specific URDF geometry. You might need to adjust them.
# Let's assume a simple square configuration. L is half the distance between opposite rotors.
L = 0.15 # Meters (adjust based on your URDF)
rotor_positions = [
    [ L,  L, 0], # Front-Right (Motor 0 - typically spins CW)
    [-L,  L, 0], # Front-Left  (Motor 1 - typically spins CCW)
    [-L, -L, 0], # Back-Left   (Motor 2 - typically spins CW)
    [ L, -L, 0]  # Back-Right  (Motor 3 - typically spins CCW)
]

# Control Gains (These would eventually be determined by your RL agent or a PID controller)
# For now, we'll set simple desired thrusts
# Base thrust per motor for hovering
base_thrust_per_motor = hover_thrust / 4.0

print("Running simulation with basic hover control...")
# Simulation loop
for i in range(10 * 240): # Run for 10 seconds
    
    # --- RL Agent Action Would Go Here ---
    # In a real RL setup, your agent would output desired actions here.
    # Actions could be:
    # 1. Direct thrust for each motor (like we calculate below)
    # 2. Desired overall thrust, roll, pitch, yaw rates (which a mixer then converts to motor thrusts)
    
    # Example: Simple Hover Control (no adjustments)
    # For now, let's just command each motor to produce its share of the hover thrust.
    # In a real scenario, you'd add adjustments based on sensor readings (position, orientation, velocity)
    # to correct errors and achieve desired movements.
    
    # Desired control inputs (Example - replace with agent's output)
    thrust_cmd = base_thrust_per_motor # Base thrust
    roll_cmd = 0.0   # No roll adjustment
    pitch_cmd = 0.0  # No pitch adjustment
    yaw_cmd = 0.0    # No yaw adjustment

    # --- Mixer ---
    # Convert desired commands (thrust, roll, pitch, yaw) into individual motor thrusts
    # This is a simplified mixer. Signs depend on motor spin direction and configuration.
    # Adjust signs if your drone behaves unexpectedly (e.g., flips over).
    thrusts = [
        thrust_cmd - pitch_cmd + roll_cmd - yaw_cmd,  # Motor 0 (Front-Right)
        thrust_cmd - pitch_cmd - roll_cmd + yaw_cmd,  # Motor 1 (Front-Left)
        thrust_cmd + pitch_cmd - roll_cmd - yaw_cmd,  # Motor 2 (Back-Left)
        thrust_cmd + pitch_cmd + roll_cmd + yaw_cmd   # Motor 3 (Back-Right)
    ]
    
    # Clip thrusts to be non-negative (motors can't pull down)
    thrusts = [max(0, t) for t in thrusts]
    
    # --- Apply Forces ---
    # Apply the calculated thrust for each rotor
    for motor_index in range(4):
        force_vector = [0, 0, thrusts[motor_index]] # Thrust is purely upward in the body frame
        position_vector = rotor_positions[motor_index] # Apply force at the rotor location
        
        # Apply force in the LINK_FRAME (body frame of the quadcopter)
        p.applyExternalForce(
            objectUniqueId=quadcopterId,
            linkIndex=-1, # Apply force to the base link
            forceObj=force_vector,
            posObj=position_vector,
            flags=p.LINK_FRAME
        )

    # --- Step Simulation ---
    p.stepSimulation()
    time.sleep(1./240.)

    # Optional: Get and print quadcopter state
    if i % 60 == 0: # Print 4 times per second
        quadPos, quadOrnQuat = p.getBasePositionAndOrientation(quadcopterId)
        quadOrnEuler = p.getEulerFromQuaternion(quadOrnQuat)
        quadLinVel, quadAngVel = p.getBaseVelocity(quadcopterId)
        
        print(f"Step {i}: Pos {np.round(quadPos, 2)}, Orn {np.round(np.degrees(quadOrnEuler), 1)} deg, "
              f"LinVel {np.round(quadLinVel, 2)}, AngVel {np.round(quadAngVel, 2)}")
              # f"Thrusts {np.round(thrusts, 2)}") # Uncomment to see thrusts


print("Simulation finished.")

# Disconnect from the physics server
p.disconnect()

print("PyBullet Quadcopter simulation completed.")


# --- Notes for RL Integration ---
#
# 1. Environment Wrapper: You'll typically wrap this PyBullet simulation in an OpenAI Gym
#    (or Gymnasium) compatible environment class. This class will handle:
#    - `__init__`: Setting up the PyBullet simulation.
#    - `reset()`: Resetting the simulation to a starting state (e.g., random initial position/orientation)
#                 and returning the initial observation.
#    - `step(action)`:
#        - Takes the agent's `action` (e.g., the 4 motor thrusts, or desired roll/pitch/yaw/thrust).
#        - Applies the action to the quadcopter using `p.applyExternalForce` (as shown above).
#        - Steps the simulation (`p.stepSimulation()`).
#        - Gets the new state (observation).
#        - Calculates the reward based on the new state (e.g., distance to target, stability, energy used).
#        - Determines if the episode is `done` (e.g., crashed, reached target, time limit exceeded).
#        - Returns `(observation, reward, done, info)`.
#    - `render()`: (Optional) For visualizing.
#    - `close()`: Disconnecting from PyBullet.
#
# 2. State (Observation): What the RL agent "sees". This could include:
#    - Quadcopter position (`p.getBasePositionAndOrientation`)
#    - Quadcopter orientation (Quaternion or Euler angles)
#    - Linear velocity (`p.getBaseVelocity`)
#    - Angular velocity (`p.getBaseVelocity`)
#    - Maybe target position, distance to target, etc., depending on the task.
#    Normalize these values before feeding them to the neural network.
#
# 3. Action Space: What the RL agent controls. Common choices:
#    - Direct motor thrusts: A continuous space (e.g., Box(0, max_thrust, shape=(4,))) for each of the 4 motors.
#    - Desired attitude rates/thrust: A continuous space for desired roll rate, pitch rate, yaw rate, and overall thrust. A lower-level controller (like a PID or the mixer shown above) would then convert these into motor thrusts. This can sometimes be easier to learn.
#
# 4. Reward Function: This is critical and task-dependent. Examples:
#    - Goal Reaching: Negative distance to target position. Large reward for reaching the goal.
#    - Hovering: Penalty for deviation from desired altitude/position. Penalty for large velocities/angular rates.
#    - Stability: Penalty for large tilt angles (roll/pitch). Penalty for high angular velocity.
#    - Energy Efficiency: Penalty for high thrust values.
#    - Survival: Small positive reward for each timestep the drone doesn't crash.
#    - Crash Penalty: Large negative reward if the drone hits the ground or goes out of bounds.
#
# 5. RL Algorithm: Use libraries like Stable Baselines3, RLlib, Tianshou, etc., to implement algorithms
#    like PPO, SAC, TD3 which work well with continuous control tasks.
#