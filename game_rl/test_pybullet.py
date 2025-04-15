import pybullet as p
import time
import pybullet_data

print("Starting PyBullet test...")

# Connect to the physics server
# Use p.GUI for a graphical interface, or p.DIRECT for non-graphical
physicsClient = p.connect(p.GUI) 
print(f"Connected to physics server with ID: {physicsClient}")

# Add the PyBullet data path to the search path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity
p.setGravity(0, 0, -9.81)

# Load a ground plane
print("Loading ground plane...")
planeId = p.loadURDF("plane.urdf")
print(f"Loaded plane with ID: {planeId}")

# Load a simple object (optional, e.g., a cube)
print("Loading a cube...")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
cubeId = p.loadURDF("cube_small.urdf", startPos, startOrientation)
print(f"Loaded cube with ID: {cubeId}")

# Run the simulation for a few seconds
print("Running simulation for 5 seconds...")
for i in range(5000): # PyBullet default simulation step is 240 Hz
    p.stepSimulation()
    time.sleep(1./240.) # Sleep to match the simulation frequency

    # Optional: Get and print cube position
    if i % 240 == 0: # Print once per second
        cubePos, cubeOrn = p.getBasePositionAndOrientation(cubeId)
        print(f"Step {i}, Cube Position: {cubePos}")


print("Simulation finished.")

# Disconnect from the physics server
p.disconnect()

print("PyBullet test completed successfully.")
