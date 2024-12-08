import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import pickle
import time

def load_trajectory(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def visualize_trajectory(trajectory):
    # Load model
    xml_path = Path("XML/humanoid.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Create viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset to initial state
        mujoco.mj_resetData(model, data)
        
        # Playback trajectory
        for action in trajectory['actions']:
            # Apply action
            data.ctrl[:] = action
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Control playback speed
            time.sleep(model.opt.timestep)
            
            if viewer.is_stopped():
                break

if __name__ == "__main__":
    # Create trajectories directory if it doesn't exist
    Path("trajectories").mkdir(exist_ok=True)
    
    # Load and visualize trajectory
    trajectory = load_trajectory("trajectories/trajectory.pkl")
    visualize_trajectory(trajectory) 