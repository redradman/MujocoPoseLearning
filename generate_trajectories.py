from stable_baselines3 import PPO
from custom_env import HumanoidEnv
from pathlib import Path
import xml.etree.ElementTree as ET

def generate_trajectory_xml(model_path, num_steps=1000, step_interval=5):
    """Generate trajectory and save directly as XML with keyframes"""
    # Setup environment
    project_root = Path(__file__).parent
    xml_path = project_root / "XML" / "humanoid.xml"
    
    env = HumanoidEnv({
        "model_path": str(xml_path),
        "render_mode": None,
        "duration": 30.0,
        "reward_config": {"type": "walk"}
    })
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Parse the original XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find or create keyframe section
    keyframe = root.find('keyframe')
    if keyframe is None:
        keyframe = ET.SubElement(root, 'keyframe')
    
    # Generate trajectory
    obs, _ = env.reset()
    
    # Add initial pose as first keyframe
    key = ET.SubElement(keyframe, 'key')
    key.set('name', 'initial_pose')
    key.set('time', "0.000")
    qpos_str = ' '.join([f"{x:.6f}" for x in env.data.qpos])
    qvel_str = ' '.join([f"{x:.6f}" for x in env.data.qvel])
    key.set('qpos', qpos_str)
    key.set('qvel', qvel_str)
    
    # Generate and save trajectory as keyframes
    timestep = env.model.opt.timestep
    
    for step in range(num_steps):
        if step % step_interval == 0:  # Save keyframe every step_interval steps
            # Create keyframe
            key = ET.SubElement(keyframe, 'key')
            time = step * timestep
            key.set('time', f"{time:.3f}")
            
            # Save current state
            qpos_str = ' '.join([f"{x:.6f}" for x in env.data.qpos])
            qvel_str = ' '.join([f"{x:.6f}" for x in env.data.qvel])
            key.set('qpos', qpos_str)
            key.set('qvel', qvel_str)
        
        # Get action from model and step environment
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            break
    
    # Create trajectories directory if it doesn't exist
    trajectories_dir = project_root / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)
    
    # Save the modified XML in trajectories directory
    output_path = trajectories_dir / "humanoid_trajectory.xml"
    tree.write(str(output_path), encoding='utf-8', xml_declaration=True)
    print(f"Trajectory XML saved to {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    model_path = "sb3_results/final_model.zip"
    xml_path = generate_trajectory_xml(model_path)
    print("\nTo view the trajectory:")
    print(f"1. Open MuJoCo GUI: mujoco {xml_path}")
    print("2. Press 'space' to start/stop playback")
    print("3. Use '[' and ']' to step through keyframes")
    print("4. Use ',' and '.' to adjust playback speed") 