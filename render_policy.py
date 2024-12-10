from stable_baselines3 import PPO
from custom_env import HumanoidEnv
from pathlib import Path
from train_sb3 import FRAMERATE, DURATION, FRAME_SKIP, REWARD_FUNCTION

def render_trained_model(model_path, num_steps=1000):
    # Load trained model
    model = PPO.load(model_path)
    
    # Create the environment with rendering enabled
    project_root = Path(__file__).parent
    xml_path = project_root / "XML" / "humanoid.xml"
    
    env = HumanoidEnv({
        "model_path": str(xml_path),
        "render_mode": "rgb_array",
        "framerate": FRAMERATE,
        "duration": DURATION,
        "reward_config": {"type": REWARD_FUNCTION},
        "frame_skip": FRAME_SKIP
    })
    
    # Reset environment
    obs, info = env.reset()
    
    # Run the model and render
    for step in range(num_steps):
        # Get action from the policy
        action, _ = model.predict(obs, deterministic=True) # make deterministic (critical)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render frame
        env.render()
        
        # Check for episode end
        if terminated or truncated:
            print(f"Episode ended after {step} steps")
            print(f"Termination reason: {info.get('truncation_info', {}).get('reason', 'timeout')}")
            break
    
    # Save the video
    if len(env.frames) > 0:
        env.save_video(episode_num=-1) # save the demonstration as episode -1
        print("Total reward: ", info.get('total_reward', 0))
    else:
        print("No frames were recorded!")
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    # Path to your trained model
    model_path = "sb3_results/final_model.zip"
    
    # Render and save video
    render_trained_model(model_path)
    print("\nVideo has been saved to the 'recordings' directory")