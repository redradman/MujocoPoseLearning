import mujoco
import numpy as np
import mediapy as media
# import os
from gymnasium import spaces
from pathlib import Path

class HumanoidEnv:
    def __init__(self, env_config):
        # Handle both dict and string inputs
        if isinstance(env_config, dict):
            self.model_path = env_config.get('model_path')
            self.duration = env_config.get('duration', 3.8)
            self.framerate = env_config.get('framerate', 60)
            self.render_mode = env_config.get('render_mode')
            self.render_interval = env_config.get('render_interval', 100)
        else:
            # For backward compatibility
            self.model_path = env_config
            self.duration = 3.8
            self.framerate = 60
            self.render_mode = None
            self.render_interval = 100

        print("Initializing environment with:")
        print(f"- model_path: {self.model_path}")
        print(f"- render_mode: {self.render_mode}")
        print(f"- framerate: {self.framerate}")
        print(f"- render_interval: {self.render_interval}")
        
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.frames = []
        self.renderer = None
        self.step_count = 0
        
        # Initialize renderer if we're going to render
        if self.render_mode == "rgb_array":
            print("Creating renderer during init")
            self.renderer = mujoco.Renderer(self.model)
        
        num_observations = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_observations,),
            dtype=np.float64
        )

        num_actions = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_actions,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Clear frames and renderer on reset
        self.frames = []
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        
        # Get initial state
        state = self._get_state()
        
        # Return state and info dict with consistent structure
        info = {
            'reward_components': {
                'forward': 0.0,
                'standing': 0.0,
                'healthy_pose': 0.0,
                'alive': 0.0,
                'total': 0.0
            },
            'height': self.data.qpos[2],
            'forward_velocity': self.data.qvel[0],
            'truncated': False,
            'terminated': False
        }
        
        self.step_count = 0
        
        return state, info

    def step(self, action):
        """Execute one environment step."""
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        state = self._get_state()
        
        # Check for truncation first
        height = self.data.qpos[2]
        truncated = height < 0.8  # Early truncation if COM is too low
        
        # Compute reward (will be 0 if truncated)
        reward = self._compute_reward()
        
        # Check for termination (episode timeout)
        terminated = self.data.time >= self.duration
        
        # Add detailed information to info dict
        info = {
            'reward_components': getattr(self, 'reward_components', {}),
            'height': height,
            'forward_velocity': self.data.qvel[0],
            'truncated': truncated,
            'terminated': terminated
        }
        
        # Increment step counter
        self.step_count += 1
        
        # Render if needed
        if self.render_mode == "rgb_array":
            self.render()
        
        return state, reward, terminated, truncated, info

    def _get_state(self):
        state = np.concatenate([self.data.qpos, self.data.qvel])
        print(f"State shape: {state.shape}")  # Debugging output
        return state

    def _compute_reward(self):
        """Compute the reward with scaled-down values and focus on stability."""
        height = self.data.qpos[2]  # z-position
        forward_vel = self.data.qvel[0]  # x-velocity
        
        # Get orientation
        orientation = self.data.qpos[3:7]  # quaternion
        euler = self.quaternion_to_euler(orientation)
        roll, pitch = euler[0], euler[1]
        
        self.reward_components = {
            'stability': 0.0,
            'posture': 0.0,
            'height': 0.0,
            'forward': 0.0,
            'energy': 0.0
        }
        
        # Early termination conditions
        if (height < 0.7 or 
            abs(roll) > 0.8 or 
            abs(pitch) > 0.8):
            return -1.0  # Negative reward for falling
        
        # Base reward for staying alive (small)
        base_reward = 0.1
        
        # 1. Stability Reward (max 0.3)
        orientation_penalty = np.square(roll) + np.square(pitch)
        self.reward_components['stability'] = 0.3 * np.exp(-3.0 * orientation_penalty)
        
        # 2. Height Maintenance (max 0.2)
        target_height = 1.3
        height_diff = abs(height - target_height)
        self.reward_components['height'] = 0.2 * np.exp(-2.0 * height_diff)
        
        # 3. Posture Control (max 0.2)
        joint_angles = self.data.qpos[7:]
        joint_velocities = self.data.qvel[6:]
        
        angle_penalty = np.sum(np.square(joint_angles)) * 0.02
        velocity_penalty = np.sum(np.square(joint_velocities)) * 0.02
        self.reward_components['posture'] = 0.2 * np.exp(-(angle_penalty + velocity_penalty))
        
        # 4. Forward Motion (max 0.1)
        if height > 1.0 and abs(roll) < 0.3 and abs(pitch) < 0.3:
            self.reward_components['forward'] = 0.1 * np.clip(forward_vel, 0, 1.0)
        
        # 5. Energy Efficiency (max 0.1)
        ctrl = self.data.ctrl
        energy_penalty = np.sum(np.square(ctrl)) * 0.05
        self.reward_components['energy'] = 0.1 * np.exp(-energy_penalty)
        
        # Combine rewards
        total_reward = base_reward + sum(self.reward_components.values())
        
        # Small bonus for very good stability (max additional 0.1)
        if (height > 1.0 and 
            abs(roll) < 0.2 and 
            abs(pitch) < 0.2 and
            forward_vel >= 0):
            total_reward += 0.1
        
        return float(total_reward)

    def quaternion_to_euler(self, quat):
        """Convert quaternion to euler angles."""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

    def render(self):
        """Render the current frame."""
        if self.render_mode != "rgb_array":
            return
        
        # Create renderer if it doesn't exist
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model)
        
        # Match the working example exactly
        if len(self.frames) < self.data.time * self.framerate:
            self.renderer.update_scene(self.data)
            pixels = self.renderer.render()
            self.frames.append(pixels)  # Store raw pixels without conversion

    def save_video(self, episode_num):
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        
        # Create video path inside recordings directory
        video_path = recordings_dir / f"humanoid_episode_{episode_num}.mp4"

        # Delete existing file if it exists
        if video_path.exists():
            video_path.unlink()
        
        media.write_video(
            str(video_path),
            self.frames,
            fps=self.framerate,
            codec='h264',
        )
        
        # Clear frames after saving
        self.frames = []
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
