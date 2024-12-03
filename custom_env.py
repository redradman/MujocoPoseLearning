import mujoco
import numpy as np
import mediapy as media
from gymnasium import spaces
from pathlib import Path
from reward_functions import REWARD_FUNCTIONS

CLIP_OBSERVATION_VALUE = np.inf # decided on no clipping for now (might have to revise if experiencing exploding gradient problem)
ACTION_CLIP_VALUE = 0.4 # change from 0.4 to 0.6 to allow for more movement (might have to revise again)

class HumanoidEnv:
    def __init__(self, env_config):
        # Handle both dict and string inputs
        if isinstance(env_config, dict):
            self.model_path = env_config.get('model_path')
            self.duration = env_config.get('duration', 3.8)
            self.framerate = env_config.get('framerate', 60)
            self.render_mode = env_config.get('render_mode')
            self.render_interval = env_config.get('render_interval', 100)
            self.reward_config = env_config.get('reward_config', {'type': 'default'})
        else:
            # For backward compatibility
            self.model_path = env_config
            self.duration = 3.8
            self.framerate = 60
            self.render_mode = None
            self.render_interval = 100
            self.reward_config = {'type': 'default'}

        print("Initializing environment with:")
        print(f"- model_path: {self.model_path}")
        print(f"- render_mode: {self.render_mode}")
        print(f"- framerate: {self.framerate}")
        print(f"- render_interval: {self.render_interval}")
        print(f"- reward_config: {self.reward_config}")
        
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
            low=-CLIP_OBSERVATION_VALUE,
            high=CLIP_OBSERVATION_VALUE,
            shape=(num_observations,),
            dtype=np.float64
        )

        num_actions = self.model.nu
        self.action_space = spaces.Box(
            low=-ACTION_CLIP_VALUE,
            high=ACTION_CLIP_VALUE,
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
        
        # Get key state information
        height = self.data.qpos[2]
        orientation = self.data.qpos[3:7]
        euler = self.quaternion_to_euler(orientation)
        roll, pitch = euler[0], euler[1]
        
        ############# Check multiple conditions for truncation #############
        truncated = False
        truncation_info = {}
        
        # Height check (too low or too high)
        if height < 0.7:
            truncated = True
            truncation_info['reason'] = 'too_low'
        elif height > 2.0:  # Jumping/unstable behavior
            truncated = True
            truncation_info['reason'] = 'too_high'
        
        # Orientation check (falling over)
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            truncated = True
            truncation_info['reason'] = 'bad_orientation'

        # Joint angle limits
        joint_angles = self.data.qpos[7:]
        if np.any(np.abs(joint_angles) > 2.0):  # ~115 degrees
            truncated = True
            truncation_info['reason'] = 'joint_limit'
        
        # Energy consumption check
        # if np.sum(np.square(self.data.ctrl)) > 100.0:
        #     truncated = True
        #     truncation_info['reason'] = 'excessive_force'
        ###################### End check for truncation ######################

        # Compute reward (will be 0 if truncated)
        reward = self._compute_reward()
        
        # Check for termination (episode timeout)
        terminated = self.data.time >= self.duration
        
        # Add detailed information to info dict
        info = {
            'reward_components': getattr(self, 'reward_components', {}),
            'height': height,
            'orientation': {
                'roll': roll,
                'pitch': pitch
            },
            'forward_velocity': self.data.qvel[0],
            'truncated': truncated,
            'truncation_info': truncation_info,
            'terminated': terminated
        }
        
        # Increment step counter
        self.step_count += 1
        
        # Render if needed
        if self.render_mode == "rgb_array":
            self.render()
        
        return state, reward, terminated, truncated, info

    def _get_state(self):
        """Get the current state of the environment."""
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        state = np.concatenate([qpos, qvel])
        return np.clip(state, -CLIP_OBSERVATION_VALUE, CLIP_OBSERVATION_VALUE)

    def _compute_reward(self):
        """Compute reward based on the configured reward function."""
        reward_type = self.reward_config.get('type', 'default')
        reward_params = self.reward_config.get('params', None)
        
        if reward_type not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        return REWARD_FUNCTIONS[reward_type](self.data, reward_params)

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
