import mujoco
import numpy as np
import mediapy as media
from gymnasium import spaces, Env
from pathlib import Path
# from utils import quaternion_to_euler
from reward_functions import REWARD_FUNCTIONS

CLIP_OBSERVATION_VALUE = np.inf # decided on no clipping for now (might have to revise if experiencing exploding gradient problem)
ACTION_CLIP_VALUE = 1 # allow the full range of motion

class HumanoidEnv(Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }
    
    def __init__(self, env_config):
        super().__init__()
        # Handle both dict and string inputs
        if isinstance(env_config, dict):
            self.model_path = env_config.get('model_path')
            self.duration = env_config.get('duration', 15)
            self.framerate = env_config.get('framerate', 60)
            self.render_mode = env_config.get('render_mode')
            self.render_interval = env_config.get('render_interval', 100)
            self.reward_config = env_config.get('reward_config', {'type': 'default'})
            self.frame_skip = env_config.get('frame_skip', 5)  # Default to 5 like gym
            self.grace_period_length = env_config.get('grace_period_length', 300)
            self.grace_period_steps = 0
            self.total_reward = 0.0
            self.run_name = env_config.get('run_name')
        else:
            # For backward compatibility
            self.model_path = env_config
            self.duration = 15
            self.framerate = 60
            self.render_mode = None
            self.render_interval = 100
            self.reward_config = {'type': 'default'}
            self.frame_skip = 5
            self.grace_period_steps = 0
            self.total_reward = 0.0

        # print("Initializing environment with:")
        # print(f"- model_path: {self.model_path}")
        # print(f"- render_mode: {self.render_mode}")
        # print(f"- framerate: {self.framerate}")
        # print(f"- render_interval: {self.render_interval}")
        # print(f"- reward_config: {self.reward_config}")
        # print(f"- frame_skip: {self.frame_skip}")
        
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.frames = []
        self.renderer = None
        self.step_count = 0
        self.init_qpos = self.data.qpos.copy()
        self.init_qpos[2] = 1.282  # Set initial height
        self.init_qpos[3:7] = [1, 0, 0, 0]  # Set quaternion to upright orientation
        self.init_qvel = np.zeros_like(self.data.qvel)  # Zero initial velocities
        
        # Initialize renderer if we're going to render
        if self.render_mode == "rgb_array":
            print("Creating renderer during init")
            self.renderer = mujoco.Renderer(self.model)

        sample_state = self._get_state()
        obs_size = sample_state.size
        # obs_size += self.data.crtl.size
                
        # num_observations = (
        #     self.model.nq +      # Joint positions
        #     self.model.nv +      # Joint velocities
        #     3 +                  # COM position (x, y, z)
        #     3 +                  # COM velocity (x, y, z)
        #     self.model.nbody * 6 # Contact forces (6D per body)
        # )
        
        self.observation_space = spaces.Box(
            low=-CLIP_OBSERVATION_VALUE,
            high=CLIP_OBSERVATION_VALUE,
            shape=(obs_size,),
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
        
        # Set initial pose
        self.data.qpos[:] = self.init_qpos.copy()
        self.data.qvel[:] = self.init_qvel.copy()
        
        # Smaller noise for stability
        pos_noise = np.random.uniform(low=-0.01, high=0.01, size=self.data.qpos.shape)
        vel_noise = np.random.uniform(low=-0.01, high=0.01, size=self.data.qvel.shape)
        
        # Don't add noise to critical components
        pos_noise[2] *= 0.1  # Reduce height noise
        pos_noise[3:7] = 0  # No noise in orientation
        
        self.data.qpos[:] += pos_noise
        self.data.qvel[:] += vel_noise
        
        # Forward the simulation a tiny bit to stabilize
        for _ in range(1):
            mujoco.mj_step(self.model, self.data)
        
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
        self.total_reward = 0.0
        
        return state, info

    def step(self, action):
        """Modified step function to incorporate frame skipping"""
        # Increment step counter before any checks
        self.step_count += 1

        # Apply the action for frame_skip times
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)
        
        # Get state information
        state = self._get_state()
        height = self.data.qpos[2]


        # Truncation condition with adjusted grace period

        # Adjusted grace period steps (e.g., 240 steps for ~6 seconds)
        # grace_period_length = 100
        # if height < 0.8:
        #     self.grace_period_steps += 1
        #     if self.grace_period_steps >= grace_period_length:
        #         truncated = True
        #         truncation_info['reason'] = 'collapsed'
        #         reward = 0  
        #     else:
        #         reward *= ((grace_period_length - self.step_count) / grace_period_length) * 0.3
        # elif height > 0.8:
        #     self.grace_period_steps = 0

        # Truncation condition
        # min_height = 0.8
        # max_roll_pitch = np.pi / 4  # 45 degrees
        # truncated = False
        # truncation_info = {}

        # if height < min_height:
        #     truncated = True
        #     truncation_info['reason'] = 'fallen'
        #     reward = 0.0
        # else:
        #     # Check for excessive tilt
        #     orientation = self.data.qpos[3:7]
        #     roll, pitch, _ = quaternion_to_euler(orientation)

        #     if abs(roll) > max_roll_pitch or abs(pitch) > max_roll_pitch:
        #         truncated = True
        #         truncation_info['reason'] = 'lost_balance'
        #         reward = 0.0
        truncated = False
        truncation_info = {}
        if self.step_count >= 750:  # Force truncation after N steps
            truncated = True
            truncation_info['reason'] = 'timeout'
            reward = 0.0
        # Calculate reward if not truncated
        else:
            reward = self._compute_reward()
        self.total_reward += reward

        # Termination condition (episode timeout)
        terminated = self.data.time >= self.duration

        # Info dictionary
        info = {
            'reward_components': getattr(self, 'reward_components', {}),
            'height': height,
            'step_count': self.step_count,
            'truncated': truncated,
            'truncation_info': truncation_info,
            'terminated': terminated,
            'total_reward': self.total_reward
        }

        # Render if needed
        if self.render_mode == "rgb_array":
            self.render()
        
        return state, reward, terminated, truncated, info

    def _get_state(self):
        """Get the current state of the environment.
        
        Returns a concatenated array of:
        - Joint positions (qpos)
        - Joint velocities (qvel)
        - Center of mass position
        - Center of mass velocity
        - External contact forces
        """
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        # com_inertia = self.data.cinert.flatten()
        # com_velocity = self.data.cvel.flatten()
        # actuator_forces = self.data.qfrc_actuator.flatten()
        # contact_forces = self.data.cfrc_ext[1:].flatten()

        state = np.concatenate((
            position[2:],
            velocity,
            # com_inertia,
            # com_velocity,
            # actuator_forces,
            # contact_forces,
        ))

        # print(state)
        # print("*-------*")

        return state

    def _compute_reward(self):
        """Compute reward based on the configured reward function."""
        reward_type = self.reward_config.get('type', 'default')
        reward_params = self.reward_config.get('params', None)
        
        if reward_type not in REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward type: {reward_type}")
        
        return REWARD_FUNCTIONS[reward_type](self.data, reward_params)

    def render(self):
        """Render the current frame."""
        if self.render_mode != "rgb_array":
            return
        
        # Create renderer if it doesn't exist
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Match the working example exactly
        if len(self.frames) < self.data.time * self.framerate:
            self.renderer.update_scene(self.data)
            # Explicitly select the 'back' camera
            camera_id = self.model.camera('side').id
            self.renderer.update_scene(self.data, camera=camera_id)
            pixels = self.renderer.render()
            self.frames.append(pixels)

    def save_video(self, episode_num):
        recordings_dir = Path("recordings")
        
        # If run_name is provided, create a subdirectory for this experiment
        if hasattr(self, 'run_name'):
            recordings_dir = recordings_dir / self.run_name
        
        recordings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create video path inside recordings directory
        video_path = recordings_dir / f"episode_{episode_num}.mp4"

        # Delete existing file if it exists
        if video_path.exists():
            video_path.unlink()
        
        if len(self.frames) > 0:
            media.write_video(
                str(video_path),
                self.frames,
                fps=self.framerate,
                codec='h264',
            )
        else:
            print("No frames to save!")
        
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
