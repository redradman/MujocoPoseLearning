from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
from custom_env import HumanoidEnv
from pathlib import Path
from multiprocessing import Value
import ctypes
import numpy as np

# Global synchronized counter
global_episode_count = Value(ctypes.c_int, 0)

class VideoRecorderCallback(BaseCallback):
    def __init__(self, render_interval: int, episode_counter: Value, env_kwargs: dict, run_name: str, env_id: int = 0, verbose=1):
        super().__init__(verbose)
        self.render_interval = render_interval
        self.env_id = env_id
        self.episode_counter = episode_counter
        self.env_kwargs = env_kwargs
        self.run_name = run_name
        
    def _init_callback(self) -> None:
        project_root = Path(__file__).parent
        xml_path = project_root / "XML" / "humanoid.xml"
        
        self.render_env = HumanoidEnv({
            "model_path": str(xml_path),
            "render_mode": "rgb_array",
            "framerate": self.env_kwargs.get('framerate', 60),
            "duration": 10.0,
            "reward_config": {
                "type": self.env_kwargs.get('reward_function', 'walk'),
            },
            "frame_skip": self.env_kwargs.get('frame_skip', 3),
            "run_name": self.run_name  # Pass run_name to environment
        })
        
        # Initialize episode tracking for each env
        self.episode_counts = np.zeros(self.env_kwargs.get('n_envs', 8), dtype=np.int32)

    def _on_step(self) -> bool:
        # Check for episode terminations
        dones = self.locals.get('dones')  # Get done signals from all envs
        if dones is not None:
            for env_idx, done in enumerate(dones):
                if done:
                    with self.episode_counter.get_lock():
                        self.episode_counter.value += 1
                        current_episode = self.episode_counter.value
                        
                    # Only render on the specified interval
                    if current_episode % self.render_interval == 0:
                        # Get the current policy
                        obs = self.render_env.reset()[0]
                        done = False
                        while not done:
                            action, _ = self.model.predict(obs, ) # removed deterministic = True 
                            obs, _, terminated, truncated, _ = self.render_env.step(action)
                            done = terminated or truncated
                        self.render_env.save_video(current_episode)  # Add this line to save the video
        return True

    def _on_rollout_end(self) -> None:
        pass  # Episode counting is now handled in _on_step

class RewardStatsCallback(BaseCallback):
    """
    Callback for logging episode reward statistics to TensorBoard:
    - mean episode reward
    - min episode reward
    - max episode reward
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Get the most recent reward and add it to current episode total
        rewards = self.locals['rewards']  # Get rewards for all envs
        dones = self.locals['dones']      # Get done flags for all envs
        
        # Update current episode rewards
        for reward, done in zip(rewards, dones):
            self.current_episode_reward += reward
            
            if done:
                # Episode finished, store the total reward
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                
                # Log stats if we have enough episodes
                if len(self.episode_rewards) >= 10:  # Log every 10 episodes
                    mean_reward = np.mean(self.episode_rewards)
                    min_reward = np.min(self.episode_rewards)
                    max_reward = np.max(self.episode_rewards)
                    
                    # Log to tensorboard
                    self.logger.record("reward/mean_episode", mean_reward)
                    self.logger.record("reward/min_episode", min_reward)
                    self.logger.record("reward/max_episode", max_reward)
                    
                    # Clear buffer
                    self.episode_rewards = []
        
        return True

def make_env(env_config, rank):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = HumanoidEnv(env_config)
        return env
    return _init

def generate_run_name(env_kwargs, ppo_kwargs):
    """Generate a descriptive name from hyperparameters."""
    # Extract all PPO parameters
    components = [
        f"lr{ppo_kwargs.get('learning_rate', 'unk')}",
        f"bs{ppo_kwargs.get('batch_size', 'unk')}",
        f"nsteps{ppo_kwargs.get('n_steps', 'unk')}",
        f"nepochs{ppo_kwargs.get('n_epochs', 'unk')}",
        f"gamma{ppo_kwargs.get('gamma', 'unk')}",
        f"gae{ppo_kwargs.get('gae_lambda', 'unk')}",
        f"clip{ppo_kwargs.get('clip_range', 'unk')}",
        f"ent{ppo_kwargs.get('ent_coef', 'unk')}",
    ]
    
    # Add network architecture if available
    if 'policy_kwargs' in ppo_kwargs and 'net_arch' in ppo_kwargs['policy_kwargs']:
        net_arch = ppo_kwargs['policy_kwargs']['net_arch']
        if 'pi' in net_arch:
            components.append(f"pi{'-'.join(map(str, net_arch['pi']))}")
        if 'vf' in net_arch:
            components.append(f"vf{'-'.join(map(str, net_arch['vf']))}")
    
    # Add environment parameters
    components.extend([
        f"nenvs{env_kwargs.get('n_envs', 'unk')}",
        f"rew{env_kwargs.get('reward_function', 'unk')}"
    ])
    
    # Join components with underscores
    return "PPO_" + "_".join(components)

def train_humanoid(env_kwargs, ppo_kwargs):
    # Get the absolute path to your project root
    project_root = Path(__file__).parent
    xml_path = project_root / "XML" / "humanoid.xml"
    storage_path = project_root / "sb3_results"
    storage_path.mkdir(parents=True, exist_ok=True)

    # Generate descriptive run name
    run_name = generate_run_name(env_kwargs, ppo_kwargs)
    tensorboard_log_dir = storage_path / "tensorboard_logs" / run_name
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    # Base environment configuration
    env_config = {
        "model_path": str(xml_path),
        "render_mode": None,
        "framerate": env_kwargs.get('framerate', 60),
        "duration": 10.0,
        "reward_config": {
            "type": env_kwargs.get('reward_function', 'walk'),
            # "params": {
            #     "target_height": 1.2,
            #     "min_height": 0.8,
            #     "max_roll_pitch": np.pi / 6,
            #     "height_weight": 1.0,
            #     "orientation_weight": 1.0,
            #     "time_weight": 0.1
            # }
        },
        "frame_skip": env_kwargs.get('frame_skip', 3),
    }

    # Create vectorized environment
    env = SubprocVecEnv([make_env(env_config, i) for i in range(env_kwargs.get('n_envs', 8))])

    # Create the model
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml can be used for guideance
    # https://github.com/openai/baselines/blob/master/baselines/ppo1/run_mujoco.py also very useful
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log=str(tensorboard_log_dir),
        verbose=1,
        **ppo_kwargs
    )

    # Setup callbacks
    callbacks = CallbackList([
        ProgressBarCallback(),  # Progress bar
        RewardStatsCallback(), # add reward stats to the tensorboard
        VideoRecorderCallback(
            render_interval=env_kwargs.get('render_interval', 2500),
            episode_counter=global_episode_count,
            env_kwargs=env_kwargs,  # Pass env_kwargs to the callback
            run_name=run_name  # Pass run_name to the callback
        )  # Video recording
    ])
    # Train the model
    model.learn(
        total_timesteps=env_kwargs.get('total_timesteps', 20_000_000),
        callback=callbacks
    )

    # Save the final model
    model.save(str(storage_path / "final_model"))

    env.close() 