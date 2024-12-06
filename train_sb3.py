from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
from custom_env import HumanoidEnv
import torch
from pathlib import Path
from multiprocessing import Value
import ctypes
import numpy as np

RENDER_INTERVAL = 1000
N_ENVS = 10
REWARD_FUNCTION = "walk"
# Global synchronized counter
global_episode_count = Value(ctypes.c_int, 0)

class VideoRecorderCallback(BaseCallback):
    def __init__(self, render_interval: int, episode_counter: Value, env_id: int = 0, verbose=1):
        super().__init__(verbose)
        self.render_interval = render_interval
        self.env_id = env_id
        self.episode_counter = episode_counter
        
    def _init_callback(self) -> None:
        # Create a single environment for rendering when callback is initialized
        project_root = Path(__file__).parent
        xml_path = project_root / "XML" / "humanoid.xml"
        
        self.render_env = HumanoidEnv({
            "model_path": str(xml_path),
            "render_mode": "rgb_array",
            "framerate": 60,
            "duration": 30.0,
            "reward_config": {
                "type": REWARD_FUNCTION,
            }
        })
        
        # Initialize episode tracking for each env
        self.episode_counts = np.zeros(N_ENVS, dtype=np.int32)

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
                            action, _ = self.model.predict(obs, deterministic=True)
                            obs, _, terminated, truncated, _ = self.render_env.step(action)
                            done = terminated or truncated
                        self.render_env.save_video(current_episode)
        return True

    def _on_rollout_end(self) -> None:
        pass  # Episode counting is now handled in _on_step

class RewardStatsCallback(BaseCallback):
    """
    Callback for logging reward statistics to TensorBoard:
    - mean reward
    - min reward
    - max reward
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards_buffer = []
        
    def _on_step(self) -> bool:
        # Get the most recent reward
        reward = self.locals['rewards'][0]  # [0] because we want scalar value
        self.rewards_buffer.append(reward)
        
        # Log stats every 100 steps
        if len(self.rewards_buffer) >= 100:
            mean_reward = np.mean(self.rewards_buffer)
            min_reward = np.min(self.rewards_buffer)
            max_reward = np.max(self.rewards_buffer)
            
            # Log to tensorboard
            self.logger.record("reward/mean", mean_reward)
            self.logger.record("reward/min", min_reward)
            self.logger.record("reward/max", max_reward)
            
            # Clear buffer
            self.rewards_buffer = []
            
        return True


def make_env(env_config, rank):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = HumanoidEnv(env_config)
        return env
    return _init

def main():
    # Get the absolute path to your project root
    project_root = Path(__file__).parent
    xml_path = project_root / "XML" / "humanoid.xml"
    storage_path = project_root / "sb3_results"
    storage_path.mkdir(parents=True, exist_ok=True)

    # Base environment configuration
    env_config = {
        "model_path": str(xml_path),
        "render_mode": None,
        "framerate": 60,
        "duration": 30.0,
        "reward_config": {
            "type": REWARD_FUNCTION,
        }
    }

    # Create vectorized environment
    env = SubprocVecEnv([make_env(env_config, i) for i in range(N_ENVS)])

    # Define network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        ),
        activation_fn=torch.nn.Tanh
    )

    # Create the model
    # values adopted from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=64,
        # target_kl=0.02,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        # ent_coef=0.01,
        max_grad_norm=0.4,
        use_sde=True,
        sde_sample_freq=4,
        tensorboard_log=str(storage_path / "tensorboard_logs"),
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    # Setup callbacks
    callbacks = CallbackList([
        ProgressBarCallback(),  # Progress bar
        RewardStatsCallback(), # add reward stats to the tensorboard
        VideoRecorderCallback(render_interval=RENDER_INTERVAL, episode_counter=global_episode_count)  # Video recording
    ])

    # Train the model
    TOTAL_TIMESTEPS = 10_000_000
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks
    )

    # Save the final model
    model.save(str(storage_path / "final_model"))

if __name__ == '__main__':
    main() 