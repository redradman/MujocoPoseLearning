from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
from custom_env import HumanoidEnv
import torch
from pathlib import Path

RENDER_INTERVAL = 100
N_ENVS = 12

class VideoRecorderCallback(BaseCallback):
    def __init__(self, render_interval: int, env_id: int = 0, verbose=1):
        super().__init__(verbose)
        self.render_interval = render_interval
        self.env_id = env_id
        self.episode_count = 0
        
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
                "type": "mujoco_humanoid",
            }
        })

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.episode_count += 1
        if self.episode_count % self.render_interval == 0:
            # Get the current policy
            obs = self.render_env.reset()[0]
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.render_env.step(action)
                done = terminated or truncated
            self.render_env.save_video(self.episode_count)

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
            "type": "mujoco_humanoid",
        }
    }

    # Create vectorized environment
    env = SubprocVecEnv([make_env(env_config, i) for i in range(N_ENVS)])

    # Define network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        ),
        activation_fn=torch.nn.ReLU
    )

    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=5.0,
        ent_coef=0.1,
        vf_coef=0.5,
        max_grad_norm=0.6,
        use_sde=True,
        sde_sample_freq=4,
        tensorboard_log=str(storage_path / "tensorboard_logs"),
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    # Setup callbacks
    callbacks = CallbackList([
        ProgressBarCallback(),  # Progress bar
        VideoRecorderCallback(render_interval=RENDER_INTERVAL)  # Video recording
    ])

    # Train the model
    TOTAL_TIMESTEPS = 1_000_000
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks
    )

    # Save the final model
    model.save(str(storage_path / "final_model"))

if __name__ == '__main__':
    main() 