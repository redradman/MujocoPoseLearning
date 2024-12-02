import ray
from ray import tune, train
from ray.tune import register_env
from pathlib import Path
import torch
from register_env import env_creator  # Import the env_creator
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Register the custom environment
register_env("HumanoidEnv", env_creator)

# Check for MPS availability
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
@ray.remote
class EpisodeCounter:
    def __init__(self):
        self.count = 0
    
    def increment_and_check(self):
        self.count += 1
        return self.count
    
    def get_count(self):
        return self.count

# First, create a global counter actor before training starts
episode_counter = None

class RenderingCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_counter = 0
    
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        self.episode_counter += 1
        env = base_env.get_sub_environments()[0]
        env.render_mode = "rgb_array"
        env.frames = []

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        env = base_env.get_sub_environments()[0]
        
        # Get render interval from config
        render_interval = env.render_interval
        
        # Save video based on configured interval
        if self.episode_counter % render_interval == 0:
            env.save_video(self.episode_counter)
        else:
            # Clear frames without saving
            env.frames = []
            if env.renderer:
                env.renderer.close()
                env.renderer = None

# Get the absolute path to your project root
project_root = Path(__file__).parent
xml_path = project_root / "XML" / "humanoid.xml"

# Define storage path for results
storage_path = project_root / "ray_results"
storage_path.mkdir(parents=True, exist_ok=True)

# Verify the file exists
if not xml_path.exists():
    raise FileNotFoundError(f"XML file not found at: {xml_path}")

# Print the paths for debugging
print(f"Using XML file at: {xml_path}")
print(f"Saving results to: {storage_path}")

# Initialize Ray
ray.init(
    dashboard_host="127.0.0.1",
    include_dashboard=True,
    ignore_reinit_error=True,
    log_to_driver=False,
)

# THEN create the config
config = {
    "env_config": {
        "model_path": str(xml_path),
        "render_mode": "rgb_array",
        "framerate": 60,
        "duration": 30.0,
        "render_interval": 100,
        "reward_config": {
            "type": "stand_still",
            # params below is not explcitly used but it is a good way to pass parameters into the reward functions
            # "params": {
            #     # Any parameters specific to the reward function
            #     "forward_weight": 1.0,
            #     "energy_weight": 0.1,
            # }
        }
    },

    # "framework": "torch", # Use PyTorch
    # Reduce number of workers for rendering
    "num_workers": 8,  # Use only one worker for rendering
    "num_envs_per_worker": 1,
    "use_gae": True,
    # Reduce learning rate and use a more conservative schedule
    "lr": 5e-4,
    "lr_schedule": [
        [0, 5e-4],
        [100_000, 2e-4],
        [1_000_000, 1e-4],
    ],
    
    # More conservative PPO settings
    "clip_param": 0.3,             
    "entropy_coeff": 0.01,        
    "gamma": 0.995,          
    "lambda_": 0.95,           
    
    # Smaller batch sizes for more stable updates
    "train_batch_size": 8000,      
    "sgd_minibatch_size": 128,      
    "num_sgd_iter": 10,             
    
    "vf_clip_param": 5.0,
    # Add gradient clipping
    "grad_clip": 0.6,               # Add gradient clipping
    
    # More conservative exploration
    "exploration_config": {
        "type": "StochasticSampling",
        "random_timesteps": 60000,    
    },
    
    # Normalize observations
    "normalize_actions": True,
    "normalize_observations": True,
    "normalize_advantages": True,
    
    # Rest of your existing config...
    "env": "HumanoidEnv",
    
   # Add model configuration
    "model": {
        "fcnet_hiddens": [128, 64, 128],
        # "fcnet_activation": "tanh",
        "vf_share_layers": False,    # Separate value network
        "free_log_std": True,
    },
    
    # Add the callbacks configuration
    "callbacks": RenderingCallbacks,  # Add this line to enable rendering callbacks
}

# Run the training
tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=train.RunConfig(
        storage_path=str(storage_path),
        name="humanoid_training",
        stop={"training_iteration": 10000},
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=50,
            checkpoint_score_attribute="episode_reward_mean",
            num_to_keep=10,
            checkpoint_at_end=True
        ),
        verbose=3,
    ),
)

tuner.fit()










# import logging

# # Set logging levels
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("ray").setLevel(logging.INFO)
# logging.getLogger("ray.rllib").setLevel(logging.INFO)
# logging.getLogger("ray.tune").setLevel(logging.INFO)

# # Add a custom logger for our rendering
# render_logger = logging.getLogger("render")
# render_logger.setLevel(logging.DEBUG)

# # Create a console handler with a higher log level
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# render_logger.addHandler(console_handler)



# call back fucntion with logging for debugging

# class RenderingCallbacks(DefaultCallbacks):
#     def __init__(self):
#         super().__init__()
#         self.episode_counter = 0
#         render_logger.info("Initialized RenderingCallbacks")
    
#     def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
#         self.episode_counter += 1
#         env = base_env.get_sub_environments()[0]
#         render_logger.info(f"=== Starting render for episode {self.episode_counter} ===")
#         env.render_mode = "rgb_array"
#         env.frames = []
#         render_logger.debug("Render mode set to rgb_array, frames initialized")

#     def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
#         env = base_env.get_sub_environments()[0]
#         render_logger.info(f"=== Ending episode {self.episode_counter} ===")
        
#         # Get render interval from config
#         render_interval = env.render_interval
        
#         # Save video based on configured interval
#         if self.episode_counter % render_interval == 0:
#             render_logger.info(f"Saving video for episode {self.episode_counter}")
#             render_logger.debug(f"Number of frames collected: {len(env.frames)}")
#             render_logger.debug(f"Episode length: {episode.length}")
#             env.save_video(self.episode_counter)
#         else:
#             # Clear frames without saving
#             env.frames = []
#             if env.renderer:
#                 env.renderer.close()
#                 env.renderer = None

# from ray.rllib.algorithms.ppo import PPO