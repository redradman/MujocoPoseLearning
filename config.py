"""
Sample configuration file for humanoid training
"""

config = {
    # Environment parameters
    "env_kwargs": {
        "total_timesteps": 5_000_000,
        "render_interval": 2500,
        "n_envs": 8,
        "reward_function": "stand",
        "frame_skip": 3,
        "framerate": 60
    },
    
    # PPO parameters
    "ppo_kwargs": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 20,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "policy_kwargs": {
            "activation_fn": "ReLU",  # Will be converted to torch.nn.ReLU
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64]
            }
        }
    }
} 