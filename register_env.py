# from ray.tune import register_env
from custom_env import HumanoidEnv

def env_creator(env_config):
    print("Creating environment with config:", env_config)
    env = HumanoidEnv(env_config)
    print(f"Environment created with render_mode: {env.render_mode}")
    return env

# No need to register here as it's done in train_rllib.py 