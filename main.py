from train_sb3 import train_humanoid
import argparse
import torch
import importlib.util

def load_config_from_file(config_path):
    """Load configuration from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Convert activation function string to actual torch class if needed
    if 'policy_kwargs' in config_module.config.get('ppo_kwargs', {}):
        activation_name = config_module.config['ppo_kwargs']['policy_kwargs'].get('activation_fn')
        if isinstance(activation_name, str):
            config_module.config['ppo_kwargs']['policy_kwargs']['activation_fn'] = getattr(torch.nn, activation_name)
    
    return config_module.config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a humanoid with PPO')
    
    # Add config file option
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Environment parameters
    parser.add_argument('--total_timesteps', type=int, default=20_000_000,
                        help='Total timesteps for training')
    parser.add_argument('--render_interval', type=int, default=2500,
                        help='Interval between video recordings')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='Number of parallel environments')
    parser.add_argument('--reward_function', type=str, default="walk",
                        help='Type of reward function to use')
    parser.add_argument('--frame_skip', type=int, default=3,
                        help='Number of frames to skip')
    parser.add_argument('--framerate', type=int, default=60,
                        help='Framerate for rendering')
    
    # PPO hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048,
                        help='Number of steps per update')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Minibatch size')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='Clipping parameter')
    parser.add_argument('--ent_coef', type=float, default=0.0,
                        help='Entropy coefficient')
    
    # Policy network parameters
    parser.add_argument('--net_arch_pi', type=int, nargs='+', default=[64, 64],
                        help='Policy network architecture')
    parser.add_argument('--net_arch_vf', type=int, nargs='+', default=[64, 64],
                        help='Value function network architecture')
    
    return parser.parse_args()

def print_training_info(args, env_kwargs, ppo_kwargs):
    """Print training configuration information."""
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    
    print("\nEnvironment Parameters:")
    print("-"*30)
    for key, value in env_kwargs.items():
        print(f"{key:20}: {value}")
    
    print("\nPPO Parameters:")
    print("-"*30)
    for key, value in ppo_kwargs.items():
        if key == 'policy_kwargs':
            print("\nPolicy Network Configuration:")
            print("-"*30)
            for pk, pv in value.items():
                print(f"{pk:20}: {pv}")
        else:
            print(f"{key:20}: {value}")
    print("\n" + "="*50 + "\n")

def main():
    args = parse_args()
    
    if args.config:
        # Load configuration from file
        config = load_config_from_file(args.config)
        env_kwargs = config['env_kwargs']
        ppo_kwargs = config['ppo_kwargs']
    else:
        # Create policy kwargs dictionary
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(
                pi=args.net_arch_pi,
                vf=args.net_arch_vf
            )
        )
        
        # Create PPO kwargs dictionary
        ppo_kwargs = dict(
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs
        )
        
        # Create environment kwargs dictionary
        env_kwargs = dict(
            total_timesteps=args.total_timesteps,
            render_interval=args.render_interval,
            n_envs=args.n_envs,
            reward_function=args.reward_function,
            frame_skip=args.frame_skip,
            framerate=args.framerate
        )
    
    # Print training configuration
    print_training_info(args, env_kwargs, ppo_kwargs)
    
    # Train the model
    train_humanoid(env_kwargs=env_kwargs, ppo_kwargs=ppo_kwargs)

if __name__ == "__main__":
    main()