# Humanoid Walking Agent using MuJoCo and Stable Baselines3

## Project Overview
This project implements a reinforcement learning system designed to teach a humanoid (the `humanoid.xml` file from mujoco) locomotion **without requiring reference motions or pre-trained weights**. The implementation uses **MuJoCo physics engine** for accurate simulation and **Stable Baselines3's PPO algorithm for training**, with a focus on creating a flexible and extensible framework for robotic motion research which allows for addition and selection of custom reward function 
## Features

### 1. Custom Environment Framework
- Direct MuJoCo physics engine integration for precise simulation
- Flexible environment configuration system
- Customizable reward functions where the implemented ones focus on:
  - Forward velocity matching
  - Postural stability
  - Energy efficiency
  - Balance maintenance
### 2. Advanced Training System
- **Proximal Policy Optimization (PPO) implementation with optimized hyperparameters**:
  - Two-layer MLP network (256 units each)
  - ReLU activation functions
  - Carefully tuned learning rates and batch sizes
- **Parallel environment training using SubprocVecEnv**
- Comprehensive configuration system:

```python
config = {
    # Environment parameters
    "env_kwargs": {
        "total_timesteps": 20_000_000,
        "render_interval": 5000,
        "n_envs": 8,
        "reward_function": "stand",
        "frame_skip": 3,
        "framerate": 60
    },
    
    # PPO parameters
    "ppo_kwargs": {
        "learning_rate": 5e-5,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.002,
        "policy_kwargs": {
            "activation_fn": "ReLU",  # Will be converted to torch.nn.ReLU
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256]
            }
        }
    }
```
### 3. Visualization and Analysis Tools
- Real-time rendering during training set by `render_interval` saved to recording directory
- Detailed state monitoring including:
	- Tensorboard data saved in the `sb3_results` directory
	- Outputting critical values concerning learning to the command line
- The trained policy can be saved into an XML that can be loaded into mujoco where the keyframes would clearly show the values of torque in the joints (look at `generate_trajectories.py`)
- The trained policy can also be rendered and saved as a video into the recordings directory 
	- saved as `episode_-1.mp4`
	- key point for this rendering is that actions would be deterministic:
		- `model.predict(obs, deterministic=True)` 
## Setup and Installation
### Prerequisites
- Conda package manager
- MuJoCo physics engine
- Python 3.10
### Environment Setup
```bash
conda env create -f environment.yml
conda activate mujo
```
## Usage
### Training
The system supports various training configurations through command-line arguments or configuration files:
```bash
python main.py --config config.py
```
Key configuration options include:
- Learning rate: 5e-5
- Batch size: 128
- Network architecture: [256, 256] for both policy and value functions
- Frame skip: 3 (for simulation efficiency)
- Multiple parallel environments: 8 (default)
### Visualization
Render the deterministic and learned policy and save into the `recordings/` directory: 
```bash
python render_policy.py
```
### Trajectory Generation
Save the XML file containing all of the steps of the deterministic actions taken by the trained policy. This would allow the XML file to be loaded into Mujoco and observe the key differences
```bash
python generate_trajectories.py
```

## Technical Implementation
### State Space
The environment provides comprehensive state information including:
- _Joint positions and velocities_
- Center of mass position and velocity
- Contact forces
- Actuator states

_Note: For all of the reward functions the policy was trained with Joint positions and velocities but the above values can easily be used fro training._
### Reward System
A variety of rewards functions were created and experimented with. Two key reward function that proved quite successful are `stand_reward` identified by key `stand` and `robust_kneeling_reward` identified by `kneeling`. The keys are used in the config or arg-parse to specify the reward functions for training. 
### Training Architecture
- Policy network: Dual 256-unit hidden layers with ReLU activation
- Value network: Matching architecture for stable learning
- The policy and value network are not shared
- PPO-specific parameters optimized for humanoid locomotion
## Project Structure
- Main training implementation: `train_sb3.py`
- Environment definition: `custom_env.py`
- Reward functions: `reward_functions.py`
- Configuration: `config.py`
- Visualization tools: `render_policy.py`, `generate_trajectories.py`
- main file for starting training: `main.py`
# Results

