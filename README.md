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
### 3. Flexible Configuration Options
- Command-line interface for customizing training parameters:
  - **Environment Parameters:**
    - `--total_timesteps`: Total timesteps for training (default: 20M)
    - `--render_interval`: Interval between video recordings (default: 2500)
    - `--n_envs`: Number of parallel environments (default: 8)
    - `--reward_function`: Type of reward function to use (default: "walk")
    - `--frame_skip`: Number of frames to skip (default: 3)
    - `--framerate`: Framerate for rendering (default: 60)
  
  - **PPO Hyperparameters:**
    - `--learning_rate`: Learning rate (default: 1e-4)
    - `--n_steps`: Number of steps per update (default: 2048)
    - `--batch_size`: Minibatch size (default: 128)
    - `--n_epochs`: Number of epochs (default: 20)
    - `--gamma`: Discount factor (default: 0.99)
    - `--gae_lambda`: GAE lambda parameter (default: 0.95)
    - `--clip_range`: Clipping parameter (default: 0.2)
    - `--ent_coef`: Entropy coefficient (default: 0.0)
  
  - **Network Architecture:**
    - `--net_arch_pi`: Policy network architecture (default: [64, 64])
    - `--net_arch_vf`: Value function network architecture (default: [64, 64])
  
  - **Configuration File:**
    - `--config`: Path to config file for loading predefined settings

Example usage:

### 4. Visualization and Analysis Tools
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
The agent has been trained with the observation of position and velocity of the joints (without any additional data). Furthermore, unlike the humanoid-v5 environment in gymnasium, there is no restriction placed on the action space (gymnasium clips the value from -0.4 to 0.4 but this environment does not clip the action space). The agent is free to exercises maximum torque in the joints by sending maximum value of the control signal. However, in the both results obtained below the agent is being penalized for the energy expenditure. Therefore, the policy has to learn the optimal way to move the body to acheive the desired pose while minimizing the energy expenditure. 

Additionally, the starting position and the intial velocity of the humanoid is being purturbed to test the robustness of the policy during the training to make improve the generalization of the policy 

The key results that were acheived are the standing and kneeling poses. For the hyperparams, take a look at the `config.py` file.
****
## Standing Reward Function Results

| Video | Description |
|--------------|-------------|
| ![Standing_episode_5000](https://github.com/user-attachments/assets/a54adad1-6fcd-48fa-82e8-cfd6492a0c5a) | **Episode 5,000**: Humanoid is able to apply some torque to maintain the standing position for a short time but quickly loses balance |
| ![Standing_episode_10000](https://github.com/user-attachments/assets/c484b76a-70a2-40a7-bdde-001cf3e6eda0) | **Episode 10,000**: Humanoid has learned to stand for a longer duration but still struggles with balance |
| ![Standing_episode_20000](https://github.com/user-attachments/assets/9fe72f3c-4d95-4831-82eb-b53e979ccb13) | **Episode 20,000**: Humanoid is standing for even longer duration and can maintain a slight amount of balance before falling |
| ![Standing_episode_25000](https://github.com/user-attachments/assets/58ba04cd-b97e-4009-a561-7db00e216209) | **Episode 25,000**: Humanoid has learned to use its hands and torso to maintain the standing position |
| ![Standing Result](https://github.com/user-attachments/assets/b0977aa8-e676-445c-aabd-ba82e07c0198) | **Result (trained policy)**: More stable actions compared to episode 25,000 and the humanoid has learned to use its hand torso to maintain balance **while reducing the necessary torque needed for maintaining balance** |

The standing reward function successfully achieved its primary objective of maintaining a stable upright posture. The agent learned to:
- Maintain target height of 1.282m
- Keep balanced orientation with minimal deviation
- Efficiently use joint torques
- Distribute weight evenly between feet
## Kneeling Reward Function Results
| Video | Description |
|--------------|-------------|
| ![Kneeling_episode_1000](https://github.com/user-attachments/assets/a2ff63b6-5f2c-4c4a-9e37-7724c7853f14) | **Episode 1,000**: Humanoid has learned to bend its knees and land on them but quickly loses balance |
| ![Kneeling_episode_5000](https://github.com/user-attachments/assets/af8aeb74-92de-4297-a775-e1703434209f) | **Episode 5,000**: Humanoid has achieved the ability to bend its knees and have some balance. It quickly loses the balance but has learned to compensate for it and it is using its hands to prevent falling to the ground |
| ![Kneeling_episode_10000](https://github.com/user-attachments/assets/9a026161-527e-4101-a72e-16560bd648fe) | **Episode 10,000**: Humanoid has learned to use its torso to maintain balance but it has not stabilized this learning and has not learn to prevent falling from the back |
| ![Kneeling_episode_15000](https://github.com/user-attachments/assets/fd37038c-f5e0-4a75-8e2e-4fa78965013e) | **Episode 15,000**: Relatively stable results but with some jerky motions but the agent has learned to maintain position and does not fall |
| ![Kneeling_episode_20000](https://github.com/user-attachments/assets/1af0ab3b-6df5-472e-ab6f-24fd2102bfc3)| **Episode 20,000**: Slightly less jerky motions and stable results suggesting policy convergence |
| ![Kneeling Result](https://github.com/user-attachments/assets/b3ccf65e-53e0-4e05-b0c5-65dd00b792de) | **Result (trained policy)**: Learned to kneel fast and maintain the position and keep it stable without falling. Jerky motions are reduced in hands but still present in the legs |

The kneeling reward function produced an unexpected but interesting result. While originally designed for standing, the agent discovered a stable kneeling posture that:
- Minimizes energy expenditure
- Maintains stable orientation
- Achieves good balance
- Effectively distributes contact forces

Both reward functions demonstrate the ability of the PPO algorithm to find stable solutions, even if they're not the initially intended ones. The kneeling behavior emerged as a locally optimal solution that satisfied the reward criteria in an unexpected way.

