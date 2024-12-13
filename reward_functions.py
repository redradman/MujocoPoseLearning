import numpy as np
from utils import quaternion_to_euler
"""
MuJoCo Environment Data Structure (env_data) Guide (for the humanoid MJCF model):

1. Position Data (env_data.qpos):
   - Shape: [nq] where nq is the number of position coordinates
   - Index mapping:
     [0:3]   - Root position (x, y, z) of the torso/pelvis
     [3:7]   - Root orientation as quaternion (w, x, y, z)
     [7:]    - Joint angles for each degree of freedom

2. Velocity Data (env_data.qvel):
   - Shape: [nv] where nv is the number of velocity coordinates
   - Index mapping:
     [0:3]   - Linear velocity of the torso (vx, vy, vz)
     [3:6]   - Angular velocity of the torso (wx, wy, wz)
     [6:]    - Joint velocities

3. Control/Action Data (env_data.ctrl):
   - Joint actuator forces/torques
   - Each element corresponds to one actuator

4. Common Access Patterns:
   - Torso/Root Position: env_data.qpos[0:3]
     * x-position = qpos[0]  # Forward/backward
     * y-position = qpos[1]  # Left/right
     * z-position = qpos[2]  # Up/down (height)
   
   - Torso Orientation: env_data.qpos[3:7]
     * Quaternion (w, x, y, z)
     * w = qpos[3]  # Real component
     * x = qpos[4]  # i component
     * y = qpos[5]  # j component
     * z = qpos[6]  # k component
   
   - Linear Velocity: env_data.qvel[0:3]
     * x-velocity = qvel[0]  # Forward/backward velocity
     * y-velocity = qvel[1]  # Left/right velocity
     * z-velocity = qvel[2]  # Up/down velocity
   
   - Angular Velocity: env_data.qvel[3:6]
     * Around x-axis = qvel[3]
     * Around y-axis = qvel[4]
     * Around z-axis = qvel[5]

5. Joint Information:
   - Joint Angles: env_data.qpos[7:]
     * Order follows XML model definition
     * Typically starts from hip joints down to ankle joints
   
   - Joint Velocities: env_data.qvel[6:]
     * Corresponds to joint angle velocities
     * Same order as joint angles

6. Additional Data:
   - Contact Forces: env_data.contact.force
   - External Forces: env_data.xfrc_applied
   - Sensor Data: env_data.sensordata

Note: Exact indices might vary based on the specific humanoid XML model being used.
      Always verify the model structure and DOF counts for your specific case.
"""
#######################################################################################

def robust_kneeling_reward(env_data, params=None):
    """
    After training the agent learn to kneel down (intially intended as a function to reamin standing)
    """
    # Default parameters with recommended values
    default_params = {
        'target_height': 1.282,  # Matching the target height from other reward functions
        'min_height': 0.85,
        'max_roll_pitch': np.pi / 6,  # 30 degrees
        'com_radius': 0.1,  # Maximum allowed horizontal CoM displacement
        'energy_weight': 0.3,
        'posture_weight': 0.3,
        'com_weight': 0.2,
        'foot_weight': 0.1,
        'alive_weight': 0.1
    }
    # Update default parameters with any provided ones
    params = {**default_params, **(params or {})}
    
    # Extract state information
    qpos = env_data.qpos
    qvel = env_data.qvel
    current_height = qpos[2]
    orientation = qpos[3:7]  # quaternion
    time_alive = env_data.time
    # Low height check
    if current_height < params['min_height']:
        reward = current_height**2
        return reward
    # 1. Posture Maintenance Component
    roll, pitch, _ = quaternion_to_euler(orientation)
    orientation_error = (roll ** 2 + pitch ** 2) / (params['max_roll_pitch'] ** 2)
    posture_reward = np.exp(-5.0 * orientation_error)
    
    # Height maintenance
    height_error = np.square(current_height - params['target_height'])
    height_reward = np.exp(-5.0 * height_error)
    
    # Combined posture reward
    posture_score = 0.7 * posture_reward + 0.3 * height_reward
    
    # 2. Center of Mass (CoM) Stability
    com_pos = env_data.subtree_com[0]  # Get CoM position
    com_vel = env_data.subtree_linvel[0]  # Get CoM velocity
    
    # Horizontal distance from center
    com_horizontal_dist = np.sqrt(com_pos[0]**2 + com_pos[1]**2)
    com_stability = np.exp(-10.0 * (com_horizontal_dist / params['com_radius']))
    
    # Penalize rapid CoM movements
    com_vel_penalty = np.exp(-0.1 * np.sum(com_vel**2))
    com_score = 0.7 * com_stability + 0.3 * com_vel_penalty
    
    # 3. Foot Contact Stability (using cfrc_ext instead of contact_force)
    # Get external forces on feet (assuming they're the last bodies)
    left_foot_force = np.sum(np.abs(env_data.cfrc_ext[-2]))
    right_foot_force = np.sum(np.abs(env_data.cfrc_ext[-1]))
    
    # Encourage balanced foot contact
    total_force = left_foot_force + right_foot_force + 1e-8  # Avoid division by zero
    force_symmetry = min(left_foot_force, right_foot_force) / total_force
    foot_balance = force_symmetry
    
    # 4. Energy Efficiency
    # Calculate joint power as product of torque and velocity
    actuator_velocities = qvel[6:]  # Skip root joint velocities
    actuator_forces = env_data.qfrc_actuator[-len(actuator_velocities):]  # Match the size
    joint_power = np.sum(np.square(actuator_forces * actuator_velocities))
    energy_efficiency = np.exp(-0.01 * joint_power)
    
    # 5. Alive Bonus (increases with time, encourages staying upright)
    alive_bonus = 1.0 - np.exp(-0.5 * time_alive)
    
    # Combine all components with weights
    reward = (
        params['posture_weight'] * posture_score +
        params['com_weight'] * com_score +
        params['foot_weight'] * foot_balance +
        params['energy_weight'] * energy_efficiency +
        params['alive_weight'] * alive_bonus
    )

    # print(reward, posture_score, com_score, foot_balance, energy_efficiency, alive_bonus)
    

    # without multiplication
    # max reward is 0.6 
    # min reward is 0.0
    return reward

def stand_reward(env_data, params=None):
    """
    Reward function that encourages stable walking behavior with:
    1. Forward velocity reward
    2. Stability components (height, orientation)
    3. Energy efficiency
    4. Foot alternation
    """
    # Constants
    target_velocity = 1.0  # Desired forward velocity (m/s)
    target_height = 1.282  # Desired torso height
    min_height = 0.8      # Minimum acceptable height
    
    # Extract state variables
    current_height = env_data.qpos[2]
    forward_velocity = env_data.qvel[0]  # x-axis velocity
    orientation = env_data.qpos[3:7]
    roll, pitch, _ = quaternion_to_euler(orientation)
    
    # Get foot contact forces (assuming last two bodies are feet)
    left_foot_force = np.sum(np.abs(env_data.cfrc_ext[-2]))
    right_foot_force = np.sum(np.abs(env_data.cfrc_ext[-1]))
    
    # Early termination for falling
    if current_height < min_height:
        return 0.0
    
    # 1. Velocity Reward: Gaussian around target velocity
    velocity_diff = forward_velocity - target_velocity
    velocity_reward = np.exp(-2.0 * (velocity_diff ** 2))
    
    # 2. Posture Reward
    height_diff = current_height - target_height
    height_reward = np.exp(-2.0 * (height_diff ** 2))
    orientation_reward = np.exp(-3.0 * (roll**2 + pitch**2))
    posture_reward = 0.5 * height_reward + 0.5 * orientation_reward
    
    # 3. Energy Efficiency: penalize excessive joint torques
    torque_penalty = np.exp(-0.05 * np.sum(np.square(env_data.ctrl)))
    
    # 4. Foot Alternation Reward
    total_force = left_foot_force + right_foot_force + 1e-8
    force_symmetry = min(left_foot_force, right_foot_force) / total_force
    foot_reward = 1.0 - force_symmetry  # Reward alternating foot contacts
    
    # Combine rewards with weights
    reward = (0.4 * velocity_reward +      # Primary objective
             0.3 * posture_reward +        # Stability
             0.2 * foot_reward +           # Walking pattern
             0.1 * torque_penalty)         # Efficiency
    
    # Save state if needed
    if params is not None:
        params["previous_qpos"] = env_data.qpos.copy()
    
    return reward

def walk_reward(env_data, params=None):
    """
    Reward function that encourages stable walking behavior with:
    1. Forward velocity reward
    2. Stability components (height, orientation)
    3. Energy efficiency
    4. Foot alternation
    """
    # Constants
    target_velocity = 10.0  # Desired forward velocity (m/s)
    target_height = 1.282  # Desired torso height
    min_height = 0.8      # Minimum acceptable height
    # Extract state variables
    current_height = env_data.qpos[2]
    forward_velocity = env_data.qvel[0]  # x-axis velocity
    orientation = env_data.qpos[3:7]
    roll, pitch, _ = quaternion_to_euler(orientation)
    
    # Get foot contact forces (assuming last two bodies are feet)
    
    # Early termination for falling
    if current_height < min_height:
        return 0.1 * current_height / min_height
    
    # 1. Velocity Reward: Gaussian around target velocity
    velocity_diff = forward_velocity - target_velocity
    velocity_reward = np.exp(-0.5 * (velocity_diff ** 2))
    
    # 2. Posture Reward
    height_diff = current_height - target_height
    height_reward = np.exp(-2.0 * (height_diff ** 2))
    orientation_reward = np.exp(-3.0 * (roll**2 + pitch**2))
    posture_reward = 0.5 * height_reward + 0.5 * orientation_reward
    
    # 3. Energy Efficiency: penalize excessive joint torques
    torque_penalty = np.exp(-0.05 * np.sum(np.square(env_data.ctrl)))
    # print(velocity_reward, posture_reward, torque_penalty)
    
    # Combine rewards with weights
    reward = (velocity_reward +      # Primary objective
            #  0.3 * posture_reward +        # Stability
            #  0.2 * torque_penalty)         # Efficiency
            posture_reward * torque_penalty)
    
    # Save state if needed
    if params is not None:
        params["previous_qpos"] = env_data.qpos.copy()
    
    return reward

# Dictionary mapping reward names to functions
REWARD_FUNCTIONS = {
    'default': stand_reward,
    'kneeling': robust_kneeling_reward,
    'stand': stand_reward,
    'walk': walk_reward
}

