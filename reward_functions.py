import numpy as np

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
########################################################################################
def default_reward(env_data, params=None):
    """Original reward function focusing on balanced walking."""
    height = env_data.qpos[2]
    forward_vel = env_data.qvel[0]
    orientation = env_data.qpos[3:7]
    # joint_angles = env_data.qpos[7:]
    # joint_velocities = env_data.qvel[6:]
    # ctrl = env_data.ctrl
    
    # Early termination
    if height < 0.7:
        return -1.0
        
    # Base reward for staying alive
    reward = 0.1
    
    # Forward progress
    reward += np.clip(forward_vel, 0, 1.0) * 0.2
    
    # Stability (using orientation)
    euler = quaternion_to_euler(orientation)
    roll, pitch = euler[0], euler[1]
    orientation_penalty = np.square(roll) + np.square(pitch)
    reward += 0.3 * np.exp(-3.0 * orientation_penalty)
    
    # Height maintenance
    height_diff = abs(height - 1.3)
    reward += 0.2 * np.exp(-2.0 * height_diff)
    
    return float(reward)
########################################################################################
def forward_only(env_data, params=None):
    """Simple reward based only on forward velocity."""
    height = env_data.qpos[2]
    forward_vel = env_data.qvel[0]
    
    if height < 0.7:
        return -1.0
    
    return float(np.clip(forward_vel, 0, 1.0))
########################################################################################
def stability_focused(env_data, params=None):
    """Reward function prioritizing stability and balance."""
    height = env_data.qpos[2]
    orientation = env_data.qpos[3:7]
    joint_velocities = env_data.qvel[6:]
    
    euler = quaternion_to_euler(orientation)
    roll, pitch = euler[0], euler[1]
    
    if height < 0.7 or abs(roll) > 0.8 or abs(pitch) > 0.8:
        return -1.0
    
    orientation_score = 1.0 - (abs(roll) + abs(pitch)) / 1.6
    height_score = np.exp(-2.0 * abs(height - 1.3))
    velocity_penalty = np.sum(np.square(joint_velocities)) * 0.01
    
    reward = (
        0.5 * orientation_score +
        0.3 * height_score -
        0.2 * velocity_penalty
    )
    
    return float(reward)
########################################################################################
def energy_efficient(env_data, params=None):
    """Reward function focusing on energy efficiency."""
    height = env_data.qpos[2]
    forward_vel = env_data.qvel[0]
    ctrl = env_data.ctrl
    
    if height < 0.7:
        return -1.0
    
    energy_cost = np.sum(np.square(ctrl)) * 0.1
    progress_reward = np.clip(forward_vel, 0, 1.0) * 0.1
    
    reward = progress_reward - energy_cost
    
    if height > 1.0 and forward_vel > 0:
        reward += 0.1
    
    return float(reward)

########################################################################################
def just_keep_standing_reward(env_data, params=None):
    """Reward function that prioritizes staying upright for longer periods."""
    height = env_data.qpos[2]
    orientation = env_data.qpos[3:7]
    
    # Get roll and pitch to check orientation
    euler = quaternion_to_euler(orientation)
    roll, pitch = euler[0], euler[1]
    
    # Early termination conditions
    if (height < 0.7 or  # Too low
        abs(roll) > 1.0 or  # Too tilted (side)
        abs(pitch) > 1.0):  # Too tilted (front/back)
        return -1.0
    
    # Base survival reward
    reward = 1.0
    
    # Small penalties for deviation from ideal posture
    posture_penalty = (abs(roll) + abs(pitch)) * 0.1
    height_penalty = abs(height - 1.3) * 0.1
    
    # Final reward: base reward minus small penalties to encourage good posture
    reward = reward - posture_penalty - height_penalty
    
    return float(reward)

####################### Helper function used by reward functions #######################
########################################################################################
def quaternion_to_euler(quat):
    """Convert quaternion to euler angles."""
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

# def comprehensive_walking_reward(env_data, params=None):
#     """Advanced reward function combining multiple components for natural walking."""
#     # Extract state information
#     height = env_data.qpos[2]  # z-position (height)
#     orientation = env_data.qpos[3:7]  # quaternion
#     forward_vel = env_data.qvel[0]  # forward velocity
#     lateral_vel = env_data.qvel[1]  # side velocity
#     ctrl = env_data.ctrl  # control actions
    
#     # Get orientation in euler angles
#     euler = quaternion_to_euler(orientation)
#     roll, pitch, _ = euler
    
#     # 1. Early termination with smaller penalty
#     if height < 0.8 and forward_vel < 0.2:
#         return -1.0
    
#     reward = 0.0
    
#     # 2. Stability First (Primary Reward)
#     orientation_score = 1 / (1 + 5 * (np.square(roll) + np.square(pitch)))
#     reward += 0.2 * orientation_score
    
#     # Height maintenance
#     target_height = 1.3
#     height_score = np.exp(-3.0 * abs(height - target_height)) / np.exp(-3.0)
#     reward += 0.2 * height_score
    
#     # 3. Forward Progress (Only if stable)
#     stability_factor = min(orientation_score, height_score)
#     forward_reward = 0.2 * np.tanh(forward_vel)
#     reward += forward_reward * stability_factor
    
#     # 4. Energy Efficiency
#     ctrl_cost = np.sum(np.square(ctrl)) * 0.001
#     reward -= ctrl_cost
    
#     # 5. Natural Motion Penalties
#     lateral_penalty = 0.01 * max(0, abs(lateral_vel) - 0.1)
#     reward -= lateral_penalty
    
#     joint_velocities = env_data.qvel[6:]
#     joint_penalty = np.sum(np.square(joint_velocities[joint_velocities > 1.0])) * 0.0001
#     reward -= joint_penalty

#     actuator_usage_penalty = 0.001 * (np.sum(np.square(ctrl)) + np.sum(abs(ctrl)))
#     reward -= actuator_usage_penalty

#     # Fall penalty
#     if height < 0.8:
#         reward -= 0.5
    
#     # Reward Clipping
#     reward = np.clip(reward, -1.0, 1.0)
    
#     # Dynamic Scaling (example: scale rewards based on training progress)
#     # if params and 'training_progress' in params:
#     #     progress = params['training_progress']
#     #     scaling_factor = 0.5 + 0.5 * progress  # Scale from 0.5 to 1.0
#     #     reward *= scaling_factor
    
#     return float(reward)

def mujoco_style_walking_reward(env_data, params=None):
    """MuJoCo-style walking reward, normalized to [0, 1].
    Combines forward velocity reward with a 'healthy' state bonus,
    and includes penalties for falling, energy usage, and joint angles."""
    
    # Extract state information
    height = env_data.qpos[2]         # z-position
    orientation = env_data.qpos[3:7]  # quaternion
    forward_vel = env_data.qvel[0]    # x-velocity
    ctrl = env_data.ctrl              # control signals
    
    # Convert quaternion to euler angles
    euler = quaternion_to_euler(orientation)
    roll, pitch = euler[0], euler[1]
    
    # Early termination
    if height < 0.8:  # fallen
        return 0.0
        
    # 1. Alive bonus (0.1)
    alive_bonus = 0.1
    
    # 2. Forward velocity reward (0 to 0.4)
    # Reward is maximum at target_velocity
    target_velocity = 1.5
    velocity_reward = 0.4 * np.clip(forward_vel / target_velocity, 0.0, 1.0)
    
    # 3. Posture reward (0 to 0.3)
    # Penalize non-neutral orientation and height deviation
    target_height = 1.3
    orientation_cost = (roll**2 + pitch**2)
    height_deviation = abs(height - target_height)
    posture_reward = 0.3 * np.exp(-3.0 * (orientation_cost + height_deviation))
    
    # 4. Energy penalty (0 to 0.2 penalty)
    # Penalize excessive action/control values
    energy_penalty = 0.2 * np.clip(np.sum(np.square(ctrl)) / len(ctrl), 0.0, 1.0)
    
    # Combine all components
    reward = alive_bonus + velocity_reward + posture_reward - energy_penalty
    
    # Normalize to [0, 1]
    return float(np.clip(reward, 0.0, 1.0))

def mujoco_humanoid_reward(env_data, params=None):
    """
    Canonical MuJoCo humanoid reward function, normalized to [0, 1].
    Reference: https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid_v4.py
    """
    
    # Extract state information
    qpos = env_data.qpos
    qvel = env_data.qvel
    ctrl = env_data.ctrl
    
    # Constants from MuJoCo
    ALIVE_BONUS = 5.0
    FORWARD_WEIGHT = 1.25
    CTRL_COST_WEIGHT = 0.1
    CONTACT_COST_WEIGHT = 5e-7
    CONTACT_COST_RANGE = [0, 10]
    
    # 1. Alive bonus
    alive_bonus = ALIVE_BONUS
    
    # 2. Forward velocity reward (x-axis)
    forward_reward = FORWARD_WEIGHT * qvel[0]
    
    # 3. Control cost (squared L2 norm of control signals)
    ctrl_cost = CTRL_COST_WEIGHT * np.sum(np.square(ctrl))
    
    # 4. Contact cost (based on contact forces)
    contact_cost = 0
    # if hasattr(env_data, 'cfrc_ext'):
    #     contact_cost = CONTACT_COST_WEIGHT * np.sum(
    #         np.square(env_data.cfrc_ext))
    # contact_cost = np.clip(contact_cost, CONTACT_COST_RANGE[0], 
    #                       CONTACT_COST_RANGE[1])
    
    # Combine rewards and costs
    reward = alive_bonus + forward_reward - ctrl_cost - contact_cost
    
    # Normalize to [0, 1] based on typical ranges
    # Note: These scaling factors are based on empirical observations
    MAX_EXPECTED_REWARD = 20.0  # Typical maximum reward value
    MIN_EXPECTED_REWARD = -5.0  # Typical minimum reward value
    
    normalized_reward = (reward - MIN_EXPECTED_REWARD) / \
                       (MAX_EXPECTED_REWARD - MIN_EXPECTED_REWARD)
    
    return float(np.clip(normalized_reward, 0.0, 1.0))

# Dictionary mapping reward names to functions
REWARD_FUNCTIONS = {
    'default': default_reward,
    'forward_only': forward_only,
    'stability': stability_focused,
    'energy_efficient': energy_efficient,
    'stand_still': just_keep_standing_reward,
    # 'walking': comprehensive_walking_reward,
    'simple_walking': mujoco_style_walking_reward,
    'mujoco_humanoid': mujoco_humanoid_reward
} 