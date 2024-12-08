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

def humanoid_standing_reward(env_data, params=None):
    """
    Computes a normalized, non-negative reward for encouraging the humanoid to stand 
    steadily for over 30 seconds. The reward factors in height, orientation, angular 
    stability, and how long the humanoid has been standing. 

    Returns:
        normalized_reward (float): A value in [0, 1], where 0 means poor standing performance 
                                   and 1 means ideal long-term standing stability.
    """

    # -----------------------
    # Parameters & References
    # -----------------------
    target_height = 1.282   # Desired standing height
    max_time = 30            # Time threshold for full time-based reward
    roll_pitch_weight = 1.0 # Weight in exponent for orientation penalty
    height_weight = 2.0     # Weight in exponent for height deviation
    angvel_weight = 1.0     # Weight in exponent for angular velocity

    # -----------------------
    # Extract State Variables
    # -----------------------
    current_height = env_data.qpos[2]

    # Orientation (roll, pitch, yaw) from quaternion
    orientation = env_data.qpos[3:7]
    roll, pitch, _ = quaternion_to_euler(orientation)

    # Angular velocity of the torso
    angular_velocity = env_data.qvel[3:6]
    ang_vel_norm = np.linalg.norm(angular_velocity)

    # Time alive in the simulation
    time_alive = env_data.time

    # -----------------------
    # Compute Sub-Rewards
    # -----------------------

    # 1. Height Reward: Peak at target height, decreases with deviation
    height_diff = current_height - target_height
    # Using an exponential decay: reward = exp(-height_weight * height_diff^2)
    height_reward = np.exp(-height_weight * (height_diff ** 2))

    # 2. Orientation Reward: Favor small roll/pitch
    # Orientation deviation: roll^2 + pitch^2
    upright_reward = np.exp(-roll_pitch_weight * (roll**2 + pitch**2))

    # 3. Angular Velocity Reward: Favor minimal rotational movement
    angular_velocity_reward = np.exp(-angvel_weight * ang_vel_norm)

    # 4. Time Standing Reward: 
    # Linearly scale from 0 at t=0s to 1 at t=30s (or above)
    time_standing_reward = min(time_alive / max_time, 1.0)

    # -----------------------
    # Combine and Normalize
    # -----------------------
    # Multiply sub-rewards. Each is in [0,1], so the product will also be in [0,1].
    # This ensures overall normalization without needing division by a max value.
    normalized_reward = (height_reward * 
                         upright_reward * 
                         angular_velocity_reward * 
                         time_standing_reward)

    # Save state for potential future use (e.g., smoothness calculation)
    if params is not None:
        params["previous_qpos"] = env_data.qpos.copy()

    # -----------------------
    # Return the Final Reward
    # -----------------------
    # Bounds:
    # lower_bound = 0.0
    # upper_bound = 1.0
    return normalized_reward

def humanoid_balanced_standing_reward(env_data, params=None):
    """
    Computes a balanced, additive reward for humanoid standing that encourages good height, 
    orientation, minimized angular velocity, natural posture, low torque usage, and 
    prolonged standing. The rewards are summed (with weights) and then scaled back into [0,1].
    """
    # -----------------------
    # Parameters & References
    # -----------------------
    target_height = 1.282
    max_time = 30.0
    roll_pitch_weight = 2.0
    height_weight = 2.0
    angvel_weight = 1.0
    # posture_weight = 0.5
    torque_weight = 0.001

    # Weighted combination parameters
    # Adjust these weights to emphasize different aspects of the standing task.
    w_height = 1.0
    w_orientation = 1.0
    w_angvel = 1.0
    w_time = 1.0
    w_posture = 1.0
    w_torque = 1.0

    # # Indices for arm joints or relevant posture joints (example indices)
    # arm_joint_indices = [10, 11, 12, 13]
    # arm_target_angles = np.array([0.0, 0.0, 0.0, 0.0])  # desired neutral angles

    # -----------------------
    # Extract State Variables
    # -----------------------
    current_height = env_data.qpos[2]

    # Orientation
    orientation = env_data.qpos[3:7]
    roll, pitch, yaw = quaternion_to_euler(orientation)

    # Angular velocity
    angular_velocity = env_data.qvel[3:6]
    ang_vel_norm = np.linalg.norm(angular_velocity)

    # Time alive
    time_alive = env_data.time

    # Joint angles
    # joint_angles = env_data.qpos[7:]
    # arm_angles = joint_angles[arm_joint_indices]

    # Control torques
    torques = env_data.ctrl

    # -----------------------
    # Compute Sub-Rewards [0,1]
    # -----------------------

    # Height Reward
    height_diff = current_height - target_height
    height_reward = np.exp(-height_weight * (height_diff ** 2))

    # Orientation Reward (penalize large roll/pitch)
    orientation_reward = np.exp(-roll_pitch_weight * (roll**2 + pitch**2))

    # Angular Velocity Reward
    angular_velocity_reward = np.exp(-angvel_weight * ang_vel_norm)

    # Time Standing Reward (0 to 1 as time goes from 0 to 30s)
    time_standing_reward = min(time_alive / max_time, 1.0)

    # Posture Reward (penalize deviation from target arm angles)
    # arm_deviation = np.linalg.norm(arm_angles - arm_target_angles)
    # posture_reward = np.exp(-posture_weight * arm_deviation)

    # Torque Reward (less torque = better)
    torque_norm = np.sum(np.abs(torques))
    torque_reward = np.exp(-torque_weight * torque_norm)

    # -----------------------
    # Combine Additively and Scale
    # -----------------------
    # Weighted sum of rewards:
    weighted_sum = (w_height * height_reward +
                    w_orientation * orientation_reward +
                    w_angvel * angular_velocity_reward +
                    w_time * time_standing_reward +
                    # w_posture * posture_reward +
                    w_torque * torque_reward)

    total_weight = (w_height + w_orientation + w_angvel + w_time + w_posture + w_torque)
    
    # Normalize the weighted sum to keep it in [0, 1]
    # Since each sub-component is already in [0,1], dividing by total_weight ensures we get a value ≤ 1.
    normalized_reward = weighted_sum / total_weight

    # Save state for potential future use
    if params is not None:
        params["previous_qpos"] = env_data.qpos.copy()

    return normalized_reward


def quaternion_to_euler(quat):
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat

    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def humanoid_walking_reward(env_data, params=None):
    """
    Computes a multiplicative ("AND"-style) reward to encourage a humanoid to walk forward at a desired speed 
    while maintaining good posture, appropriate height, and efficient use of torque. Each component must be 
    good for the overall reward to remain high.
    """

    # -----------------------
    # Parameters
    # -----------------------
    target_forward_speed = 1.0    # Target forward speed (m/s)
    height_target = 1.3           # Desired torso height
    roll_pitch_weight = 2.0       # Penalize non-upright posture more heavily
    height_weight = 2.0           # Penalize deviation from target height
    torque_weight = 0.001         # Weight for torque penalty

    # Extract State Variables
    forward_vel = env_data.qvel[0]
    current_height = env_data.qpos[2]

    # Orientation (roll, pitch, yaw)
    orientation = env_data.qpos[3:7]
    roll, pitch, yaw = quaternion_to_euler(orientation)

    # Torques
    torques = env_data.ctrl

    # -----------------------
    # Compute Sub-Rewards (each in [0,1])
    # -----------------------

    # Forward velocity: target speed = 1.0 m/s
    # Use a Gaussian-like reward for hitting target speed
    vel_diff = forward_vel - target_forward_speed
    forward_vel_reward = np.exp(-(vel_diff ** 2))

    # Posture: penalize large roll and pitch
    posture_reward = np.exp(-roll_pitch_weight * (roll**2 + pitch**2))

    # Height: penalize deviation from target height
    height_diff = current_height - height_target
    height_reward = np.exp(-height_weight * (height_diff ** 2))

    # Torque: less torque is better
    torque_norm = np.sum(np.abs(torques))
    torque_reward = np.exp(-torque_weight * torque_norm)

    # -----------------------
    # Multiplicative Combination
    # -----------------------
    # If any component is low, the product is low, enforcing that all conditions must be met.
    reward = (forward_vel_reward *
              posture_reward *
              height_reward *
              torque_reward)

    # Save state if needed
    if params is not None:
        params["previous_qpos"] = env_data.qpos.copy()

    return reward

# Dictionary mapping reward names to functions
REWARD_FUNCTIONS = {
    'default': humanoid_standing_reward,
    'stand': humanoid_standing_reward,
    'stand_additive': humanoid_balanced_standing_reward,
    'walk': humanoid_walking_reward
}
