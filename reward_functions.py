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
    roll_pitch_weight = 2.0 # Weight in exponent for orientation penalty
    height_weight = 1.0     # Weight in exponent for height deviation
    angvel_weight = 0.5     # Weight in exponent for angular velocity

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
    normalized_reward = (height_reward +
                         upright_reward *
                         angular_velocity_reward + 
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
    if current_height < 1.0:
        return np.log(current_height)
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

    if current_height < 0.8 or current_height > 1.7:
        epsilon = 1e-8
        return epsilon + 0.1 * (current_height / 0.8)

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

def humanoid_gym_reward(env_data, params=None):
    """
    Implements a reward function similar to Gymnasium's Humanoid environment.
    """
    # Parameters
    forward_reward_weight = 1.25
    ctrl_cost_weight = 0.1
    contact_cost_weight = 5e-7
    contact_cost_range = (-np.inf, 10.0)
    healthy_reward = 5.0
    healthy_z_range = (1.0, 2.0)

    # Extract State Variables
    current_height = env_data.qpos[2]
    forward_vel = env_data.qvel[0]  # x-axis velocity
    
    # Compute Rewards/Costs
    is_healthy = healthy_z_range[0] < current_height < healthy_z_range[1]
    healthy_reward_value = healthy_reward if is_healthy else 0.0
    
    forward_reward = forward_reward_weight * forward_vel
    
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(env_data.ctrl))
    
    # Fixed contact cost calculation
    contact_forces = []
    for i in range(env_data.ncon):
        contact_forces.append(env_data.contact[i].dist)
    contact_forces = np.array(contact_forces)
    contact_cost = contact_cost_weight * np.sum(np.square(contact_forces))
    contact_cost = np.clip(contact_cost, contact_cost_range[0], contact_cost_range[1])

    # Combine Rewards
    reward = healthy_reward_value + forward_reward - ctrl_cost - contact_cost
    
    if params is not None:
        params["previous_qpos"] = env_data.qpos.copy()
    
    return reward

def improved_standing_reward(env_data, params=None):
    """
    Reward function with stronger emphasis on time alive and progressive difficulty,
    aligned with lenient truncation at height < 0.3
    """
    # Core parameters
    target_height = 1.282
    height_weight = 1.0
    orientation_weight = 0.5
    
    # Extract state
    current_height = env_data.qpos[2]
    orientation = env_data.qpos[3:7]
    roll, pitch, _ = quaternion_to_euler(orientation)
    time_alive = env_data.time
    
    # Severe height penalty only when very low (matching truncation threshold)
    if current_height < 0.3:
        return 0.2 * (current_height / 0.3)  # Matches truncation scaling
    
    # Height-based reward scaling
    height_diff = current_height - target_height
    height_reward = np.exp(-height_weight * (height_diff ** 2))
    
    # Orientation reward with more tolerance
    orientation_reward = np.exp(-orientation_weight * (roll**2 + pitch**2))
    
    # Time reward with exponential growth to encourage longer standing
    time_reward = 1.0 - np.exp(-0.2 * time_alive)  # Faster growth
    
    # Progressive weighting that heavily favors time after initial stability
    time_weight = min(0.9 * (1.0 - np.exp(-0.5 * time_alive)), 0.9)  # Max 90% weight for time
    posture_weight = 1.0 - time_weight
    
    # Combine rewards with emphasis on time alive
    reward = (posture_weight * (0.6 * height_reward + 0.4 * orientation_reward) +
             time_weight * time_reward)
    
    # Larger survival bonus to encourage longer episodes
    survival_bonus = 0.2 * (time_alive / 10.0)  # Scales up to 0.2 over 10 seconds
    reward += survival_bonus
    
    return reward

def standing_time_reward(env_data, params=None):
    # Default parameters
    # target_height = params.get('target_height', 1.2) if params else 1.2
    # min_height = params.get('min_height', 0.8) if params else 0.8
    # height_weight = params.get('height_weight', 5.0) if params else 5.0
    # orientation_weight = params.get('orientation_weight', 5.0) if params else 5.0
    # time_weight = params.get('time_weight', 0.1) if params else 0.1

    # # Extract state
    current_height = env_data.qpos[2]
    orientation = env_data.qpos[3:7]
    roll, pitch, _ = quaternion_to_euler(orientation)
    time_alive = env_data.time

    # # Orientation reward (value between 0 and 1)
    orientation_error = (roll ** 2 + pitch ** 2)
    orientation_reward = 1 / np.exp(5 * orientation_error)

    crtl = 1 / np.exp(np.square(env_data.ctrl).sum())
    healthy_reward = 1/ np.exp((current_height - 1.282)**2) * time_alive
    reward = healthy_reward + crtl + orientation_reward
    if current_height < 0.85:
        reward *= 0.2
        return reward
    return reward

def robust_standing_reward(env_data, params=None):
    """
    A comprehensive reward function for robust humanoid standing based on multiple stability metrics.
    Incorporates posture maintenance, CoM stability, foot placement, energy efficiency, and disturbance rejection.
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

def simple_standing_reward(env_data, params=None):
    """
    A simple reward function focused on maintaining upright posture and height.
    Rewards:
    1. Being at the right height
    2. Staying upright
    3. Minimizing movement
    """
    # Constants
    target_height = 1.282
    min_height = 0.9
    
    # Extract state
    current_height = env_data.qpos[2]
    orientation = env_data.qpos[3:7]
    roll, pitch, _ = quaternion_to_euler(orientation)
    
    # Early termination with low reward if too low
    if current_height < min_height:
        return current_height**5
    
    # Height reward (1.0 at target height, decreasing as we move away)
    height_diff = current_height - target_height
    height_reward = np.exp(-2.0 * (height_diff ** 2))
    
    # Orientation reward (1.0 when upright, decreasing as we tilt)
    orientation_reward = np.exp(-3.0 * (roll ** 2 + pitch ** 2))
    
    # Movement penalty (1.0 when still, decreasing with movement)
    velocity = env_data.qvel[0:6]  # Linear and angular velocity of torso
    movement_reward = np.exp(-0.1 * np.sum(velocity ** 2))
    
    # Combine rewards (weighted sum)
    reward = (0.5 * height_reward + 
             0.25 * orientation_reward + 
             0.25 * movement_reward)

    # reward = height_reward * 5
    
    return reward

# Dictionary mapping reward names to functions
REWARD_FUNCTIONS = {
    'default': humanoid_standing_reward,
    'stand': humanoid_standing_reward,
    'stand_additive': humanoid_balanced_standing_reward,
    'walk': humanoid_walking_reward,
    'gym': humanoid_gym_reward,
    'another_stand': improved_standing_reward,
    'standing_time': standing_time_reward,
    'robust_stand': robust_standing_reward,
    'simple_stand': simple_standing_reward
}
