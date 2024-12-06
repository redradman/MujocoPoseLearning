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

def humaniod_walking_reward(env_data, params=None):
    """
    Reward function for humanoid walking that encourages:
    - Maintaining appropriate height
    - Avoiding extreme joint angles
    - Moving forward
    - Staying alive (not falling)
    
    Returns:
    - reward: float between 0 and 1
    """
    # Extract relevant state information from env_data
    # Position data
    root_pos = env_data.qpos[0:3]      # (x, y, z) position of torso
    root_orient = env_data.qpos[3:7]    # (w, x, y, z) quaternion orientation of torso
    joint_angles = env_data.qpos[7:]    # Array of joint angles

    # Velocity data
    root_lin_vel = env_data.qvel[0:3]   # (vx, vy, vz) linear velocity of torso
    root_ang_vel = env_data.qvel[3:6]   # (wx, wy, wz) angular velocity of torso
    joint_vel = env_data.qvel[6:]       # Array of joint velocities

    # Unpack commonly used values
    x_pos, y_pos, z_pos = root_pos      # Position components
    vx, vy, vz = root_lin_vel          # Linear velocity components
    wx, wy, wz = root_ang_vel          # Angular velocity components
    
    ctrl_cost_weight = 1.25
    control_cost = ctrl_cost_weight * np.sum(np.square(env_data.ctrl))

    healthy_reward = 5.0


    # Use z_pos instead of directly accessing env_data
    # height_reward = 5/np.exp(np.square(z_pos - 1))

    velocity_reward = vx * 0.02

    angular_velocity_reward = 5/np.exp(np.sum(np.square(root_ang_vel)))
    
    # Use vx instead of directly accessing env_data
    
    # reward = height_reward * angular_velocity_reward * velocity_reward
    reward = velocity_reward + healthy_reward + angular_velocity_reward - control_cost
    # print(reward, velocity_reward, healthy_reward, height_reward, angular_velocity_reward, control_cost)
    return reward


    ctrl_cost_weight = 0.05
    control_cost = ctrl_cost_weight * np.sum(np.square(env_data.ctrl))
    
    # Height reward - peaks at z=1.0, falls off exponentially
    target_height = 1.0
    height_reward = 3.0 * np.exp(-5.0 * np.square(z_pos - target_height))
    
    # Forward velocity reward - scaled by height reward to encourage upright movement
    velocity_scale = height_reward * 0.5  # Only get full velocity reward when upright
    velocity_reward = velocity_scale * vx
    
    # Stability reward - rewards low angular velocities and vertical movement
    stability_reward = 2.0 * np.exp(-2.0 * (np.sum(np.square(root_ang_vel)) + np.square(vy) + np.square(vz)))
    
    # Alive bonus (small constant reward)
    alive_bonus = 0.5
    
    # Total reward
    reward = velocity_reward + height_reward + stability_reward + alive_bonus - control_cost

# Dictionary mapping reward names to functions
REWARD_FUNCTIONS = {
    'default': humaniod_walking_reward,
    'walk': humaniod_walking_reward  # Add this line
}