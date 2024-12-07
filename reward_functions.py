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
    - Moving forward
    - Minimizing control effort
    - Maintaining stable orientation

    Returns:
    - reward: float
    """
    # Desired height
    desired_z = 1.0
    z_pos = env_data.qpos[2]
    height_reward = np.exp(-0.5 * (z_pos - desired_z)**2)

    # Forward velocity
    x_vel = env_data.qvel[0]
    velocity_reward = x_vel * 0.5  # Scale appropriately

    # Control effort
    control_cost_weight = 0.01
    control_cost = control_cost_weight * np.sum(np.square(env_data.ctrl))

    # Add smoothness penalty to discourage rapid changes
    action_smoothness_penalty = 0.005 * np.sum(np.square(np.diff(env_data.ctrl)))

    # Orientation stability (e.g., keeping pitch and roll near zero)
    orientation = env_data.qpos[3:7]
    euler = quaternion_to_euler(orientation)
    roll, pitch = euler[0], euler[1]
    orientation_penalty = - (np.abs(roll) + np.abs(pitch)) * 0.1  # Adjust weight as needed

    # Combine rewards
    reward = height_reward + velocity_reward + orientation_penalty - control_cost - action_smoothness_penalty
    # print(reward, height_reward, velocity_reward, orientation_penalty, control_cost) # for debugging
    return reward

def quaternion_to_euler(quat):
    """Convert quaternion to euler angles."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

# Dictionary mapping reward names to functions
REWARD_FUNCTIONS = {
    'default': humaniod_walking_reward,
    'walk': humaniod_walking_reward  # Add this line
}