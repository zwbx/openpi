import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video  # 添加视频保存功能
import numpy as np
from geometry import quat2mat, mat2euler
from transforms3d.euler import euler2axangle
# OpenPI imports
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def preprocess_widowx_proprio(eef_pos) -> np.array:
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]) 
    """convert ee rotation to the frame of top-down
    https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L167
    """
    # StateEncoding.POS_EULER: xyz + rpy + pad + gripper(openness)
    proprio = eef_pos
    rm_bridge = quat2mat(proprio[3:7])
    rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
    gripper_openness = proprio[7] # from simpler, 0 for close, 1 for open
    raw_proprio = np.concatenate(
        [
            proprio[:3],
            rpy_bridge_converted,
            np.zeros(1),
            [gripper_openness],
        ]
    )
    return raw_proprio


# Initialize OpenPI policy
# config_name = "pi05_simpler_low_mem_finetune"
config_name = "pi05_simpler"
# checkpoint_dir = "/opt/tiger/openpi/checkpoints/pi05_simpler_low_mem_finetune/pi05_simpler_low_mem_finetune/20000"
checkpoint_dir = "/mnt/hdfs/wenbo/vla/pi05_simpler_28000/"

print("Loading OpenPI policy...")
training_config = _config.get_config(config_name)
policy = _policy_config.create_trained_policy(training_config, checkpoint_dir)
print("Policy loaded successfully!")

env = simpler_env.make('widowx_carrot_on_plate')
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

done, truncated = False, False
images = []  # 收集每一帧图像用于视频保存
timestep = 0


image = get_image_from_maniskill2_obs_dict(env, obs)
images.append(image)  # 收集图像帧
while not (done or truncated):
   # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
   # action[6:7]: gripper (the meaning of open / close depends on robot URDF)

    # import pdb; pdb.set_trace()
    # Prepare observation for OpenPI policy
    # Extract robot state from observation
    robot_state = preprocess_widowx_proprio(obs['agent']['eef_pos'])  # Robot joint positions and gripper state
    policy_input = {
        'image': image,  # RGB image from SimplerEnv
        'state': robot_state,  # Actual robot state from environment
        'prompt': instruction
    }

    # Get action from OpenPI policy
    action_dict = policy.infer(policy_input)

    action_sequence = action_dict['actions'][:10,:7]  # Extract action sequence
    #    import pdb; pdb.set_trace()

    # Execute the complete action sequence
    for i, action in enumerate(action_sequence):
        print(f"Step {timestep}, Executing action {i+1}/{len(action_sequence)}: {action}")

        # Convert rotation from euler angles to axis-angle representation
        action_rotation_delta = action[3:6]  # Assuming these are euler angles (roll, pitch, yaw)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action_scale = 1.0  # Default action scale

        action[-1] = 2.0 * (action[-1] > 0.5) - 1.0
        action[3:6] = action_rotation_axangle * action_scale

        obs, reward, done, truncated, info = env.step(action) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold

        # Collect image after each action step
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)

        # Update timestep for each action in the sequence
        timestep += 1

        break
        # Check if episode is done after each action
        if done or truncated:
            break

    # Check for new instruction after completing the action sequence
    new_instruction = env.get_language_instruction()
    if new_instruction != instruction:
        # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
        instruction = new_instruction
        print("New Instruction", instruction)

    # Note: timestep is now updated inside the action loop for each executed action

episode_stats = info.get('episode_stats', {})
success = episode_stats.get('success', False)
print("Episode stats", episode_stats)

# 保存视频
video_filename = f"openpi_debug_episode_success_{success}.mp4"
write_video(video_filename, images, fps=5)
print(f"Video saved as {video_filename}")