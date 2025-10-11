"""
Debug script for testing OpenPI policy with TTT (Test-Time Training) layers.

This script demonstrates how to use the TTT-enabled model for inference in SimplerEnv.
TTT layers perform online optimization during each forward pass, adapting to the current input.

Key differences from debug.py:
1. Uses "pi05_simpler_zscore_ttt" config (with TTT layers)
2. Loads TTT checkpoint (contains additional TTT parameters)
3. TTT layers automatically execute during policy.infer() - no special setup needed
4. Each forward pass performs test-time training optimization
"""

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
import numpy as np
from geometry import quat2mat, mat2euler
from transforms3d.euler import euler2axangle
# OpenPI imports
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.models_pytorch.transformers_replace.models.gemma.ttt_with_gate import TTTLossTracker


def preprocess_widowx_proprio(eef_pos) -> np.array:
    """
    Convert end-effector pose to the frame of top-down view.

    Reference: https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L167
    """
    default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    # StateEncoding.POS_EULER: xyz + rpy + pad + gripper(openness)
    proprio = eef_pos
    rm_bridge = quat2mat(proprio[3:7])
    rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
    gripper_openness = proprio[7]  # from simpler, 0 for close, 1 for open
    raw_proprio = np.concatenate(
        [
            proprio[:3],
            rpy_bridge_converted,
            np.zeros(1),
            [gripper_openness],
        ]
    )
    return raw_proprio


# ============================================================================
# TTT Configuration Setup
# ============================================================================

# Use TTT-enabled config
config_name = "pi05_simpler_zscore_ttt"

# Select TTT checkpoint - choose the latest or best performing checkpoint
# Available checkpoints: 1000, 2000, ..., 19000
checkpoint_dir = "/opt/tiger/openpi/checkpoints/pi05_simpler_zscore_ttt/pi05_simpler_zscore_ttt/19000"

# Alternative checkpoints (uncomment to use):
# checkpoint_dir = "/opt/tiger/openpi/checkpoints/pi05_simpler_zscore_ttt/pi05_simpler_zscore_ttt/15000"
# checkpoint_dir = "/opt/tiger/openpi/checkpoints/pi05_simpler_zscore_ttt/pi05_simpler_zscore_ttt/10000"

print("=" * 80)
print("TTT-Enabled OpenPI Policy Debug Script")
print("=" * 80)
print(f"Config: {config_name}")
print(f"Checkpoint: {checkpoint_dir}")
print()

# ============================================================================
# Load Policy
# ============================================================================

print("Loading OpenPI policy with TTT...")
training_config = _config.get_config(config_name)
policy = _policy_config.create_trained_policy(training_config, checkpoint_dir)

# ============================================================================
# Verify TTT Configuration
# ============================================================================

print("\n" + "=" * 80)
print("TTT Configuration Verification")
print("=" * 80)

model_config = training_config.model
print(f"✓ use_ttt: {model_config.use_ttt}")
print(f"✓ ttt_layer_type: {model_config.ttt_layer_type}")
print(f"✓ ttt_layer_positions: {model_config.ttt_layer_positions}")
print(f"✓ use_dual_form: {model_config.use_dual_form}")
print(f"✓ ttt_base_lr: {model_config.ttt_base_lr}")

print("\n" + "=" * 80)
print("How TTT Works During Inference:")
print("=" * 80)
print("""
During each policy.infer() call:
1. Input flows through Vision Encoder (PaliGemma)
2. Action Expert (Gemma) processes with TTT layers:
   - For each layer with TTT:
     a. Attention Block computes standard attention
     b. TTT Layer performs online optimization:
        * Initialize W1, b1 from checkpoint
        * Compute input-dependent learning rate (eta)
        * Execute closed-form optimization (dual form)
        * Output optimized features
     c. MLP Block processes the result
3. Output: action predictions [batch, horizon, action_dim]

Key Point: TTT automatically executes in every forward pass - no special code needed!
""")

print("=" * 80)
print("Policy loaded successfully!")
print("=" * 80)
print()

# ============================================================================
# Initialize Environment
# ============================================================================

env = simpler_env.make('widowx_carrot_on_plate')
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Environment Info:")
print(f"  Task: widowx_carrot_on_plate")
print(f"  Instruction: {instruction}")
print(f"  Reset info: {reset_info}")
print()

done, truncated = False, False
images = []  # Collect frames for video
timestep = 0

# ============================================================================
# Initialize TTT Loss Tracker
# ============================================================================

# Get global singleton loss tracker
loss_tracker = TTTLossTracker()
print("✓ TTT Loss Tracker initialized (tracks reconstruction loss per layer)")
print()

# ============================================================================
# Main Rollout Loop
# ============================================================================

image = get_image_from_maniskill2_obs_dict(env, obs)
images.append(image)

print("=" * 80)
print("Starting Episode Rollout with TTT")
print("=" * 80)

while not (done or truncated):
    # Prepare observation for OpenPI policy
    robot_state = preprocess_widowx_proprio(obs['agent']['eef_pos'])
    policy_input = {
        'image': image,  # RGB image from SimplerEnv
        'state': robot_state,  # Robot state (position, orientation, gripper)
        'prompt': instruction
    }

    # ========================================================================
    # Get action from OpenPI policy with TTT
    # ========================================================================
    # During this call, TTT layers perform online optimization:
    # - Each TTT layer adapts its parameters to the current input
    # - Optimization is done via closed-form solution (dual form)
    # - Input-dependent learning rate ensures adaptive updates

    # Reset loss tracker for this inference step
    loss_tracker.reset()

    action_dict = policy.infer(policy_input)

    # Print TTT reconstruction losses for all layers
    loss_tracker.next_step()
    print()  # Add blank line for readability
    action_sequence = action_dict['actions'][:10, :7]  # Extract action sequence

    # Execute actions in the environment
    for i, action in enumerate(action_sequence):
        print(f"[Timestep {timestep}] Executing action {i+1}/{len(action_sequence)}")
        print(f"  Action: {action}")

        # Convert rotation from euler angles to axis-angle representation
        action_rotation_delta = action[3:6]  # euler angles (roll, pitch, yaw)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action_scale = 1.0

        action[-1] = 2.0 * (action[-1] > 0.5) - 1.0  # Binarize gripper
        action[3:6] = action_rotation_axangle * action_scale

        # Step environment
        obs, reward, done, truncated, info = env.step(action)

        # Collect image
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        timestep += 1

        break  # Execute only first action, then replan (closed-loop control)

        if done or truncated:
            break

    # Check for new instruction (for long-horizon tasks)
    new_instruction = env.get_language_instruction()
    if new_instruction != instruction:
        instruction = new_instruction
        print(f"[New Instruction] {instruction}")

# ============================================================================
# Save Results
# ============================================================================

episode_stats = info.get('episode_stats', {})
success = episode_stats.get('success', False)

print()
print("=" * 80)
print("Episode Complete")
print("=" * 80)
print(f"Episode stats: {episode_stats}")
print(f"Success: {success}")
print(f"Total timesteps: {timestep}")

# Save video with TTT identifier in filename
video_filename = f"openpi_ttt_debug_success_{success}_steps_{timestep}.mp4"
write_video(video_filename, images, fps=5)
print(f"\n✓ Video saved as: {video_filename}")
print("=" * 80)
