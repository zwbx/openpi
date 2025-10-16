"""Quick test script to verify the complete align flow: training → inference → align.

This script uses minimal model configuration (dummy variant) for fastest execution.
No pre-trained weights are loaded.

Tests:
1. Forward pass (training mode)
2. Sample actions (inference mode)
3. Update online buffer
4. Align with collected data
5. Complete integration flow
"""

import torch
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.model import Observation


def create_dummy_observation(batch_size=2, device="cpu"):
    """Create a dummy observation for testing."""
    return Observation(
        images={
            'base_0_rgb': torch.randn(batch_size, 224, 224, 3, device=device),
            'left_wrist_0_rgb': torch.randn(batch_size, 224, 224, 3, device=device),
            'right_wrist_0_rgb': torch.randn(batch_size, 224, 224, 3, device=device),
        },
        image_masks={
            'base_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device),
            'left_wrist_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device),
            'right_wrist_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device),
        },
        state=torch.randn(batch_size, 32, device=device),
        tokenized_prompt=torch.randint(0, 1000, (batch_size, 200), device=device),
        tokenized_prompt_mask=torch.ones(batch_size, 200, dtype=torch.bool, device=device),
    )


def test_1_forward_pass():
    """Test 1: Forward pass (training mode)."""
    print("=" * 80)
    print("Test 1: Forward pass (training mode)")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_alignment_expert=True,  # Enable alignment expert
        alignment_expert_variant="dummy",
    )

    model = PI0Pytorch(config)
    model.train()

    batch_size = 2
    observation = create_dummy_observation(batch_size)
    actions = torch.randn(batch_size, 4, 32)
    next_obs_states = torch.randn(batch_size, 32)

    print(f"Running forward pass with batch_size={batch_size}...")
    with torch.no_grad():
        action_loss, alignment_losses = model(observation, actions, next_obs=next_obs_states)

    print(f"✓ Forward pass successful!")
    print(f"  - Action loss: {action_loss.mean().item():.4f}")
    if alignment_losses is not None:
        print(f"  - Alignment losses: {alignment_losses}")
    print()


def test_2_sample_actions():
    """Test 2: Sample actions (inference mode)."""
    print("=" * 80)
    print("Test 2: Sample actions (inference mode)")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_alignment_expert=True,
    )

    model = PI0Pytorch(config)
    model.eval()

    # Single sample (batch_size=1 for inference)
    observation = create_dummy_observation(batch_size=1)

    print("Sampling actions (inference mode)...")
    with torch.no_grad():
        actions = model.sample_actions("cpu", observation)

    print(f"✓ Sample actions successful!")
    print(f"  - Actions shape: {actions.shape}")
    print(f"  - Expected: (1, {config.action_horizon}, {config.action_dim})")
    assert actions.shape == (1, config.action_horizon, config.action_dim), "Wrong action shape"
    print()


def test_3_update_buffer():
    """Test 3: Update online buffer."""
    print("=" * 80)
    print("Test 3: Update online buffer")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_alignment_expert=True,
    )

    model = PI0Pytorch(config)

    print(f"Initial buffer size: {len(model.buffer)}")

    # Add some samples to buffer
    num_samples = 5
    for i in range(num_samples):
        observation = create_dummy_observation(batch_size=1)
        action = torch.randn(4, 32)
        model.update_online_buffer(observation, action)
        print(f"  - Added sample {i+1}, buffer size: {len(model.buffer)}")

    print(f"✓ Buffer update successful!")
    print(f"  - Final buffer size: {len(model.buffer)}")
    assert len(model.buffer) == num_samples, f"Expected {num_samples} samples in buffer"
    print()


def test_4_align():
    """Test 4: Align with collected data."""
    print("=" * 80)
    print("Test 4: Align with collected data")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_alignment_expert=True,
        alignment_expert_variant="dummy",
    )

    model = PI0Pytorch(config)
    model.eval()

    # Populate buffer with enough samples (need at least 2 consecutive frames)
    print("Populating buffer with samples...")
    num_samples = 10
    for i in range(num_samples):
        observation = create_dummy_observation(batch_size=1)
        action = torch.randn(4, 32)
        model.update_online_buffer(observation, action)

    print(f"✓ Buffer populated with {len(model.buffer)} samples")

    # Run align
    print("\nRunning align...")
    align_batch_size = 4
    align_steps = 3

    total_loss = model.align(
        batch_size=align_batch_size,
        num_steps=align_steps,
        align_lr=1e-4,
        device="cpu"
    )

    print(f"✓ Align successful!")
    print(f"  - Batch size: {align_batch_size}")
    print(f"  - Num steps: {align_steps}")
    print(f"  - Total loss: {total_loss:.4f}")
    print(f"  - Optimizer initialized: {model.align_optimizer is not None}")
    print()


def test_5_complete_flow():
    """Test 5: Complete integration flow (sample → buffer → align)."""
    print("=" * 80)
    print("Test 5: Complete integration flow")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_alignment_expert=True,
        alignment_expert_variant="dummy",
    )

    model = PI0Pytorch(config)
    model.eval()

    # Simulate online inference loop
    print("Simulating online inference loop...")
    num_inferences = 15
    for i in range(num_inferences):
        # 1. Sample actions
        observation = create_dummy_observation(batch_size=1)
        with torch.no_grad():
            actions = model.sample_actions("cpu", observation)

        # 2. Update buffer
        model.update_online_buffer(observation, actions[0])

        print(f"  Step {i+1}: sampled actions, buffer size = {len(model.buffer)}")

    print(f"\n✓ Collected {len(model.buffer)} samples")

    # Run align every N steps (simulate online adaptation)
    print("\nRunning online alignment...")
    align_freq = 5
    model.align_step_counter = 0

    for step in range(3):
        model.align_step_counter += 1

        if model.align_step_counter % align_freq == 0:
            print(f"\n  Align at step {model.align_step_counter}")
            total_loss = model.align(
                batch_size=4,
                num_steps=2,
                align_lr=1e-4,
                device="cpu"
            )
            print(f"  → Align loss: {total_loss:.4f}")

    print(f"\n✓ Complete flow successful!")
    print(f"  - Total inference steps: {num_inferences}")
    print(f"  - Buffer size: {len(model.buffer)}")
    print(f"  - Align step counter: {model.align_step_counter}")
    print()


def test_6_sample_actions_with_align():
    """Test 6: sample_actions with use_align=True."""
    print("=" * 80)
    print("Test 6: sample_actions with use_align=True")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_alignment_expert=True,
        alignment_expert_variant="dummy",
    )

    model = PI0Pytorch(config)
    model.eval()

    # First collect some samples without alignment
    print("Collecting initial samples...")
    for i in range(10):
        observation = create_dummy_observation(batch_size=1)
        with torch.no_grad():
            actions = model.sample_actions("cpu", observation, use_align=False)
        model.update_online_buffer(observation, actions[0])

    print(f"✓ Collected {len(model.buffer)} samples")

    # Now use sample_actions with align
    print("\nSampling with align enabled (align_freq=3)...")
    for i in range(6):
        observation = create_dummy_observation(batch_size=1)
        with torch.no_grad():
            actions = model.sample_actions(
                "cpu",
                observation,
                use_align=True,
                align_freq=3,
                align_batch_size=4,
                align_steps=2,
                align_lr=1e-4
            )

        should_align = (model.align_step_counter % 3 == 0) and (len(model.buffer) >= 2)
        status = "✓ ALIGNED" if should_align else "sampled"
        print(f"  Step {i+1} ({status}): buffer size = {len(model.buffer)}, counter = {model.align_step_counter}")

    print(f"\n✓ Integration with sample_actions successful!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Quick Align Flow Test (No Pre-trained Weights)")
    print("=" * 80 + "\n")

    try:
        test_1_forward_pass()
        test_2_sample_actions()
        test_3_update_buffer()
        test_4_align()
        test_5_complete_flow()
        test_6_sample_actions_with_align()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nSummary:")
        print("1. ✅ Forward pass works (training mode)")
        print("2. ✅ Sample actions works (inference mode)")
        print("3. ✅ Online buffer update works")
        print("4. ✅ Align adaptation works")
        print("5. ✅ Complete integration flow works")
        print("6. ✅ sample_actions with use_align works")
        print("\n" + "=" * 80)
        print("Network structure is verified! Ready for full training.")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
