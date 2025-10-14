"""Test script to verify TTT layer is called during training.

This script tests:
1. Models without TTT (use_ttt=False) work as before
2. Models with TTT (use_ttt=True) call TTT during training
"""

import torch
import torch.nn as nn

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def test_ttt_compatibility():
    """Test backward compatibility with models without TTT."""
    print("=" * 80)
    print("Test 1: Model without TTT (use_ttt=False)")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        use_ttt=False,  # Disable TTT
    )

    model = PI0Pytorch(config)

    # Check that Action Expert layers don't have TTT
    has_ttt = False
    for layer in model.paligemma_with_expert.gemma_expert.model.layers:
        if hasattr(layer, 'ttt_layer') and layer.ttt_layer is not None:
            has_ttt = True
            break

    print(f"Action Expert has TTT layer: {has_ttt}")
    assert not has_ttt, "Model with use_ttt=False should not have TTT layers"
    print("✅ PASS: Model without TTT has no TTT layers\n")


def test_ttt_enabled():
    """Test that TTT layers are created when enabled."""
    print("=" * 80)
    print("Test 2: Model with TTT (use_ttt=True)")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        use_ttt=True,  # Enable TTT
        ttt_layer_positions=None,  # All layers
    )

    model = PI0Pytorch(config)

    # Check that Action Expert layers have TTT
    ttt_count = 0
    for i, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers):
        if hasattr(layer, 'ttt_layer') and layer.ttt_layer is not None:
            ttt_count += 1
            print(f"  Layer {i}: has TTT ✓")

    print(f"\nTotal layers with TTT: {ttt_count}/{len(model.paligemma_with_expert.gemma_expert.model.layers)}")
    assert ttt_count > 0, "Model with use_ttt=True should have TTT layers"
    print("✅ PASS: Model with TTT has TTT layers in Action Expert\n")

    # Check that VLM doesn't have TTT
    vlm_has_ttt = False
    for layer in model.paligemma_with_expert.paligemma.language_model.layers:
        if hasattr(layer, 'ttt_layer') and layer.ttt_layer is not None:
            vlm_has_ttt = True
            break

    print(f"VLM has TTT layer: {vlm_has_ttt}")
    assert not vlm_has_ttt, "VLM should not have TTT layers"
    print("✅ PASS: VLM correctly does not have TTT layers\n")


def test_ttt_forward_pass():
    """Test that TTT is called during forward pass."""
    print("=" * 80)
    print("Test 3: TTT forward pass (training mode)")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        use_ttt=True,
        ttt_layer_positions=[17],  # Only last layer
    )

    model = PI0Pytorch(config)
    model.train()  # Training mode

    # Check which layers have TTT
    print("Checking TTT layer positions:")
    for i, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers):
        has_ttt = hasattr(layer, 'ttt_layer') and layer.ttt_layer is not None
        if has_ttt:
            print(f"  Layer {i}: has TTT ✓")

    # Hook to check if TTT is called
    ttt_called = {'count': 0}

    def ttt_forward_hook(module, input, output):
        ttt_called['count'] += 1
        print(f"  TTT layer called! (call #{ttt_called['count']})")

    # Register hook on TTT layer
    for i, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers):
        if hasattr(layer, 'ttt_layer') and layer.ttt_layer is not None:
            layer.ttt_layer.register_forward_hook(ttt_forward_hook)
            print(f"  Registered hook on layer {i}")

    # Create dummy input
    batch_size = 2
    observation = type('obj', (object,), {
        'images': {
            'base_0_rgb': torch.randn(batch_size, 224, 224, 3),
            'left_wrist_0_rgb': torch.randn(batch_size, 224, 224, 3),
            'right_wrist_0_rgb': torch.randn(batch_size, 224, 224, 3),
        },
        'image_masks': {
            'base_0_rgb': torch.ones(batch_size, dtype=torch.bool),
            'left_wrist_0_rgb': torch.ones(batch_size, dtype=torch.bool),
            'right_wrist_0_rgb': torch.ones(batch_size, dtype=torch.bool),
        },
        'state': torch.randn(batch_size, 32),
        'tokenized_prompt': torch.randint(0, 1000, (batch_size, 200)),
        'tokenized_prompt_mask': torch.ones(batch_size, 200, dtype=torch.bool),
    })
    actions = torch.randn(batch_size, 4, 32)

    print("\nRunning forward pass...")
    with torch.no_grad():  # Just for testing, no need for gradients
        loss = model(observation, actions)

    print(f"\nTTT called {ttt_called['count']} times during forward pass")
    assert ttt_called['count'] > 0, "TTT should be called during training forward pass"
    print("✅ PASS: TTT is correctly called during training\n")


def test_ttt_selective_layers():
    """Test TTT with selective layer positions."""
    print("=" * 80)
    print("Test 4: TTT with selective layers [14, 15, 16, 17]")
    print("=" * 80)

    config = Pi0Config(
        pi05=True,
        action_horizon=4,
        action_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        use_ttt=True,
        ttt_layer_positions=[14, 15, 16, 17],  # Last 4 layers
    )

    model = PI0Pytorch(config)

    # Check TTT positions
    ttt_positions = []
    for i, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers):
        if hasattr(layer, 'ttt_layer') and layer.ttt_layer is not None:
            ttt_positions.append(i)

    print(f"TTT enabled at layers: {ttt_positions}")
    print(f"Expected: [14, 15, 16, 17]")
    assert ttt_positions == [14, 15, 16, 17], f"TTT should only be at layers [14, 15, 16, 17], got {ttt_positions}"
    print("✅ PASS: TTT correctly enabled at specified layers\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TTT Training Compatibility Tests")
    print("=" * 80 + "\n")

    try:
        test_ttt_compatibility()
        test_ttt_enabled()
        test_ttt_selective_layers()
        test_ttt_forward_pass()

        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nSummary:")
        print("1. ✅ Models without TTT work as before (backward compatible)")
        print("2. ✅ Models with TTT have TTT layers in Action Expert only")
        print("3. ✅ TTT layers are correctly positioned based on config")
        print("4. ✅ TTT is called during training forward pass")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
