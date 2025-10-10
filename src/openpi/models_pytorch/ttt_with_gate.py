"""
TTT Layer with Adaptive RMSNorm Gate Integration

This module implements the TTT (Test-Time Training) layer with integrated
adaptive normalization and gating mechanism for use in diffusion models.

Based on the original TTTLinear implementation from ttt.py, but adapted to:
1. Include Adaptive RMSNorm internally (not in the decoder layer)
2. Return both output and gate for dual-gate residual design
3. Support conditional gating based on diffusion timestep
4. Batch-parallel optimization (no sequential scan, no position IDs)
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn.functional as F
from torch import nn


def ln_fwd(x, gamma, beta, eps=1e-6):
    """Batch forward for LayerNorm."""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    """Batch backward for LayerNorm fused with L2 loss."""
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )
    return z


class GemmaRMSNorm(nn.Module):
    """
    Gemma RMS Normalization with optional adaptive conditioning.

    Copied from modeling_gemma.py for standalone use.
    """
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        # Apply adaptive normalization
        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)


class TTTWithAdaptiveNorm(nn.Module):
    """
    TTT Layer with Adaptive RMSNorm integrated.

    This class combines TTTLinear functionality with adaptive normalization
    that can be conditioned on diffusion timestep.

    Args:
        num_heads: Number of attention heads
        hidden_size: Hidden dimension size
        mini_batch_size: Mini-batch size for TTT optimization (not used in batch-parallel mode)
        rope_theta: RoPE theta parameter (not used in batch-parallel mode)
        use_adarms: Whether to use adaptive RMS normalization
        adarms_cond_dim: Dimension of the adaptive condition (timestep embedding)
        eps: Epsilon for normalization stability
    """

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        mini_batch_size: int = None,  # Kept for compatibility, not used
        rope_theta: float = 10000.0,  # Kept for compatibility, not used
        use_adarms: bool = False,
        adarms_cond_dim: Optional[int] = None,
        eps: float = 1e-6,
        use_dual_form: bool = True,  # Whether to use dual form (more memory efficient)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.use_adarms = use_adarms
        self.adarms_cond_dim = adarms_cond_dim
        self.eps = eps
        self.use_dual_form = use_dual_form

        # Initialize Q/K/V/O projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # Initialize TTT LayerNorm parameters
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

        # Post normalization and output projection
        self.post_norm = nn.LayerNorm(self.hidden_size, eps=self.eps)
        self.ttt_base_lr = 1.0

        # TTT model parameters
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        # Adaptive RMS normalization (use GemmaRMSNorm for consistency with GemmaDecoderLayer)
        self.input_norm = GemmaRMSNorm(
            dim=hidden_size,
            eps=eps,
            cond_dim=adarms_cond_dim if (use_adarms and adarms_cond_dim is not None) else None
        )


    def get_qkv_projections(self, hidden_states):
        """Get Q, K, V projections."""
        XQ = self.q_proj(hidden_states)
        XK = self.k_proj(hidden_states)
        XV = self.v_proj(hidden_states)
        return XQ, XK, XV

    def ttt_batch_parallel(
        self,
        XQ: torch.Tensor,
        XK: torch.Tensor,
        XV: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch-parallel TTT optimization for denoising.

        Each sample in the batch maintains its own W parameters and optimizes
        over all tokens in the sequence simultaneously (no sequential scan).

        Args:
            XQ, XK, XV: Query, Key, Value projections [B, num_heads, L, head_dim]
            X: Original hidden states [B, L, hidden_size]

        Returns:
            Output tensor [B, num_heads, L, head_dim]
        """
        B, num_heads, L, head_dim = XQ.shape

        # Initialize W and b for each sample in batch
        # [B, num_heads, head_dim, head_dim]
        W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone()
        # [B, num_heads, 1, head_dim]
        b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1).clone()

        # Learning rate for closed-form solution
        # Since we use dual form with closed-form solution, eta is fixed
        eta = self.ttt_base_lr / head_dim  # Scalar eta

        # TTT optimization: minimize reconstruction error
        X1 = XK  # [B, num_heads, L, head_dim]
        Z1 = torch.einsum("bhld,bhdf->bhlf", X1, W1_init) + b1_init  # [B, num_heads, L, head_dim]
        reconstruction_target = XV - XK  # Residual design

        # Compute gradient
        ln_weight = self.ttt_norm_weight.reshape(1, num_heads, 1, head_dim)
        ln_bias = self.ttt_norm_bias.reshape(1, num_heads, 1, head_dim)
        grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

        if self.use_dual_form:
            # Dual form: more memory efficient, avoids storing grad_W1
            # Compute attention matrix: [B, num_heads, L, L]
            Attn1 = torch.tril(XQ @ X1.transpose(-2, -1))

            # Compute b1_bar: [B, num_heads, 1, head_dim]
            # b1_bar = b1_init - eta * sum(grad_l_wrt_Z1)
            b1_bar = b1_init - eta * grad_l_wrt_Z1.sum(dim=2, keepdim=True)

            # Compute Z1_bar directly without explicitly updating W1
            # Z1_bar = XQ @ W1_init - eta * Attn1 @ grad_l_wrt_Z1 + b1_bar
            # [B, num_heads, L, head_dim]
            Z1_bar = XQ @ W1_init - (eta * Attn1) @ grad_l_wrt_Z1 + b1_bar
        else:
            # Primal form: explicitly compute gradients and update parameters
            # Gradient: dL/dW = X1^T @ grad_l_wrt_Z1
            grad_W1 = torch.einsum("bhld,bhlf->bhdf", X1, grad_l_wrt_Z1)  # [B, num_heads, head_dim, head_dim]
            grad_b1 = grad_l_wrt_Z1.sum(dim=2, keepdim=True)  # [B, num_heads, 1, head_dim]

            # Update W and b
            W1_updated = W1_init - eta * grad_W1
            b1_updated = b1_init - eta * grad_b1

            # Forward with updated parameters
            Z1_bar = torch.einsum("bhld,bhdf->bhlf", XQ, W1_updated) + b1_updated

        # Normalize
        Z1_normalized = ln_fwd(Z1_bar, ln_weight, ln_bias)

        # Output with residual
        output = XQ + Z1_normalized  # [B, num_heads, L, head_dim]

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        adarms_cond: Optional[torch.Tensor] = None,
        cache_params: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with integrated adaptive normalization.

        Args:
            hidden_states: Input tensor [B, L, hidden_size]
            adarms_cond: Adaptive condition (timestep embedding) [B, cond_dim]
            cache_params: Cache parameters (not used in batch denoising mode)

        Returns:
            Tuple of (ttt_output, gate):
                - ttt_output: TTT layer output [B, L, hidden_size]
                - gate: Adaptive gate [B, 1] or None if not using adarms
        """
        # Step 1: Apply Adaptive RMS normalization (returns normalized_states and gate)
        normalized_hidden_states, gate = self.input_norm(hidden_states, adarms_cond)

        # Step 3: Run batch-parallel TTT (no position encoding, no scan)
        B, L = normalized_hidden_states.shape[:2]

        # Get Q, K, V projections
        XQ, XK, XV = self.get_qkv_projections(normalized_hidden_states)

        # Reshape to [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Batch-parallel TTT optimization (each sample maintains its own W)
        output = self.ttt_batch_parallel(XQ, XK, XV, normalized_hidden_states)

        # Reshape back: [B, num_heads, L, head_dim] -> [B, L, hidden_size]
        output = output.transpose(1, 2).reshape(B, L, self.hidden_size)

        # Post-normalization and projection
        output = self.post_norm(output)
        output = self.o_proj(output)

        return output, gate


if __name__ == "__main__":
    """Test the TTTWithAdaptiveNorm implementation."""
    print("Testing TTTWithAdaptiveNorm...")

    # Test configuration
    batch_size = 4
    seq_len = 16
    num_heads = 8
    hidden_size = 512
    mini_batch_size = 64
    adarms_cond_dim = 256

    # Create model instance
    print(f"\nCreating TTTWithAdaptiveNorm:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - mini_batch_size: {mini_batch_size}")

    # Test 1: Without AdaRMS (no adaptive gating)
    print("\n" + "="*60)
    print("Test 1: TTT without AdaRMS")
    print("="*60)
    model_no_adarms = TTTWithAdaptiveNorm(
        num_heads=num_heads,
        hidden_size=hidden_size,
        mini_batch_size=mini_batch_size,
        rope_theta=10000.0,
        use_adarms=False,
        adarms_cond_dim=None,
    )

    # Create input tensor
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input shape: {input_tensor.shape}")

    # Forward pass
    output_tensor, gate = model_no_adarms(input_tensor, adarms_cond=None)
    print(f"Output shape: {output_tensor.shape}")
    print(f"Gate: {gate}")
    assert output_tensor.shape == input_tensor.shape, "Output shape mismatch!"
    assert gate is None, "Gate should be None when not using AdaRMS!"
    print("✓ Test 1 passed!")

    # Test 2: With AdaRMS (adaptive gating enabled)
    print("\n" + "="*60)
    print("Test 2: TTT with AdaRMS")
    print("="*60)
    model_with_adarms = TTTWithAdaptiveNorm(
        num_heads=num_heads,
        hidden_size=hidden_size,
        mini_batch_size=mini_batch_size,
        rope_theta=10000.0,
        use_adarms=True,
        adarms_cond_dim=adarms_cond_dim,
    )

    # Create input and condition tensors
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    adarms_cond = torch.randn(batch_size, adarms_cond_dim)
    print(f"Input shape: {input_tensor.shape}")
    print(f"AdaRMS condition shape: {adarms_cond.shape}")

    # Forward pass
    output_tensor, gate = model_with_adarms(input_tensor, adarms_cond=adarms_cond)
    print(f"Output shape: {output_tensor.shape}")
    print(f"Gate shape: {gate.shape}")
    print(f"Gate mean per sample: {gate.mean(dim=-1)[:4, 0].tolist()}")
    assert output_tensor.shape == input_tensor.shape, "Output shape mismatch!"
    assert gate is not None, "Gate should not be None when using AdaRMS!"
    # Gate shape is [B, 1, hidden_size] for broadcasting with [B, L, hidden_size]
    assert gate.shape == (batch_size, 1, hidden_size), f"Gate shape should be ({batch_size}, 1, {hidden_size}), got {gate.shape}"
    print("✓ Test 2 passed!")

    # Test 3: Backward pass (gradient flow)
    print("\n" + "="*60)
    print("Test 3: Gradient flow test")
    print("="*60)
    model_with_adarms.zero_grad()
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    adarms_cond = torch.randn(batch_size, adarms_cond_dim, requires_grad=True)

    output_tensor, gate = model_with_adarms(input_tensor, adarms_cond)

    # Compute loss and backward
    if gate is not None:
        # Gate is already [B, 1, hidden_size], broadcasts correctly with [B, L, hidden_size]
        loss = (gate * output_tensor).sum()
    else:
        loss = output_tensor.sum()

    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"Input grad shape: {input_tensor.grad.shape}")
    print(f"Input grad norm: {input_tensor.grad.norm().item():.4f}")
    if adarms_cond.grad is not None:
        print(f"AdaRMS cond grad shape: {adarms_cond.grad.shape}")
        print(f"AdaRMS cond grad norm: {adarms_cond.grad.norm().item():.4f}")
    assert input_tensor.grad is not None, "Gradient should flow to input!"
    print("✓ Test 3 passed!")

    # Test 4: Different sequence lengths
    print("\n" + "="*60)
    print("Test 4: Different sequence lengths")
    print("="*60)
    for test_seq_len in [8, 16, 32, 64]:
        input_tensor = torch.randn(batch_size, test_seq_len, hidden_size)
        output_tensor, gate = model_with_adarms(input_tensor, adarms_cond)
        assert output_tensor.shape == input_tensor.shape, f"Failed for seq_len={test_seq_len}"
        print(f"  seq_len={test_seq_len}: ✓")
    print("✓ Test 4 passed!")

    print("\n" + "="*60)
    print("All tests passed! ✓✓✓")
    print("="*60)
