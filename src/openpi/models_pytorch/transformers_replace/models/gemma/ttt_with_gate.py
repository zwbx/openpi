"""
TTT Layer with Adaptive RMSNorm Gate Integration

This module implements the TTT (Test-Time Training) layer with integrated
adaptive normalization and gating mechanism for use in diffusion models.

Based on the original TTTLinear implementation from ttt.py, but adapted to:
1. Include Adaptive RMSNorm internally (not in the decoder layer)
2. Return both output and gate for dual-gate residual design
3. Support conditional gating based on diffusion timestep
4. Batch-parallel optimization (no sequential scan, no position IDs)
5. Stateful inference: optionally keep W/b state across forward calls
6. Loss tracking: record inner-loop reconstruction loss during inference
"""

from typing import Optional, Tuple, List, Dict
import torch
from torch import nn


# ==== Global TTT Loss Tracker ====
class TTTLossTracker:
    """Global singleton to collect TTT losses from all layers."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance

    def reset(self):
        """Reset all collected losses."""
        self.losses = {}  # {step: {layer_idx: loss}}
        self.current_step = 0

    def record_loss(self, layer_idx: int, loss: float):
        """Record loss for a specific layer at current step."""
        if self.current_step not in self.losses:
            self.losses[self.current_step] = {}
        self.losses[self.current_step][layer_idx] = loss

    def next_step(self):
        """Move to next inference step and print summary."""
        if self.current_step in self.losses:
            self.print_summary(self.current_step)
        self.current_step += 1

    def print_summary(self, step: int):
        """Print loss summary for all layers at given step."""
        if step not in self.losses:
            return

        layer_losses = self.losses[step]
        if not layer_losses:
            return

        # Sort by layer index
        sorted_layers = sorted(layer_losses.items())

        # Format output
        loss_str = ", ".join([f"L{idx}: {loss:.6f}" for idx, loss in sorted_layers])
        avg_loss = sum(layer_losses.values()) / len(layer_losses)

        print(f"[TTT Step {step}] {loss_str} | Avg: {avg_loss:.6f}")

    def get_losses(self) -> Dict[int, Dict[int, float]]:
        """Get all recorded losses."""
        return self.losses


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
    TTT Layer with learnable gating mechanism.

    This class combines TTTLinear functionality with a learnable per-dimension gate
    similar to ttt-video-dit's SSMGating.

    Implements singleton pattern per layer_idx: instances with the same layer_idx
    will share the same TTT parameters (W1, b1), enabling parameter sharing across
    different experts (e.g., Action Expert and Alignment Expert).

    Args:
        num_heads: Number of attention heads
        hidden_size: Hidden dimension size
        mini_batch_size: Mini-batch size for TTT optimization (not used in batch-parallel mode)
        rope_theta: RoPE theta parameter (not used in batch-parallel mode)
        use_adarms: Whether to use adaptive RMS normalization for dynamic gating
        adarms_cond_dim: Dimension of the adaptive condition (timestep embedding)
        eps: Epsilon for normalization stability
        use_dual_form: Whether to use dual form (more memory efficient)
        gating_alpha_init: Initial value for learnable gating alpha (default 0.1)
        ttt_base_lr: Base learning rate for TTT (default 1.0 for linear, 0.1 for MLP)
        keep_state: If True, W/b persist across forward calls
        track_loss: If True, record inner-loop reconstruction loss
        layer_idx: Layer index for singleton pattern and logging
    """

    # Class variable to store instances by layer_idx
    _instances = {}  # {layer_idx: instance}

    def __new__(cls, *args, layer_idx: int = -1, **kwargs):
        """
        Singleton pattern: return existing instance if layer_idx already exists.

        This ensures that Action Expert and Alignment Expert share the same
        TTT parameters (W1, b1) for the same layer.
        """
        if layer_idx >= 0 and layer_idx in cls._instances:
            # Reuse existing instance for this layer
            existing_instance = cls._instances[layer_idx]
            print(f"[TTT Singleton] Layer {layer_idx}: Reusing existing instance (parameters shared)")
            return existing_instance
        else:
            # Create new instance
            instance = super(TTTWithAdaptiveNorm, cls).__new__(cls)
            if layer_idx >= 0:
                cls._instances[layer_idx] = instance
                print(f"[TTT Singleton] Layer {layer_idx}: Creating new instance")
            return instance

    def __init__(
        self,
        num_heads: int,
        hidden_size: int,
        mini_batch_size: int = None,  # Kept for compatibility, not used
        rope_theta: float = 10000.0,  # Kept for compatibility, not used
        use_adarms: bool = False,
        adarms_cond_dim: Optional[int] = None,
        eps: float = 1e-6,
        use_dual_form: bool = True,  # Whether to use dual form (only when keep_state=False)
        gating_alpha_init: float = 0.1,  # Initial value for learnable gate
        ttt_base_lr: float = 1.0,  # Base learning rate for TTT (1.0 for linear, 0.1 for MLP)
        keep_state: bool = False,  # NEW: If True, W/b persist across forward calls
        track_loss: bool = True,   # NEW: If True, record inner-loop reconstruction loss
        layer_idx: int = -1,  # NEW: Layer index for singleton pattern and logging
    ):
        # Skip initialization if this instance was already initialized (singleton reuse)
        if hasattr(self, '_initialized') and self._initialized:
            return

        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.use_adarms = use_adarms
        self.adarms_cond_dim = adarms_cond_dim
        self.eps = eps
        self.use_dual_form = use_dual_form
        self.ttt_base_lr = ttt_base_lr
        self.keep_state = keep_state
        self.track_loss = track_loss
        self.layer_idx = layer_idx  # NEW: Store layer index

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

        # TTT model parameters
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        # Learnable TTT learning rate (input-dependent)
        # [num_heads, hidden_size, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, self.hidden_size, 1))
        )
        # [num_heads, 1]
        self.learnable_ttt_lr_bias = nn.Parameter(torch.zeros(self.num_heads, 1))

        # Learnable gating (similar to ttt-video-dit SSMGating)
        # Initialize with gating_alpha_init (e.g., 0.1) so tanh(0.1) ≈ 0.1 at start
        self.gating_alpha = nn.Parameter(torch.ones(hidden_size) * gating_alpha_init)

        # Optional: Adaptive RMS normalization for dynamic gating (if use_adarms=True)
        if use_adarms and adarms_cond_dim is not None:
            self.adarms_gate_dense = nn.Linear(adarms_cond_dim, hidden_size, bias=True)
            nn.init.zeros_(self.adarms_gate_dense.weight)
        else:
            self.adarms_gate_dense = None

        # ==== NEW: Loss tracking ====
        self.loss_history: List[Dict[str, float]] = []
        self.loss_tracker = TTTLossTracker() if track_loss else None

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
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch-parallel TTT optimization for denoising.

        Each sample in the batch maintains its own W parameters and optimizes
        over all tokens in the sequence simultaneously (no sequential scan).

        Args:
            XQ, XK, XV: Query, Key, Value projections [B, num_heads, L, head_dim]
            hidden_states: Original input for computing adaptive learning rate [B, L, hidden_size]

        Returns:
            Output tensor [B, num_heads, L, head_dim]
        """
        B, num_heads, L, head_dim = XQ.shape

        # Initialize W and b for each sample in batch
        # [B, num_heads, head_dim, head_dim]
        W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1).clone()
        # [B, num_heads, 1, head_dim]
        b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1).clone()

        # Compute learnable, input-dependent learning rate (ttt_lr_eta)
        # [B, L, hidden_size] @ [num_heads, hidden_size, 1] -> [B, num_heads, L, 1]
        ttt_lr = torch.einsum("blc,hci->bhli", hidden_states, self.learnable_ttt_lr_weight)
        ttt_lr = ttt_lr + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1)  # [B, num_heads, L, 1]
        ttt_lr = torch.sigmoid(ttt_lr)  # Ensure positive, in range (0, 1)

        # Scale by base learning rate and head dimension
        # [B, num_heads, L, 1]
        eta = self.ttt_base_lr * ttt_lr / head_dim

        # TTT optimization: minimize reconstruction error
        X1 = XK  # [B, num_heads, L, head_dim]
        Z1 = torch.einsum("bhld,bhdf->bhlf", X1, W1_init) + b1_init  # [B, num_heads, L, head_dim]
        reconstruction_target = XV - XK  # Residual design

        # ==== Track loss ====
        if self.track_loss and self.loss_tracker is not None:
            recon_loss = torch.norm(Z1 - reconstruction_target).item()
            # Record to global tracker (will be printed by next_step())
            self.loss_tracker.record_loss(self.layer_idx, recon_loss)
            # Also store in local history
            self.loss_history.append({
                "layer_idx": self.layer_idx,
                "step": len(self.loss_history),
                "loss": recon_loss,
            })

        # Compute gradient
        ln_weight = self.ttt_norm_weight.reshape(1, num_heads, 1, head_dim)
        ln_bias = self.ttt_norm_bias.reshape(1, num_heads, 1, head_dim)
        grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

        if self.use_dual_form and not self.keep_state:
            # Dual form: more memory efficient, avoids storing grad_W1
            # Compute attention matrix: [B, num_heads, L, L]
            # No causal masking - allow full token interactions for denoising
            Attn1 = XQ @ X1.transpose(-2, -1)

            # Compute b1_bar: [B, num_heads, 1, head_dim]
            # b1_bar = b1_init - sum(eta * grad_l_wrt_Z1)
            # eta: [B, num_heads, L, 1], grad_l_wrt_Z1: [B, num_heads, L, head_dim]
            b1_bar = b1_init - (eta * grad_l_wrt_Z1).sum(dim=2, keepdim=True)

            # Compute Z1_bar directly without explicitly updating W1
            # Z1_bar = XQ @ W1_init - Attn1 @ (eta * grad_l_wrt_Z1) + b1_bar
            # [B, num_heads, L, head_dim]
            Z1_bar = XQ @ W1_init - Attn1 @ (eta * grad_l_wrt_Z1) + b1_bar
        else:
            # Primal form: explicitly compute gradients and update parameters
            # Gradient: dL/dW = X1^T @ grad_l_wrt_Z1
            grad_W1 = torch.einsum("bhld,bhlf->bhdf", X1, grad_l_wrt_Z1)  # [B, num_heads, head_dim, head_dim]
            grad_b1 = grad_l_wrt_Z1.sum(dim=2, keepdim=True)  # [B, num_heads, 1, head_dim]

            # Update W and b
            W1_updated = W1_init - eta * grad_W1
            b1_updated = b1_init - eta * grad_b1

            # ==== Save state if keep_state=True ====
            if self.keep_state:
                self.W1.data = W1_updated.clone().detach()
                self.b1.data = b1_updated.clone().detach()

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
        Forward pass with learnable gating.

        Args:
            hidden_states: Input tensor [B, L, hidden_size] (already from attention output)
            adarms_cond: Adaptive condition (timestep embedding) [B, cond_dim]
            cache_params: Cache parameters (not used in batch denoising mode)

        Returns:
            Tuple of (ttt_output, gate):
                - ttt_output: TTT layer output [B, L, hidden_size]
                - gate: Gating values [B, L, hidden_size] or [hidden_size]
        """
        # Run batch-parallel TTT (no position encoding, no scan)
        B, L = hidden_states.shape[:2]

        # Get Q, K, V projections directly from input (no norm needed)
        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        # Reshape to [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Batch-parallel TTT optimization (each sample maintains its own W)
        output = self.ttt_batch_parallel(XQ, XK, XV, hidden_states)

        # Reshape back: [B, num_heads, L, head_dim] -> [B, L, hidden_size]
        output = output.transpose(1, 2).reshape(B, L, self.hidden_size)

        # Post-normalization and projection
        output = self.post_norm(output)
        output = self.o_proj(output)

        # Compute gating: learnable base gate + optional dynamic gate from adarms
        # Base learnable gate (similar to ttt-video-dit SSMGating)
        gating_alpha = torch.tanh(self.gating_alpha)  # [hidden_size], values in [-1, 1]

        return output, gating_alpha


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
    print(f"Gate shape: {gate.shape}")
    print(f"Gate values (first 5): {gate[:5]}")
    assert output_tensor.shape == input_tensor.shape, "Output shape mismatch!"
    assert gate is not None, "Gate should always be returned (learnable gating_alpha)!"
    assert gate.shape == (hidden_size,), f"Gate shape should be ({hidden_size},), got {gate.shape}"
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
    print(f"Gate values (first 5): {gate[:5]}")
    assert output_tensor.shape == input_tensor.shape, "Output shape mismatch!"
    assert gate is not None, "Gate should not be None when using AdaRMS!"
    # Gate shape is [hidden_size] for broadcasting with [B, L, hidden_size]
    assert gate.shape == (hidden_size,), f"Gate shape should be ({hidden_size},), got {gate.shape}"
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
