# TTT-Video-DiT Implementation Analysis

## Executive Summary

This document compares the TTT implementation in **ttt-video-dit** (video generation) with the current **OpenPI** implementation (action denoising). The analysis identifies techniques, optimizations, and architectural patterns that could potentially benefit OpenPI.

---

## 1. Key Architectural Differences

### 1.1 Use Case Context
- **ttt-video-dit**: Long-range video generation (3-63 seconds), diffusion transformer for CogVideoX
- **OpenPI**: Action sequence denoising for robotics, Gemma-based causal language model

### 1.2 TTT Integration Strategy

| Aspect | ttt-video-dit | OpenPI |
|--------|---------------|---------|
| **Integration mode** | Parallel to attention (both run, outputs gated) | After attention (sequential) |
| **Bidirectional processing** | Yes - forward + reverse passes | No - unidirectional only |
| **Local vs Global** | Attention for local (3s segments), TTT for global context | TTT applied to full sequence |
| **Multiple gates** | 4 separate gates (forward/reverse × text/video) | 1 gate for TTT output |

---

## 2. Techniques NOT Yet Implemented in OpenPI

### 2.1 **Bidirectional TTT Processing** ⭐⭐⭐

**What they do:**
```python
# Forward pass
emb = forward_ssm(emb, seq_metadata)
emb = gate(forward_ssm_gating, residual, emb)

# Reverse pass
emb[:, text_length:] = torch.flip(emb[:, text_length:], dims=[1])
emb = reverse_ssm(emb, seq_metadata)
emb[:, text_length:] = torch.flip(emb[:, text_length:], dims=[1])
emb = gate(backward_ssm_gating, residual, emb)
```

**Why it matters:**
- TTT processes sequence in both directions (like bidirectional RNN)
- Captures dependencies from both past and future context
- Uses same TTT parameters for both directions (parameter efficient)
- Separate gating for forward and reverse branches

**Relevance to OpenPI:**
- **Medium-High**: Action sequences have temporal dependencies in both directions
- Denoising could benefit from "future" context when cleaning up intermediate actions
- Could help with consistency across the full trajectory

**Implementation complexity:** Medium (requires sequence reversal, dual gating)

---

### 2.2 **L2 Normalization on Q/K Projections** ⭐⭐

**What they do:**
```python
# After Q/K projections, before RoPE
XQ = torch.nn.functional.normalize(to_local(XQ), p=2, dim=-1)
XK = torch.nn.functional.normalize(to_local(XK), p=2, dim=-1)
```

**Why it matters:**
- Prevents gradient explosion with long sequences
- Stabilizes training by constraining query/key magnitudes
- Standard practice in vision transformers (e.g., CLIP)

**Relevance to OpenPI:**
- **Low-Medium**: OpenPI sequences are much shorter (256 tokens vs 1000s)
- Could help with training stability if you scale to longer sequences
- Minimal overhead, easy to add

**Implementation complexity:** Trivial (1 line after projections)

---

### 2.3 **Custom Triton/CUDA Kernels** ⭐

**What they do:**
- Fused Triton kernels for forward/backward passes
- Custom CUDA kernels for TTT linear operations
- ~2-3x speedup compared to PyTorch operations

**Files:**
- `ttt/models/ssm/linear_triton.py` - Triton autograd function
- `ttt/models/ssm/kernels/linear_forward.py` - Forward kernel
- `ttt/models/ssm/kernels/linear_backward.py` - Backward kernel

**Why it matters:**
- Significantly faster training and inference
- Fuses operations to reduce memory bandwidth
- Critical for long-context training (1000+ tokens)

**Relevance to OpenPI:**
- **Low**: OpenPI sequences are short (256 tokens), PyTorch is fine
- Kernels are H100-specific, adds deployment complexity
- Only worth it if TTT becomes a bottleneck

**Implementation complexity:** Very High (requires CUDA/Triton expertise)

---

### 2.4 **Sequential Mini-batch Processing with `scan`** ⭐⭐

**What they do:**
```python
def scan(f, init, xs, checkpoint_group=0):
    """Mimic jax.lax.scan function."""
    carry = init
    for i in range(num_items):
        carry, y = f(carry, xs[i])
        sub_out_list.append(y)
    return carry, out
```

- Processes sequence in chunks (mini_batch_size=64)
- Maintains state (W, b) across mini-batches
- Each mini-batch updates parameters for next mini-batch

**Why it matters:**
- Enables processing very long sequences (1000+ tokens)
- Gradient checkpointing within scan for memory efficiency
- Sequential dependency models temporal evolution

**Relevance to OpenPI:**
- **Low**: You explicitly chose batch-parallel optimization
- Your sequences are short enough for full parallelization
- Scan adds sequential bottleneck you wanted to avoid

**Implementation complexity:** Medium (already have reference code)

**Your design choice:** OpenPI uses **batch-parallel** (all tokens optimize W simultaneously), which is correct for your use case.

---

### 2.5 **Learnable TTT Learning Rate (per-head, input-dependent)** ⭐

**What they do:**
```python
def get_eta(self, X):
    ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, learnable_ttt_lr_weight) + learnable_ttt_lr_bias
    ttt_lr = F.sigmoid(ttt_lr)
    return self.ttt_base_lr * ttt_lr / self.head_dim
```

- Learning rate depends on input features
- Per-head learning rate (each head can adapt differently)
- Gated with sigmoid to keep in [0, base_lr]

**Relevance to OpenPI:**
- **REMOVED**: You already had this and intentionally removed it!
- User said: "不需要 因为用了 dual 形式 直接求闭式解了"
- Closed-form solution doesn't need adaptive LR

**Implementation complexity:** N/A (you removed it for good reasons)

---

### 2.6 **RoPE (Rotary Position Embedding) Integration** ⭐⭐

**What they do:**
```python
# Precompute 3D frequencies for video
freqs_cis = precompute_freqs_cis_3d(
    dim=head_dim,
    height=latent_height,
    width=latent_width,
    compressed_num_frames=num_frames,
    theta=10000.0
)

# Apply to Q/K after normalization
XQ_rope, XK_rope = apply_rotary_emb(XQ, XK, freqs_cis)
```

**Why it matters:**
- Encodes temporal position information
- Helps model distinguish different time steps
- Standard for sequence models

**Relevance to OpenPI:**
- **Medium**: Gemma already uses RoPE for attention
- TTT could benefit from position information
- Currently you don't use position in TTT (batch-parallel design)

**Trade-off:**
- Pro: Better temporal awareness
- Con: Breaks batch-parallel optimization (requires sequential processing)

**Implementation complexity:** Medium (need to pass RoPE embeddings to TTT)

---

### 2.7 **MLP Variant (TTTMLP vs TTTLinear)** ⭐⭐⭐

**What they do:**
- Two TTT variants: `TTTLinear` (linear model) and `TTTMLP` (2-layer MLP)
- TTTMLP has W1, b1, W2, b2 (like feedforward network)
- More expressive reconstruction model

```python
class TTTMLP(TTTBase):
    def __init__(self, config):
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, head_dim, 4*head_dim)))
        self.b1 = nn.Parameter(torch.zeros(num_heads, 1, 4*head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(num_heads, 4*head_dim, head_dim)))
        self.b2 = nn.Parameter(torch.zeros(num_heads, 1, head_dim))
```

**Why it matters:**
- More expressive model = better reconstruction
- Standard in original TTT paper
- Adds only ~4x parameters per head

**Relevance to OpenPI:**
- **High**: Could improve denoising quality
- Action sequences may need non-linear reconstruction
- Easy to implement as alternative

**Implementation complexity:** Low (extend current TTTLinear)

---

### 2.8 **Separate Text/Video Gating** ⭐

**What they do:**
```python
class SSMGating(nn.Module):
    def __init__(self, config):
        self.gating_alpha = nn.Parameter(torch.ones(model_dim) * gating_alpha_init)

    def forward(self, x):
        gating_alpha = torch.tanh(gating_alpha)
        return gating_alpha * x

# Separate gates for text and video
self.forward_ssm_gating_video = SSMGating(config)
self.forward_ssm_gating_text = SSMGating(config)
```

**Why it matters:**
- Different modalities need different mixing ratios
- Learned per-dimension gating (not scalar)
- Tanh activation keeps values in [-1, 1]

**Relevance to OpenPI:**
- **Low**: OpenPI is single-modality (action tokens)
- Your gate comes from AdaRMS (timestep-conditioned)
- Different gating philosophy

**Implementation complexity:** Low

---

### 2.9 **Tensor Parallelism (TP) Support** ⭐

**What they do:**
```python
def init_device_mesh(self, tp_mesh: DeviceMesh):
    # Shard parameters across devices
    self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
    self.ttt_norm_weight = nn.Parameter(distribute_tensor(self.ttt_norm_weight, tp_mesh, [Shard(0)]))

    # Mark for sharded execution
    TritonLinear.sharded_mode = True
```

**Why it matters:**
- Distributes computation across multiple GPUs
- Critical for their 5B parameter model
- Reduces memory per device

**Relevance to OpenPI:**
- **Low**: OpenPI model is smaller, fits on single GPU
- Adds significant complexity
- Only needed for very large models

**Implementation complexity:** High (requires PyTorch DTensor)

---

### 2.10 **Multi-Scene Interleaving** ⭐

**What they do:**
```python
def interleave(self, x, seq_metadata):
    # Split into chunks and interleave text+video for each scene
    x_text = torch.chunk(x_text, num_chunks, dim=2)
    x_video = torch.chunk(x_video, num_chunks, dim=2)

    x_interleaved = []
    for i in range(num_chunks):
        x_interleaved.append(torch.cat((x_text[i], x_video[i]), dim=2))
    return torch.cat(x_interleaved, dim=2)
```

**Why it matters:**
- Handles multiple scenes in one forward pass
- Keeps text and video synchronized
- Memory efficient for long videos

**Relevance to OpenPI:**
- **Very Low**: Specific to multi-modal video generation
- Not applicable to action sequences

**Implementation complexity:** N/A

---

### 2.11 **Gradient Checkpointing Integration** ⭐⭐

**What they do:**
```python
forward_ssm = (
    partial(torch.utils.checkpoint.checkpoint, self.ssm, use_reentrant=False)
    if self.do_forward_ssm_remat
    else self.ssm
)

# Checkpoint group size for scan
checkpoint_group_size = min(max(scan_checkpoint_group_size, 1), num_mini_batch)
```

**Why it matters:**
- Reduces memory usage during training
- Critical for long sequences
- Controlled via config flags

**Relevance to OpenPI:**
- **Medium**: Could help if memory becomes an issue
- OpenPI sequences are short, may not need it yet
- Good to have for future scaling

**Implementation complexity:** Low (PyTorch has built-in support)

---

### 2.12 **Reconstruction Target Normalization** ⭐⭐

**What they do:**
```python
def ln_reconstruction_target(self, XV, XK):
    # Compute residual
    XV = XV - XK

    # Layer normalize the residual (not the raw values)
    mean = XV.mean(dim=-1, keepdim=True)
    std = XV.std(dim=-1, keepdim=True)
    XV = (XV - mean) / (std + eps)

    # Apply learned weight and bias
    XV = self.ttt_norm_weight * XV + self.ttt_norm_bias

    # Add back XK
    return XV + XK
```

**Why it matters:**
- Normalizes the reconstruction target before optimization
- Prevents scale mismatch between XV and XK
- Stabilizes training

**Relevance to OpenPI:**
- **Medium**: You use `reconstruction_target = XV - XK` directly
- Adding normalization could improve stability
- Might help with diverse action scales

**Implementation complexity:** Low (just add normalization)

---

## 3. Architectural Patterns

### 3.1 Module Structure

**ttt-video-dit uses clear separation:**
```
TTTWrapper (position encoding, config management)
  └── TTTBase (Q/K/V projections, shared methods)
      ├── TTTLinear (linear reconstruction model)
      └── TTTMLP (MLP reconstruction model)
```

**OpenPI has:**
```
TTTWithAdaptiveNorm (monolithic, includes AdaRMS)
```

**Consideration:** Refactoring into base + variants would make it easier to experiment with TTTMLP.

---

### 3.2 `@torch.compile` Usage

**ttt-video-dit heavily uses compilation:**
```python
@torch.compile
def get_qkv_projections(self, hidden_states):
    ...

@torch.compile
def do_attention(cur_emb):
    ...
```

**Benefits:**
- 10-30% speedup in many cases
- No code changes needed
- PyTorch 2.0+ feature

**Relevance to OpenPI:**
- **High**: Easy performance win
- Just add `@torch.compile` decorator to hot paths
- Test thoroughly (can have bugs with dynamic shapes)

**Implementation complexity:** Trivial

---

## 4. Configuration and Training Strategy

### 4.1 Staged Training

**ttt-video-dit trains in stages:**
1. 3s (full SFT)
2. 9s, 18s, 30s, 63s (only TTT + QKVO projections trainable)

**Benefits:**
- Gradually extends context
- Prevents catastrophic forgetting
- More stable training

**Relevance to OpenPI:**
- **Low**: Action sequences don't extend like videos
- You train on fixed-length sequences
- Not applicable unless you want curriculum learning

---

### 4.2 Adapter Methods

**ttt-video-dit supports 3 modes:**
- `"sft"`: Full supervised fine-tuning (all params)
- `"qkvo"`: Only Q/K/V/O projections trainable
- `"none"`: TTT only (freeze base model)

**Benefits:**
- Flexible training strategies
- Can freeze pretrained weights
- Reduces training time

**Relevance to OpenPI:**
- **Medium**: Could be useful for Pi0 experiments
- Currently you seem to train all parameters
- Could speed up TTT-specific ablations

**Implementation complexity:** Low (use `requires_grad=False`)

---

## 5. Performance Optimizations Summary

| Technique | Speedup | Complexity | OpenPI Relevance |
|-----------|---------|------------|------------------|
| Triton/CUDA kernels | 2-3x | Very High | Low |
| `@torch.compile` | 1.1-1.3x | Trivial | High |
| Tensor Parallelism | 1.5-2x | High | Low |
| Gradient Checkpointing | 0.7x speed, 0.5x memory | Low | Medium |
| Dual form (closed-form) | 1.2-1.5x | Medium | **Already have** ✓ |

---

## 6. Recommendations for OpenPI

### 6.1 High Priority (Likely Beneficial)

1. **TTTMLP Variant** ⭐⭐⭐
   - Easy to implement (extend current code)
   - More expressive reconstruction
   - Could improve denoising quality
   - **Action:** Create `TTTMLPWithAdaptiveNorm` class

2. **`@torch.compile` Decorators** ⭐⭐⭐
   - Trivial to add
   - Free 10-30% speedup
   - No architectural changes
   - **Action:** Add to `ttt_batch_parallel`, `get_qkv_projections`

3. **Reconstruction Target Normalization** ⭐⭐
   - Stabilizes training
   - Handles scale variation better
   - Simple to add
   - **Action:** Add LayerNorm to `reconstruction_target = XV - XK`

### 6.2 Medium Priority (Worth Experimenting)

4. **Bidirectional TTT** ⭐⭐⭐
   - Potentially significant quality improvement
   - Requires dual gating, sequence reversal
   - **Action:** Implement as optional flag, compare single-pass vs bidirectional

5. **L2 Normalization on Q/K** ⭐⭐
   - Training stability
   - One-line change
   - **Action:** Add after projections, monitor training

6. **Gradient Checkpointing** ⭐⭐
   - Future-proofing for longer sequences
   - Easy with PyTorch
   - **Action:** Add config flag, test memory savings

### 6.3 Low Priority (Not Immediately Useful)

7. **RoPE in TTT** ⭐
   - Conflicts with batch-parallel design
   - Attention already has RoPE
   - **Skip for now**

8. **Sequential mini-batch scan** ⭐
   - You chose batch-parallel for good reasons
   - **Skip**

9. **Triton/CUDA kernels** ⭐
   - Overkill for 256-token sequences
   - High complexity
   - **Skip unless TTT is bottleneck**

10. **Tensor Parallelism** ⭐
    - Model fits on single GPU
    - **Skip**

---

## 7. Code Snippets for Quick Wins

### 7.1 Add `@torch.compile`

```python
# In ttt_with_gate.py

@torch.compile
def get_qkv_projections(self, hidden_states):
    XQ = self.q_proj(hidden_states)
    XK = self.k_proj(hidden_states)
    XV = self.v_proj(hidden_states)
    return XQ, XK, XV

@torch.compile
def ttt_batch_parallel(self, XQ, XK, XV, X):
    # ... existing code ...
    return output
```

### 7.2 Add L2 Normalization

```python
# In ttt_batch_parallel, after reshaping Q/K
XQ = F.normalize(XQ, p=2, dim=-1)
XK = F.normalize(XK, p=2, dim=-1)
```

### 7.3 Reconstruction Target Normalization

```python
# In ttt_batch_parallel
reconstruction_target = XV - XK

# Add normalization
eps = 1e-6
mean = reconstruction_target.mean(dim=-1, keepdim=True)
std = reconstruction_target.std(dim=-1, keepdim=True)
reconstruction_target = (reconstruction_target - mean) / (std + eps)

# Apply learnable scaling
reconstruction_target = (
    self.ttt_norm_weight.reshape(1, num_heads, 1, head_dim) * reconstruction_target +
    self.ttt_norm_bias.reshape(1, num_heads, 1, head_dim)
)
```

---

## 8. Conclusion

**Key Takeaway:** Your OpenPI TTT implementation is well-designed for the action denoising task. The main differences with ttt-video-dit are due to different use cases (short action sequences vs long videos).

**Best Quick Wins:**
1. Add `@torch.compile` (5 min, 10-30% speedup)
2. Implement TTTMLP variant (2-3 hours, quality improvement)
3. Add reconstruction target normalization (30 min, stability)

**Worth Experimenting:**
- Bidirectional TTT (if quality needs boost)
- L2 norm on Q/K (if training is unstable)

**Not Recommended:**
- Sequential mini-batch processing (conflicts with your design)
- Triton kernels (overkill for short sequences)
- Learnable LR (you removed it for good reasons)

Your batch-parallel, closed-form, dual-form design is appropriate for action denoising. The ttt-video-dit techniques are optimized for their specific use case (long-range video generation).
