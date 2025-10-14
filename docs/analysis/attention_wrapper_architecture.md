# Attention Wrapper/Adapter Architecture Design

**Status**: Design proposal (not yet implemented)
**Date**: 2025-10-14
**Context**: Alternative architecture for integrating TTT and other adapters (LoRA, etc.) with attention layers

## Motivation

Currently, TTT layers are implemented as independent layers that are called separately from attention:

```python
# Current implementation in GemmaDecoderLayer
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)

        # TTT as separate layer
        if config.use_ttt and (layer_idx in ttt_layer_positions):
            self.ttt_layer = TTTWithAdaptiveNorm(...)
        else:
            self.ttt_layer = None

    def forward(self, hidden_states, ...):
        # Attention
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond)
        attn_output, attn_weights = self.self_attn(hidden_states, ...)
        hidden_states = _gated_residual(residual, attn_output, gate)

        # TTT (if enabled)
        if self.ttt_layer is not None:
            ttt_output, gate_ttt = self.ttt_layer(attn_output, cond, cache_params=None)
            if gate_ttt is not None:
                hidden_states = hidden_states + gate_ttt * ttt_output
            else:
                hidden_states = hidden_states + ttt_output

        # MLP
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond)
        mlp_output = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, mlp_output, gate)

        return hidden_states, attn_weights
```

**Problems with current approach**:
1. TTT logic is scattered across decoder layer code
2. Adding new adapters (LoRA, QLoRA, etc.) requires modifying decoder layer
3. Cannot easily swap between different adapter types
4. Harder to maintain and test different adapter configurations

## Proposed Architecture: Wrapper Pattern

Wrap the base attention module with an adapter that adds additional functionality:

```python
# Proposed implementation
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        # Create base attention
        base_attn = GemmaAttention(config, layer_idx)

        # Wrap with adapter based on config
        if config.use_ttt and (layer_idx in ttt_layer_positions):
            self.self_attn = TTTAttentionAdapter(base_attn, ttt_config)
        elif config.use_lora:
            self.self_attn = LoRAAttentionAdapter(base_attn, lora_config)
        else:
            self.self_attn = NoOpAttentionAdapter(base_attn)  # or just base_attn

        self.mlp = GemmaMLP(config)

    def forward(self, hidden_states, ...):
        # Attention (adapter handles everything)
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond)
        attn_output, attn_weights = self.self_attn(
            hidden_states,
            adarms_cond=cond,
            ...
        )
        hidden_states = _gated_residual(residual, attn_output, gate)

        # MLP (unchanged)
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond)
        mlp_output = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, mlp_output, gate)

        return hidden_states, attn_weights
```

## Design Overview

### Base Classes

```python
class BaseAttentionAdapter(nn.Module, ABC):
    """Abstract base class for attention adapters."""

    def __init__(self, base_attention: nn.Module):
        super().__init__()
        self.base_attention = base_attention
        # Copy essential attributes for compatibility
        self.layer_idx = base_attention.layer_idx
        self.config = base_attention.config
        self.head_dim = base_attention.head_dim
        # ... etc

    @abstractmethod
    def adapt(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply adaptation to attention output.

        Returns:
            (adapted_output, gate) tuple
        """
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: base attention + adaptation."""
        # 1. Base attention
        attn_output, attn_weights = self.base_attention(
            hidden_states, attention_mask, position_ids, **kwargs
        )

        # 2. Apply adaptation
        adapted_output, gate = self.adapt(
            attn_output, hidden_states, adarms_cond
        )

        # 3. Store gate for decoder layer
        self._last_gate = gate

        return adapted_output, attn_weights
```

### Concrete Implementations

#### 1. TTT Attention Adapter

```python
class TTTAttentionAdapter(BaseAttentionAdapter):
    """Wraps attention with TTT functionality."""

    def __init__(
        self,
        base_attention: nn.Module,
        num_heads: int,
        hidden_size: int,
        use_adarms: bool = False,
        adarms_cond_dim: Optional[int] = None,
        use_dual_form: bool = True,
        gating_alpha_init: float = 0.1,
        ttt_base_lr: float = 1.0,
        **kwargs
    ):
        super().__init__(base_attention)

        self.ttt_layer = TTTWithAdaptiveNorm(
            num_heads=num_heads,
            hidden_size=hidden_size,
            use_adarms=use_adarms,
            adarms_cond_dim=adarms_cond_dim,
            use_dual_form=use_dual_form,
            gating_alpha_init=gating_alpha_init,
            ttt_base_lr=ttt_base_lr,
            **kwargs
        )

    def adapt(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply TTT to attention output."""
        ttt_output, gate = self.ttt_layer(
            attn_output, adarms_cond, cache_params=None
        )
        return ttt_output, gate
```

#### 2. LoRA Attention Adapter

```python
class LoRAAttentionAdapter(BaseAttentionAdapter):
    """Wraps attention with LoRA low-rank adaptation."""

    def __init__(
        self,
        base_attention: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__(base_attention)

        hidden_size = base_attention.config.hidden_size
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def adapt(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply LoRA to attention output."""
        # LoRA applies to input, not output
        lora_output = self.lora_B(
            self.lora_A(self.dropout(hidden_states))
        ) * self.scaling

        return attn_output + lora_output, None
```

#### 3. No-Op Adapter

```python
class NoOpAttentionAdapter(BaseAttentionAdapter):
    """Identity adapter (pass-through)."""

    def adapt(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return attention output unchanged."""
        return attn_output, None
```

### Factory Function

```python
def create_attention_adapter(
    base_attention: nn.Module,
    adapter_type: str,
    adapter_config: dict,
) -> BaseAttentionAdapter:
    """Factory to create attention adapters.

    Args:
        base_attention: Base attention module
        adapter_type: "ttt", "lora", "none"
        adapter_config: Configuration dict

    Returns:
        Wrapped attention module
    """
    if adapter_type == "none" or adapter_type is None:
        return NoOpAttentionAdapter(base_attention)
    elif adapter_type == "ttt":
        return TTTAttentionAdapter(base_attention, **adapter_config)
    elif adapter_type == "lora":
        return LoRAAttentionAdapter(base_attention, **adapter_config)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
```

## Benefits

1. **Separation of Concerns**: Attention and adaptation logic are cleanly separated
2. **Extensibility**: Easy to add new adapters (LoRA, QLoRA, Adapter, etc.) without modifying decoder layer
3. **Flexibility**: Can swap adapters at runtime or based on configuration
4. **Testability**: Each adapter can be tested independently
5. **Maintainability**: Cleaner code structure with single responsibility
6. **Composability**: Could potentially chain multiple adapters (future work)

## Comparison

| Aspect | Current (Independent Layer) | Proposed (Wrapper) |
|--------|----------------------------|-------------------|
| **Code location** | Decoder layer + separate TTT call | Wrapped inside attention |
| **Adding new adapters** | Modify decoder layer forward() | Create new adapter class |
| **Swapping adapters** | Change multiple locations | Change wrapper at init |
| **Testing** | Test entire decoder layer | Test adapter in isolation |
| **Backward compatibility** | Need to check `if ttt_layer is not None` | Handled by NoOpAdapter |
| **Code complexity** | Scattered logic | Centralized in adapter |

## Integration Considerations

### When to Implement

Consider implementing this when:
1. Adding support for multiple adapter types (LoRA, QLoRA, Adapter, etc.)
2. Needing to swap adapters dynamically
3. Code complexity of current approach becomes unwieldy
4. Want to enable adapter composition (chaining multiple adapters)

### Migration Path

1. **Create adapter classes** (`attention_adapter.py`) ✓ Already done
2. **Add factory function** for easy instantiation ✓ Already done
3. **Update decoder layer** to use wrapped attention instead of separate TTT layer
4. **Update training code** in `gemma_pytorch.py` to handle wrapped attention
5. **Update tests** to verify backward compatibility
6. **Deprecate old TTT layer approach** once wrapper is stable

### Backward Compatibility

The wrapper pattern maintains backward compatibility by:
- `NoOpAttentionAdapter` provides identity behavior for models without adapters
- Attribute copying ensures compatibility with code that checks `layer_idx`, `config`, etc.
- Same forward() signature as base attention module

### Performance Considerations

- **Overhead**: Minimal - just one extra function call (adapt)
- **Memory**: Same as current approach - no additional overhead
- **Gradient flow**: No change - adapters participate in backprop normally
- **JIT compilation**: Should work with torch.jit.script (may need @torch.jit.export on adapt)

## Reference Implementation

Full implementation available in: `src/openpi/models_pytorch/transformers_replace/models/gemma/attention_adapter.py`

Key classes:
- `BaseAttentionAdapter`: Abstract base class (lines 33-125)
- `TTTAttentionAdapter`: TTT wrapper (lines 146-225)
- `LoRAAttentionAdapter`: LoRA wrapper (lines 228-287)
- `NoOpAttentionAdapter`: Identity wrapper (lines 128-143)
- `create_attention_adapter`: Factory function (lines 290-323)

## Future Work

1. **Adapter Composition**: Chain multiple adapters (e.g., TTT + LoRA)
2. **Dynamic Switching**: Switch adapters based on inference mode
3. **Adapter Merging**: Merge adapter weights into base model for deployment
4. **Mixed Adapters**: Different adapters for different layers
5. **Adapter Routing**: Route to different adapters based on input

## Related Documentation

- TTT implementation: `src/openpi/models_pytorch/transformers_replace/models/gemma/ttt_with_gate.py`
- Current decoder layer: `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`
- Training integration: `src/openpi/models_pytorch/gemma_pytorch.py` (lines 237-250)
- Test suite: `test_ttt_training.py`

## Notes

- This design was proposed on 2025-10-14 but implementation was deferred
- Decision made to document the idea for future reference
- Current implementation (independent TTT layer) is working correctly after bug fixes
- Wrapper pattern provides cleaner architecture for when multiple adapters are needed
