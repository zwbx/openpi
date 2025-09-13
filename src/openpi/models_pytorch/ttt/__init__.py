"""
TTT (Test-Time Training) integration for Action Expert.

This module provides TTT layers that can be integrated into transformer architectures
to enhance long-sequence modeling capabilities through test-time adaptation.

Core components:
- TTTConfig: Configuration for TTT layers
- TTTBase: Abstract base class for TTT implementations  
- TTTLinear: Linear TTT variant for efficiency
- TTTMLP: MLP TTT variant for expressiveness
- TTTWrapper: High-level interface for TTT layers
"""

from .ttt_config import TTTConfig
from .ttt_layers import TTTBase, TTTLinear, TTTMLP, TTTWrapper

__all__ = [
    "TTTConfig",
    "TTTBase", 
    "TTTLinear",
    "TTTMLP", 
    "TTTWrapper"
]