"""
Hyper-Connections: An alternative to residual connections
Paper: "HYPER-CONNECTIONS" (ICLR 2025)
Authors: Defa Zhu, Hongzhi Huang, et al., ByteDance
"""

from .hyper_connection import HyperConnection
from .transformer_hyper import (
    TransformerEncoderHyper,
    TransformerBlockHyper,
    MultiHeadAttention,
    FeedForward
)

__version__ = "0.1.0"

__all__ = [
    "HyperConnection",
    "TransformerEncoderHyper",
    "TransformerBlockHyper",
    "MultiHeadAttention",
    "FeedForward",
]