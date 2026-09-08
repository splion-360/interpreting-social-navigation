"""File description: Model interfaces for trajectory prediction experiments."""

from models.flat import (
    EdgeAttention,
    EdgeRNN,
    FlatSocialAttentionConfig,
    FlatSocialAttentionModel,
    NodeRNN,
)


__all__ = [
    "EdgeAttention",
    "EdgeRNN",
    "FlatSocialAttentionConfig",
    "FlatSocialAttentionModel",
    "NodeRNN",
]
