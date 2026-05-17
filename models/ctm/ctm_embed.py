"""CTMEmbed: per-particle embedding built from a single CTMPool.

Replaces the traditional ParT-style `Embed` MLP (`BatchNorm + (LayerNorm +
Linear + GELU)*N`). The pool is called once per outer CTM tick; state is
threaded across ticks so each particle's `n_neurons` neurons accumulate a
trace + synchronization recurrence even when the underlying input features
do not change.
"""

import torch
import torch.nn as nn

from .ctm_pool import CTMPool


class CTMEmbed(nn.Module):

    def __init__(self,
                 input_dim,
                 embed_dim,
                 n_neurons,
                 memory_length,
                 n_synch,
                 dropout=0.0,
                 normalize_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        self.pool = CTMPool(
            d_in=input_dim,
            d_out=embed_dim,
            n_neurons=n_neurons,
            memory_length=memory_length,
            n_synch=n_synch,
            side='first',
            dropout=dropout,
        )

    def forward(self, x, state=None):
        """x: (B, C, P). Returns (B, P, embed_dim), new_state."""
        if x.dim() != 3:
            raise ValueError(f"CTMEmbed expects (B, C, P), got {x.shape}")
        if self.input_bn is not None:
            x = self.input_bn(x)
        x = x.transpose(1, 2).contiguous()  # (B, P, C)
        return self.pool(x, state)
