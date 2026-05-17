"""CTMHead: classification head built from a single CTMPool.

Replaces the traditional `Linear -> ReLU -> Dropout -> ... -> Linear(num_classes)`
MLP head. Called once per outer CTM tick on the cls slot; state is threaded
across ticks so the head's per-class sync recurrence evolves through thought.
"""

import torch
import torch.nn as nn

from .ctm_pool import CTMPool


class CTMHead(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_classes,
                 n_neurons,
                 memory_length,
                 n_synch,
                 dropout=0.0):
        super().__init__()
        self.pool = CTMPool(
            d_in=embed_dim,
            d_out=num_classes,
            n_neurons=n_neurons,
            memory_length=memory_length,
            n_synch=n_synch,
            side='first',
            dropout=dropout,
        )

    def forward(self, cls, state=None):
        """cls: (B, embed_dim). Returns (B, num_classes) logits, new_state."""
        if cls.dim() != 2:
            raise ValueError(f"CTMHead expects (B, embed_dim), got {cls.shape}")
        logits, new_state = self.pool(cls.unsqueeze(1), state)  # L=1
        return logits.squeeze(1), new_state
