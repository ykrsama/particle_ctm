"""CTMPairEmbed: per-pair attention-bias generator built from a single CTMPool.

Replaces the traditional ParT-style `PairEmbed` (`BatchNorm + (Conv1d 1x1 +
BN + GELU)*N`). Only the symmetric, no-extra-features path is supported here
(pair_extra_dim == 0, pairwise_lv_dim <= 5), which matches the current
default config and avoids carrying over PairEmbed's two extra branches.

The pool's `n_neurons` private neurons exist per pair; trace + sync state is
threaded across outer CTM ticks.
"""

import torch
import torch.nn as nn

from ..part_layers import pairwise_lv_fts
from .ctm_pool import CTMPool


class CTMPairEmbed(nn.Module):

    def __init__(self,
                 pairwise_lv_dim,
                 pairwise_input_dim,
                 num_heads,
                 n_neurons,
                 memory_length,
                 n_synch,
                 dropout=0.0,
                 remove_self_pair=False,
                 normalize_input=True,
                 eps=1e-8):
        super().__init__()
        if pairwise_input_dim != 0:
            raise NotImplementedError(
                "CTMPairEmbed only supports pairwise_input_dim == 0 (no extra "
                "uu features). The traditional PairEmbed's 'concat'/'sum' "
                "branches are out of scope.")
        if pairwise_lv_dim <= 0 or pairwise_lv_dim > 5:
            raise ValueError(
                f"pairwise_lv_dim must be in 1..5 (got {pairwise_lv_dim})")

        self.pairwise_lv_dim = pairwise_lv_dim
        self.num_heads = num_heads
        self.remove_self_pair = remove_self_pair
        self.eps = eps

        self.input_bn = nn.BatchNorm1d(pairwise_lv_dim) if normalize_input else None
        self.pool = CTMPool(
            d_in=pairwise_lv_dim,
            d_out=num_heads,
            n_neurons=n_neurons,
            memory_length=memory_length,
            n_synch=n_synch,
            side='first',
            dropout=dropout,
        )

    def _compute_pair_fts(self, v, seq_len):
        """v: (B, 4, P) four-momentum. Returns (B, pairwise_lv_dim, num_pairs)
        plus the lower-triangular indices (i, j) used for the scatter."""
        i, j = torch.tril_indices(
            seq_len, seq_len,
            offset=-1 if self.remove_self_pair else 0,
            device=v.device,
        )
        v_expanded = v.unsqueeze(-1).expand(-1, -1, -1, seq_len)
        vi = v_expanded[:, :, i, j]
        vj = v_expanded[:, :, j, i]
        feats = pairwise_lv_fts(vi, vj, num_outputs=self.pairwise_lv_dim, eps=self.eps)
        return feats, i, j

    def forward(self, v, seq_len, state=None):
        """v: (B, 4, P). Returns (B, num_heads, P, P) bias, new_state."""
        if v.dim() != 3 or v.size(1) != 4:
            raise ValueError(f"CTMPairEmbed expects v of shape (B, 4, P), got {v.shape}")

        with torch.no_grad():
            feats, i, j = self._compute_pair_fts(v, seq_len)
        if self.input_bn is not None:
            feats = self.input_bn(feats)

        # (B, pair_lv_dim, num_pairs) -> (B, num_pairs, pair_lv_dim)
        pair_in = feats.transpose(1, 2).contiguous()
        elements, new_state = self.pool(pair_in, state)
        # elements: (B, num_pairs, num_heads)
        elements = elements.transpose(1, 2).contiguous()  # (B, num_heads, num_pairs)

        B = elements.size(0)
        y = elements.new_zeros(B, self.num_heads, seq_len, seq_len)
        y[:, :, i, j] = elements
        y[:, :, j, i] = elements
        return y, new_state
