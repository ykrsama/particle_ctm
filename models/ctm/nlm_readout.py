"""NLMReadout: the model's single NLM-style projection primitive.

Mathematically equivalent to `SuperLinear(in_dims=in_dim, out_dims=2,
N=out_dim) -> GLU -> Squeeze`: each of the `out_dim` output neurons has its
own private `(in_dim, 2)` weight matrix; the 2 channels are gated through GLU
to produce one post-activation per neuron.

Why a custom einsum instead of `SuperLinear`: `SuperLinear` expects input
shape `(B, N, in_dim)`, which would require materializing an
`(B*L, out_dim, in_dim)` broadcast tensor in our readouts. For the pair-bias
readout that broadcast is `(B * num_pairs, num_heads, pair_embed_dim + ssize)`
which is huge for `num_pairs ~ P*(P+1)/2 ~ 8k`. The `einsum('...m,mhd->...dh')`
formulation below keeps the input as `(B, L, in_dim)` and only allocates the
output `(B, L, out_dim, 2)`, which is fine.

This is the only NLM-style readout primitive used outside `GlobalCTMPool`;
every projection that is not the global pool's trace NLM (particle_embed,
pair_embed, pair_bias, Q, K, V, O, head, and the pool's `pre_proj`) is an
`NLMReadout` instance.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLMReadout(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim and out_dim must be positive, got {in_dim}, {out_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim

        bound = 1.0 / math.sqrt(in_dim + 2)
        self.w = nn.Parameter(
            torch.empty(in_dim, 2, out_dim).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(1, out_dim, 2))
        self.T = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"NLMReadout last dim {x.size(-1)} != in_dim {self.in_dim}")
        x = self.dropout(x)
        # einsum keeps leading dims intact: (..., in_dim) -> (..., out_dim, 2)
        pre = torch.einsum('...m,mhd->...dh', x, self.w) + self.b
        pre = pre / self.T
        out = F.glu(pre, dim=-1).squeeze(-1)
        return out
