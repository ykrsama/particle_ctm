"""SynapseFFN: SwiGLU-shaped FFN whose up-branch per-neuron activation is a
Neuron-Level Model over a sliding pre-activation trace.

Topology (per outer tick):

    x  (B, L, D)
     |-- w_up  : Linear(D -> d_ff)         pre = w_up(x)
     |          slide pre into trace[..., -M:]
     |          NLM (SuperLinear, one per d_ff neuron) -> activated
     |-- w_gate: Linear(D -> d_ff)         gate = SiLU(w_gate(x))
     out  = w_down(dropout(activated * gate))
     y    = LN(x + dropout(out))

The block also returns a per-token synchronisation vector built from the
activated value (first n_synch_out neurons, first-last 'first' side), to be
consumed by the model head.

State carried across ticks:
    trace         : (B, L, d_ff, memory_length)
    decay_alpha_o : (B, L, sync_size_out) or None on first tick
    decay_beta_o  : (B, L, sync_size_out) or None on first tick
    prev_out      : (B, L, D), the post-norm y of the previous tick
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm.ctm_attention import _build_nlm, _compute_sync_first_last


class SynapseFFN(nn.Module):
    def __init__(self,
                 embed_dim,
                 d_ff=None,
                 memory_length=10,
                 memory_hidden_dims=32,
                 n_synch_out=32,
                 dropout=0.0,
                 zero_init_down=True):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * embed_dim
        if n_synch_out > d_ff:
            raise ValueError(
                f"n_synch_out ({n_synch_out}) cannot exceed d_ff ({d_ff})")

        self.embed_dim = embed_dim
        self.d_ff = d_ff
        self.memory_length = memory_length
        self.n_synch_out = n_synch_out
        self.dropout_p = dropout

        self.sync_size_out = (n_synch_out * (n_synch_out + 1)) // 2

        self.w_up = nn.Linear(embed_dim, d_ff, bias=False)
        self.w_gate = nn.Linear(embed_dim, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, embed_dim, bias=False)
        if zero_init_down:
            nn.init.zeros_(self.w_down.weight)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_norm = nn.LayerNorm(embed_dim)

        self.nlm = _build_nlm(memory_length, d_ff, memory_hidden_dims, dropout)

        bound = math.sqrt(1.0 / (d_ff + memory_length))
        start_trace = torch.empty(d_ff, memory_length).uniform_(-bound, bound)
        self.start_trace = nn.Parameter(start_trace)

        self.decay_params_out = nn.Parameter(torch.zeros(self.sync_size_out))

        triu_o = torch.triu_indices(n_synch_out, n_synch_out)
        self.register_buffer('triu_i_o', triu_o[0], persistent=False)
        self.register_buffer('triu_j_o', triu_o[1], persistent=False)

    def _init_state(self, B, L, device, dtype):
        trace = self.start_trace.to(device=device, dtype=dtype) \
            .unsqueeze(0).unsqueeze(0) \
            .expand(B, L, -1, -1).contiguous()
        return {
            'trace': trace,
            'decay_alpha_o': None,
            'decay_beta_o': None,
            'prev_out': None,
        }

    def forward(self, x, state=None):
        """One outer-tick synapse FFN.

        Args:
            x:     (B, L, embed_dim) attention output for this tick.
            state: dict from previous tick, or None on first tick.

        Returns:
            y:        (B, L, embed_dim) post-norm FFN output.
            new_state: dict carrying trace + sync recurrence + prev_out.
            sync_all: (B, L, sync_size_out) per-token sync vector.
        """
        if x.dim() != 3:
            raise ValueError("x must be 3D: (B, L, embed_dim)")
        if x.size(-1) != self.embed_dim:
            raise ValueError(
                f"x last dim {x.size(-1)} != embed_dim {self.embed_dim}")

        B, L, _ = x.shape
        if state is None:
            state = self._init_state(B, L, x.device, x.dtype)

        trace = state['trace']
        alpha_o = state['decay_alpha_o']
        beta_o = state['decay_beta_o']

        pre = self.w_up(x)
        gate = F.silu(self.w_gate(x))

        new_trace = torch.cat((trace[..., 1:], pre.unsqueeze(-1)), dim=-1)

        nlm_in = new_trace.reshape(B * L, self.d_ff, self.memory_length)
        activated = self.nlm(nlm_in)
        activated = activated.reshape(B, L, self.d_ff)

        out = self.w_down(self.drop(activated * gate))
        y = self.out_norm(x + self.drop(out))

        r = torch.exp(-self.decay_params_out.clamp(0.0, 15.0)) \
            .unsqueeze(0).expand(B * L, self.sync_size_out)

        if alpha_o is not None:
            alpha_flat = alpha_o.reshape(B * L, self.sync_size_out)
            beta_flat = beta_o.reshape(B * L, self.sync_size_out)
        else:
            alpha_flat = None
            beta_flat = None

        activated_flat = activated.reshape(B * L, self.d_ff)
        sync_flat, new_alpha_flat, new_beta_flat = _compute_sync_first_last(
            activated_flat, self.n_synch_out, 'first',
            alpha_flat, beta_flat, r, self.triu_i_o, self.triu_j_o)

        sync_all = sync_flat.reshape(B, L, self.sync_size_out)
        new_alpha = new_alpha_flat.reshape(B, L, self.sync_size_out)
        new_beta = new_beta_flat.reshape(B, L, self.sync_size_out)

        new_state = {
            'trace': new_trace,
            'decay_alpha_o': new_alpha,
            'decay_beta_o': new_beta,
            'prev_out': y,
        }
        return y, new_state, sync_all
