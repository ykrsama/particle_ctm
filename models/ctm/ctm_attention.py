"""CTMAttention: multi-head attention whose Q, K, V, and output projections are
replaced by Neuron-Level Models (NLMs) plus Synchronization, following the CTM
formulation in `examples/01_mnist.py` and `models/ctm.py`.

The module is designed to be called once per outer CTM tick (LSTM-style state),
mirroring the `nn.MultiheadAttention` API otherwise.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SuperLinear, Squeeze


def _build_nlm(memory_length, n_neurons, dropout=0.0):
    """Single-SuperLinear NLM.

    Input shape:  (B, n_neurons, memory_length)
    Output shape: (B, n_neurons)
    """
    return nn.Sequential(
        SuperLinear(in_dims=memory_length, out_dims=2,
                    N=n_neurons, dropout=dropout),
        nn.GLU(),
        Squeeze(-1)
    )


def _compute_sync_first_last(activated, n_synch, side, decay_alpha, decay_beta, r):
    """Synchronization recurrence over the `first-last` neuron slice.

    Args:
        activated:    (B_flat, N) post-activations for the current tick.
        n_synch:      number of neurons selected for sync.
        side:         'first' (slice [:, :n_synch]) or 'last' (slice [:, -n_synch:]).
        decay_alpha:  (B_flat, sync_size) or None on first tick.
        decay_beta:   (B_flat, sync_size) or None on first tick.
        r:            (B_flat, sync_size) exponential decay rate per pair.

    Returns:
        synchronisation: (B_flat, sync_size)
        decay_alpha:     (B_flat, sync_size)
        decay_beta:      (B_flat, sync_size)
    """
    if side == 'first':
        selected = activated[:, :n_synch]
    elif side == 'last':
        selected = activated[:, -n_synch:]
    else:
        raise ValueError(f"unknown side: {side}")

    outer = selected.unsqueeze(2) * selected.unsqueeze(1)
    i, j = torch.triu_indices(n_synch, n_synch, device=activated.device)
    pairwise_product = outer[:, i, j]

    if decay_alpha is None or decay_beta is None:
        decay_alpha = pairwise_product
        decay_beta = torch.ones_like(pairwise_product)
    else:
        decay_alpha = r * decay_alpha + pairwise_product
        decay_beta = r * decay_beta + 1.0

    synchronisation = decay_alpha / torch.sqrt(decay_beta)
    return synchronisation, decay_alpha, decay_beta


class SwiGLU(nn.Module):
    """Stateless Llama-style FFN (no bias): w3(dropout(SiLU(w1(x)) * w2(x)))."""

    def __init__(self, d, d_ff, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


class CTMAttention(nn.Module):
    """Multi-head attention with Q/K/V/output projections built from NLMs + Sync.

    Four neuron pools (Q, K, V, O), each with its own pre-activation projection,
    sliding trace, SuperLinear-based NLM, decay parameters, and sync->embedding
    output projection.

    Per-token neurons: each sequence position carries its own private set of N
    neurons with its own trace; NLM weights are shared across positions, just as
    `nn.Linear` weights are shared across positions in standard attention.

    Trace shape per pool: (B, L_pool, N_pool, memory_length).
    Sync shape per pool:  (B, L_pool, sync_size_pool) where sync_size = n*(n+1)/2.

    Call once per outer CTM tick; pass `state` forward across ticks.
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 memory_length=4,
                 d_model_qkv=128,
                 d_model_o=128,
                 n_synch_qkv=32,
                 n_synch_o=32,
                 dropout=0.0,
                 ffn_dim=None):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if n_synch_qkv > d_model_qkv:
            raise ValueError("n_synch_qkv cannot exceed d_model_qkv")
        if n_synch_o > d_model_o:
            raise ValueError("n_synch_o cannot exceed d_model_o")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.memory_length = memory_length
        self.d_model_qkv = d_model_qkv
        self.d_model_o = d_model_o
        self.n_synch_qkv = n_synch_qkv
        self.n_synch_o = n_synch_o
        self.dropout_p = dropout

        self.sync_size_qkv = (n_synch_qkv * (n_synch_qkv + 1)) // 2
        self.sync_size_o = (n_synch_o * (n_synch_o + 1)) // 2

        # Pre-activation projections (embed_dim -> N per token, one per pool)
        self.pre_q = nn.Linear(embed_dim, d_model_qkv)
        self.pre_k = nn.Linear(embed_dim, d_model_qkv)
        self.pre_v = nn.Linear(embed_dim, d_model_qkv)
        self.pre_o = nn.Linear(embed_dim, d_model_o)

        # NLM trace processors, one per pool
        self.nlm_q = _build_nlm(memory_length, d_model_qkv, dropout)
        self.nlm_k = _build_nlm(memory_length, d_model_qkv, dropout)
        self.nlm_v = _build_nlm(memory_length, d_model_qkv, dropout)
        self.nlm_o = _build_nlm(memory_length, d_model_o, dropout)

        # Sync -> embed_dim projections
        self.q_from_sync = nn.Linear(self.sync_size_qkv, embed_dim)
        self.k_from_sync = nn.Linear(self.sync_size_qkv, embed_dim)
        self.v_from_sync = nn.Linear(self.sync_size_qkv, embed_dim)
        self.o_from_sync = nn.Linear(self.sync_size_o, embed_dim)

        # Feedback from previous tick's out into Q/K/V (zero-init no-ops)
        # A: residual into the query/key/value inputs
        self.prev_to_q = nn.Linear(embed_dim, embed_dim)
        self.prev_to_k = nn.Linear(embed_dim, embed_dim)
        self.prev_to_v = nn.Linear(embed_dim, embed_dim)
        # B: injection into the sync slice of activated (side-aware in _project_pool)
        self.prev_to_sync_q = nn.Linear(embed_dim, n_synch_qkv)
        self.prev_to_sync_k = nn.Linear(embed_dim, n_synch_qkv)
        self.prev_to_sync_v = nn.Linear(embed_dim, n_synch_qkv)
        for m in (self.prev_to_q, self.prev_to_k, self.prev_to_v,
                  self.prev_to_sync_q, self.prev_to_sync_k, self.prev_to_sync_v):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # LayerNorm on the O-pool sync projection
        self.out_norm = nn.LayerNorm(embed_dim)

        # SwiGLU FFN ("world-model knowledge base"); d_ff = 4 * d
        if ffn_dim is None:
            ffn_dim = 4 * embed_dim
        self.ffn_dim = ffn_dim
        self.ffn = SwiGLU(embed_dim, ffn_dim, dropout=dropout)
        nn.init.zeros_(self.ffn.w3.weight)  # zero-init down-proj => no-op at start
        # Normalise the post-FFN output before it is fed back as prev_out to
        # the next tick; without this, FFN residual magnitudes compound across
        # ticks via the prev_out -> Q/K/V/sync loopback.
        self.post_ffn_norm = nn.LayerNorm(embed_dim)

        # Learnable initial traces, per pool: (N_pool, memory_length)
        self._register_start_trace('q', d_model_qkv)
        self._register_start_trace('k', d_model_qkv)
        self._register_start_trace('v', d_model_qkv)
        self._register_start_trace('o', d_model_o)

        # Learnable decay parameters per pool: clamped & exponentiated to give r.
        # Init at 1.0 (r = exp(-1) ≈ 0.37) instead of 0.0 (r = 1, unbounded
        # accumulation) so the sync recurrence has real decay from the start.
        self.decay_params_q = nn.Parameter(torch.full((self.sync_size_qkv,), 1.0))
        self.decay_params_k = nn.Parameter(torch.full((self.sync_size_qkv,), 1.0))
        self.decay_params_v = nn.Parameter(torch.full((self.sync_size_qkv,), 1.0))
        self.decay_params_o = nn.Parameter(torch.full((self.sync_size_o,), 1.0))

    def _register_start_trace(self, name, n_neurons):
        bound = math.sqrt(1.0 / (n_neurons + self.memory_length))
        trace = torch.empty(n_neurons, self.memory_length).uniform_(-bound, bound)
        self.register_parameter(f'start_trace_{name}', nn.Parameter(trace))

    def _init_state(self, B, L_q, L_kv, device, dtype):
        def expand_trace(start_trace, L):
            return start_trace.to(device=device, dtype=dtype) \
                .unsqueeze(0).unsqueeze(0) \
                .expand(B, L, -1, -1).contiguous()

        return {
            'trace_q': expand_trace(self.start_trace_q, L_q),
            'trace_k': expand_trace(self.start_trace_k, L_kv),
            'trace_v': expand_trace(self.start_trace_v, L_kv),
            'trace_o': expand_trace(self.start_trace_o, L_q),
            'decay_alpha_q': None, 'decay_beta_q': None,
            'decay_alpha_k': None, 'decay_beta_k': None,
            'decay_alpha_v': None, 'decay_beta_v': None,
            'decay_alpha_o': None, 'decay_beta_o': None,
            'prev_out': None,
        }

    def _project_pool(self, x, pre_proj, nlm, trace, n_synch, side,
                      decay_alpha, decay_beta, decay_params, out_proj,
                      inject_to_sync=None):
        """Run one pool's full pipeline for one tick.

        x:        (B, L, embed_dim) input for this pool's pre-projection.
        inject_to_sync: optional (B*L, n_synch) tensor added to the slice of
                  `activated` that the sync recurrence reads — the first
                  n_synch neurons when side='first', the last n_synch when
                  side='last'.
        Returns:  (projection, new_trace, new_alpha, new_beta) where projection
                  has shape (B, L, embed_dim).
        """
        B, L, _ = x.shape
        N = trace.shape[2]

        # 1. Pre-activation per token (B, L, N)
        pre = pre_proj(x)

        # 2. Slide trace window and append new pre-activation
        new_trace = torch.cat((trace[..., 1:], pre.unsqueeze(-1)), dim=-1)

        # 3. NLM activation: flatten (B, L) -> batch dim for SuperLinear
        nlm_in = new_trace.reshape(B * L, N, self.memory_length)
        activated = nlm(nlm_in)  # (B*L, N)

        # 3b. Optional injection into the sync slice (out-of-place)
        if inject_to_sync is not None:
            if side == 'first':
                activated = activated + F.pad(inject_to_sync, (0, N - n_synch))
            else:  # 'last'
                activated = activated + F.pad(inject_to_sync, (N - n_synch, 0))

        # 4. Decay rate r broadcast over the per-token batch
        sync_size = decay_params.shape[0]
        r = torch.exp(-decay_params.clamp(0.0, 15.0)).unsqueeze(0).expand(B * L, sync_size)

        # 5. Sync recurrence on flattened batch
        if decay_alpha is not None:
            decay_alpha_flat = decay_alpha.reshape(B * L, sync_size)
            decay_beta_flat = decay_beta.reshape(B * L, sync_size)
        else:
            decay_alpha_flat = None
            decay_beta_flat = None

        sync_flat, new_alpha_flat, new_beta_flat = _compute_sync_first_last(
            activated, n_synch, side, decay_alpha_flat, decay_beta_flat, r)

        # 6. Project sync -> embed_dim and reshape back to (B, L, embed_dim)
        projection = out_proj(sync_flat).reshape(B, L, self.embed_dim)
        new_alpha = new_alpha_flat.reshape(B, L, sync_size)
        new_beta = new_beta_flat.reshape(B, L, sync_size)
        return projection, new_trace, new_alpha, new_beta

    def forward(self, query, key, value, state=None, need_weights=True,
                attn_mask=None, key_padding_mask=None):
        """Run one outer-tick attention step.

        Args:
            query:    (B, L_q, embed_dim)
            key:      (B, L_kv, embed_dim)
            value:    (B, L_kv, embed_dim)
            state:    dict returned by a previous call, or None to init.
            attn_mask: optional (L_q, L_kv) or (B*num_heads, L_q, L_kv) additive mask.
            key_padding_mask: optional (B, L_kv) bool mask; True positions are masked.

        Returns:
            attn_output:  (B, L_q, embed_dim)
            attn_weights: (B, num_heads, L_q, L_kv) or None if need_weights=False
            new_state:    dict carrying traces + sync recurrence forward
        """
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("query, key, value must be 3D: (B, L, embed_dim)")
        if query.size(-1) != self.embed_dim:
            raise ValueError(f"query last dim {query.size(-1)} != embed_dim {self.embed_dim}")

        B, L_q, _ = query.shape
        L_kv = key.size(1)
        if value.size(1) != L_kv:
            raise ValueError("key and value must have the same sequence length")

        if state is None:
            state = self._init_state(B, L_q, L_kv, query.device, query.dtype)

        # prev-tick out feedback into Q/K/V (None on first tick)
        prev_out = state.get('prev_out')
        # prev_out has shape (B, L_q, embed_dim); apply to K/V only when
        # sequence lengths match (true for self-attention, the only caller).
        prev_matches_kv = (prev_out is not None
                           and prev_out.size(1) == L_kv)

        if prev_out is not None:
            q_in = query + self.prev_to_q(prev_out)
            inj_q = self.prev_to_sync_q(prev_out).reshape(B * L_q, self.n_synch_qkv)
        else:
            q_in = query
            inj_q = None

        if prev_matches_kv:
            k_in = key + self.prev_to_k(prev_out)
            v_in = value + self.prev_to_v(prev_out)
            inj_k = self.prev_to_sync_k(prev_out).reshape(B * L_kv, self.n_synch_qkv)
            inj_v = self.prev_to_sync_v(prev_out).reshape(B * L_kv, self.n_synch_qkv)
        else:
            k_in, v_in = key, value
            inj_k = inj_v = None

        Q, trace_q, alpha_q, beta_q = self._project_pool(
            q_in, self.pre_q, self.nlm_q, state['trace_q'],
            self.n_synch_qkv, 'first',
            state['decay_alpha_q'], state['decay_beta_q'],
            self.decay_params_q, self.q_from_sync,
            inject_to_sync=inj_q)

        K, trace_k, alpha_k, beta_k = self._project_pool(
            k_in, self.pre_k, self.nlm_k, state['trace_k'],
            self.n_synch_qkv, 'first',
            state['decay_alpha_k'], state['decay_beta_k'],
            self.decay_params_k, self.k_from_sync,
            inject_to_sync=inj_k)

        V, trace_v, alpha_v, beta_v = self._project_pool(
            v_in, self.pre_v, self.nlm_v, state['trace_v'],
            self.n_synch_qkv, 'last',
            state['decay_alpha_v'], state['decay_beta_v'],
            self.decay_params_v, self.v_from_sync,
            inject_to_sync=inj_v)

        # Multi-head split: (B, L, embed_dim) -> (B, num_heads, L, head_dim)
        def split_heads(t, L):
            return t.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        Qh = split_heads(Q, L_q)
        Kh = split_heads(K, L_kv)
        Vh = split_heads(V, L_kv)

        # Build additive float mask combining attn_mask and key_padding_mask
        merged_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                merged_mask = torch.zeros_like(attn_mask, dtype=Q.dtype)
                merged_mask = merged_mask.masked_fill(attn_mask, float('-inf'))
            else:
                merged_mask = attn_mask.to(Q.dtype)
        if key_padding_mask is not None:
            kp = key_padding_mask.to(torch.bool)  # (B, L_kv)
            kp_add = torch.zeros(B, 1, 1, L_kv, device=Q.device, dtype=Q.dtype)
            kp_add = kp_add.masked_fill(kp.view(B, 1, 1, L_kv), float('-inf'))
            if merged_mask is None:
                merged_mask = kp_add
            else:
                if merged_mask.dim() == 2:
                    merged_mask = merged_mask.unsqueeze(0).unsqueeze(0)
                merged_mask = merged_mask + kp_add

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale  # (B, H, L_q, L_kv)
        if merged_mask is not None:
            attn_logits = attn_logits + merged_mask
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_dropped = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        attn_out = torch.matmul(attn_dropped, Vh)  # (B, H, L_q, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, L_q, self.embed_dim)

        # Output pool: attn_out enters via O-pool's NLM+Sync only — no
        # direct residual or sync-slot shortcut. The next tick's feedback
        # loop (prev_out -> QKV) carries the attention signal forward instead.
        out, trace_o, alpha_o, beta_o = self._project_pool(
            attn_out, self.pre_o, self.nlm_o, state['trace_o'],
            self.n_synch_o, 'first',
            state['decay_alpha_o'], state['decay_beta_o'],
            self.decay_params_o, self.o_from_sync)

        out = self.out_norm(out)
        out = self.post_ffn_norm(out + self.ffn(out))  # SwiGLU FFN + residual + norm

        new_state = {
            'trace_q': trace_q, 'trace_k': trace_k,
            'trace_v': trace_v, 'trace_o': trace_o,
            'decay_alpha_q': alpha_q, 'decay_beta_q': beta_q,
            'decay_alpha_k': alpha_k, 'decay_beta_k': beta_k,
            'decay_alpha_v': alpha_v, 'decay_beta_v': beta_v,
            'decay_alpha_o': alpha_o, 'decay_beta_o': beta_o,
            'prev_out': out,
        }

        if not need_weights:
            attn_weights = None
        return out, attn_weights, new_state
