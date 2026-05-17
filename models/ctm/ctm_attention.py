"""CTMAttention: multi-head attention whose Q/K/V/O projections are each a
`CTMPool` (Linear synapse -> trace -> NLM -> Synchronization -> sync->embed).

The attention body (split_heads / scaled dot-product / softmax / mask merge)
is a standard Transformer attention; only the four projections are CTM-style.

Call once per outer CTM tick. Pass `state` forward to thread the per-pool
traces and decay recurrences across ticks. State layout:
    state = {'q': pool_state, 'k': pool_state, 'v': pool_state, 'o': pool_state}
where each `pool_state` is the dict returned by `CTMPool.forward`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm_pool import CTMPool


class CTMAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 memory_length=4,
                 d_model_qkv=128,
                 d_model_o=128,
                 n_synch_qkv=32,
                 n_synch_o=32,
                 dropout=0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        self.pool_q = CTMPool(embed_dim, embed_dim, d_model_qkv, memory_length,
                              n_synch_qkv, side='first', dropout=dropout)
        self.pool_k = CTMPool(embed_dim, embed_dim, d_model_qkv, memory_length,
                              n_synch_qkv, side='first', dropout=dropout)
        self.pool_v = CTMPool(embed_dim, embed_dim, d_model_qkv, memory_length,
                              n_synch_qkv, side='last', dropout=dropout)
        self.pool_o = CTMPool(embed_dim, embed_dim, d_model_o, memory_length,
                              n_synch_o, side='first', dropout=dropout)

    def forward(self, query, key, value, state=None, need_weights=True,
                attn_mask=None, key_padding_mask=None):
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("query, key, value must be 3D: (B, L, embed_dim)")
        if query.size(-1) != self.embed_dim:
            raise ValueError(f"query last dim {query.size(-1)} != embed_dim {self.embed_dim}")

        B, L_q, _ = query.shape
        L_kv = key.size(1)
        if value.size(1) != L_kv:
            raise ValueError("key and value must have the same sequence length")

        if state is None:
            state = {'q': None, 'k': None, 'v': None, 'o': None}

        Q, q_state = self.pool_q(query, state['q'])
        K, k_state = self.pool_k(key,   state['k'])
        V, v_state = self.pool_v(value, state['v'])

        def split_heads(t, L):
            return t.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        Qh = split_heads(Q, L_q)
        Kh = split_heads(K, L_kv)
        Vh = split_heads(V, L_kv)

        merged_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                merged_mask = torch.zeros_like(attn_mask, dtype=Q.dtype)
                merged_mask = merged_mask.masked_fill(attn_mask, float('-inf'))
            else:
                merged_mask = attn_mask.to(Q.dtype)
        if key_padding_mask is not None:
            kp = key_padding_mask.to(torch.bool)
            kp_add = torch.zeros(B, 1, 1, L_kv, device=Q.device, dtype=Q.dtype)
            kp_add = kp_add.masked_fill(kp.view(B, 1, 1, L_kv), float('-inf'))
            if merged_mask is None:
                merged_mask = kp_add
            else:
                if merged_mask.dim() == 2:
                    merged_mask = merged_mask.unsqueeze(0).unsqueeze(0)
                merged_mask = merged_mask + kp_add

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale
        if merged_mask is not None:
            attn_logits = attn_logits + merged_mask
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_dropped = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        attn_out = torch.matmul(attn_dropped, Vh)
        attn_out = attn_out.transpose(1, 2).reshape(B, L_q, self.embed_dim)

        out, o_state = self.pool_o(attn_out, state['o'])

        new_state = {'q': q_state, 'k': k_state, 'v': v_state, 'o': o_state}
        if not need_weights:
            attn_weights = None
        return out, attn_weights, new_state
