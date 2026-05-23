"""ParticleSynapse: ParticleTransformer-style tokenisation + pair embedding +
class token, with attention reduced to a single stateless `nn.MultiheadAttention`
block iterated for `iterations` outer ticks. Each tick's FFN is replaced by a
`SynapseFFN`: a SwiGLU-shaped FFN whose up-branch nonlinearity is a Neuron-Level
Model over a sliding pre-activation trace, with per-token synchronisation on the
activated value. The cls-position sync vector drives the output head.

Per outer tick:
    seq_t   = embed_seq + prev_to_emb(prev_out_{t-1})      (zero on first tick)
    attn_t  = MHA(seq_t, seq_t, seq_t, attn_mask=pair_bias,
                  key_padding_mask=kp_mask)
    y, state, sync_all = synapse(attn_t, state)
    logits  = head(sync_all[:, 0])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from .part_layers import Embed, PairEmbed, SequenceTrimmer, trunc_normal_
from .particle_ctm import compute_normalized_entropy, get_loss, calculate_accuracy
from .synapse_block import SynapseFFN


_STATE_KEYS = ('trace', 'decay_alpha_o', 'decay_beta_o', 'prev_out')


def _flatten_state(state):
    return tuple(state[k] for k in _STATE_KEYS)


def _unflatten_state(flat):
    return dict(zip(_STATE_KEYS, flat))


class ParticleSynapse(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes=10,
                 # ParT-style embedding/pair config
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 embed_dims=(128, 512, 128),
                 pair_embed_dims=(64, 64, 64),
                 use_pre_activation_pair=False,
                 # Attention
                 num_heads=8,
                 iterations=8,
                 dropout=0.0,
                 # Synapse FFN
                 d_ff=None,
                 memory_length=10,
                 memory_hidden_dims=32,
                 n_synch_out=32,
                 # misc
                 trim=True,
                 fc_params=(),
                 activation='gelu',
                 use_grad_checkpoint=True):
        super().__init__()

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        self.embed_dim = embed_dim
        self.iterations = iterations
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.use_grad_checkpoint = use_grad_checkpoint

        self.trimmer = SequenceTrimmer(enabled=trim)
        self.embed = Embed(input_dim, list(embed_dims), activation=activation) \
            if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim,
            list(pair_embed_dims) + [num_heads],
            remove_self_pair=False,
            use_pre_activation_pair=use_pre_activation_pair,
        ) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.synapse = SynapseFFN(
            embed_dim=embed_dim,
            d_ff=d_ff,
            memory_length=memory_length,
            memory_hidden_dims=memory_hidden_dims,
            n_synch_out=n_synch_out,
            dropout=dropout,
        )

        self.prev_to_emb = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.prev_to_emb.weight)
        nn.init.zeros_(self.prev_to_emb.bias)

        sync_size_out = self.synapse.sync_size_out

        if fc_params:
            fcs = []
            in_dim = sync_size_out
            for out_dim, drop in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.head = nn.Sequential(*fcs)
        else:
            self.head = nn.Linear(sync_size_out, num_classes)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

    def _build_attn_bias(self, v, mask, P):
        """pair_embed(v) → (B, num_heads, P, P); pad with zero cls row/col.

        Returns (B*num_heads, 1+P, 1+P) additive bias, the shape
        `nn.MultiheadAttention` expects for its `attn_mask`.
        """
        if self.pair_embed is None:
            return None
        bias = self.pair_embed(v, None)  # (B, num_heads, P, P)
        B = bias.size(0)
        out = bias.new_zeros(B, self.num_heads, P + 1, P + 1)
        out[:, :, 1:, 1:] = bias
        return out.reshape(B * self.num_heads, P + 1, P + 1)

    def forward(self, x, v=None, mask=None, track=False):
        """
        x:    (B, C, P) particle features
        v:    (B, 4, P) (px, py, pz, energy) — fed to PairEmbed for the bias
        mask: (B, 1, P) boolean / 0-1; 1 = real particle, 0 = padding
        """
        x, v, mask, _ = self.trimmer(x, v, mask, None)
        B, _, P = x.shape
        device = x.device

        tokens = self.embed(x)
        if tokens.dim() == 3 and tokens.size(0) == P and tokens.size(1) == B:
            tokens = tokens.permute(1, 0, 2).contiguous()
        tokens = tokens.masked_fill(~mask.squeeze(1).unsqueeze(-1), 0.0)

        cls = self.cls_token.expand(B, 1, self.embed_dim)
        seq = torch.cat([cls, tokens], dim=1)  # (B, 1+P, embed_dim)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        particle_pad = ~mask.squeeze(1).bool()
        kp_bool = torch.cat([cls_mask, particle_pad], dim=1)  # (B, 1+P)
        # Use a float additive mask to match attn_mask dtype (MHA deprecates
        # mismatched mask dtypes).
        kp_mask = torch.zeros(B, P + 1, dtype=seq.dtype, device=device) \
            .masked_fill(kp_bool, float('-inf'))

        attn_bias = self._build_attn_bias(v, mask, P) if v is not None else None

        T = self.iterations
        predictions = torch.empty(B, self.num_classes, T, device=device, dtype=seq.dtype)
        certainties = torch.empty(B, 2, T, device=device, dtype=seq.dtype)

        attn_history = [] if track else None
        token_history = [] if track else None
        sync_history = [] if track else None

        state = None
        do_ckpt = self.use_grad_checkpoint and self.training and not track

        def _tick_fn(seq_in, *state_flat):
            st = _unflatten_state(state_flat)
            seq_t = seq_in if st['prev_out'] is None else seq_in + self.prev_to_emb(st['prev_out'])
            attn_out, _w = self.attention(
                seq_t, seq_t, seq_t,
                attn_mask=attn_bias,
                key_padding_mask=kp_mask,
                need_weights=False,
            )
            y_, new_st, sync_all_ = self.synapse(attn_out, state=st)
            return (y_, sync_all_) + _flatten_state(new_st)

        for t in range(T):
            if do_ckpt and state is not None:
                ret = _ckpt(_tick_fn, seq, *_flatten_state(state),
                            use_reentrant=False)
                y, sync_all, new_state_flat = ret[0], ret[1], ret[2:]
                state = _unflatten_state(new_state_flat)
                attn_w = None
            else:
                if state is not None and state['prev_out'] is not None:
                    seq_t = seq + self.prev_to_emb(state['prev_out'])
                else:
                    seq_t = seq
                attn_out, attn_w = self.attention(
                    seq_t, seq_t, seq_t,
                    attn_mask=attn_bias,
                    key_padding_mask=kp_mask,
                    need_weights=track,
                    average_attn_weights=False,
                )
                y, state, sync_all = self.synapse(attn_out, state=state)

            sync_cls = sync_all[:, 0]
            logits = self.head(sync_cls)
            ne = compute_normalized_entropy(logits)
            certainty = torch.stack((ne, 1 - ne), dim=-1)

            predictions[..., t] = logits
            certainties[..., t] = certainty

            if track:
                attn_history.append(attn_w)
                token_history.append(y.detach().cpu().numpy())
                sync_history.append(sync_cls.detach().cpu().numpy())

        if track:
            return predictions, certainties, attn_history, token_history, sync_history
        return predictions, certainties


__all__ = [
    'ParticleSynapse',
    'compute_normalized_entropy',
    'get_loss',
    'calculate_accuracy',
]
