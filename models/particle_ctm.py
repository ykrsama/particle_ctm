"""ParticleCTM: ParticleTransformer-style tokenisation + pair embedding +
class token, but with the multiple particle/class attention blocks replaced by
a single `CTMAttention` block iterated for `iterations` outer ticks.

Per outer tick:
    1. self-attention over (cls + particle) tokens with pair-embedding bias
       (zero bias on cls row/col).
    2. take the cls slot of the output → linear head → logits.
    3. compute normalised-entropy certainty.

State (`trace_*`, `decay_alpha_*`, `decay_beta_*`) is threaded across ticks by
CTMAttention itself; the rest of the module is stateless within a forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from .ctm.ctm_attention import CTMAttention
from .part_layers import Embed, PairEmbed, SequenceTrimmer, trunc_normal_


# Schema for CTMAttention state dict. Used to flatten/unflatten so the dict can
# pass through torch.utils.checkpoint (which expects positional tensor args).
_STATE_KEYS = (
    'trace_q', 'trace_k', 'trace_v', 'trace_o',
    'decay_alpha_q', 'decay_beta_q',
    'decay_alpha_k', 'decay_beta_k',
    'decay_alpha_v', 'decay_beta_v',
    'decay_alpha_o', 'decay_beta_o',
    'prev_out',
)


def _flatten_state(state):
    return tuple(state[k] for k in _STATE_KEYS)


def _unflatten_state(flat):
    return dict(zip(_STATE_KEYS, flat))


def compute_normalized_entropy(logits, reduction='mean'):
    preds = F.softmax(logits, dim=-1)
    log_preds = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(preds * log_preds, dim=-1)
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32, device=logits.device))
    ne = entropy / max_entropy
    if len(logits.shape) > 2 and reduction == 'mean':
        ne = ne.flatten(1).mean(-1)
    return ne


class ParticleCTM(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes=10,
                 # ParT-style embedding/pair config
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 embed_dims=(128, 512, 128),
                 pair_embed_dims=(64, 64, 64),
                 use_pre_activation_pair=False,
                 # CTMAttention config
                 num_heads=8,
                 iterations=8,
                 memory_length=10,
                 memory_hidden_dims=None,
                 d_model_qkv=128,
                 d_model_o=128,
                 n_synch_qkv=32,
                 n_synch_o=32,
                 dropout=0.0,
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

        self.ctm_attention = CTMAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            memory_length=memory_length,
            memory_hidden_dims=memory_hidden_dims,
            d_model_qkv=d_model_qkv,
            d_model_o=d_model_o,
            n_synch_qkv=n_synch_qkv,
            n_synch_o=n_synch_o,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.head = nn.Sequential(*fcs)
        else:
            self.head = nn.Linear(embed_dim, num_classes)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

    def _build_attn_bias(self, v, mask, P):
        """pair_embed(v) → (B, num_heads, P, P); pad with zero cls row/col.

        Returns (B, num_heads, 1+P, 1+P) additive attention bias, which
        broadcasts with the (B, num_heads, L_q, L_kv) logits inside CTMAttention.
        """
        if self.pair_embed is None:
            return None
        bias = self.pair_embed(v, None)  # (B, num_heads, P, P)
        B = bias.size(0)
        out = bias.new_zeros(B, self.num_heads, P + 1, P + 1)
        out[:, :, 1:, 1:] = bias
        return out

    def forward(self, x, v=None, mask=None, track=False):
        """
        x:    (B, C, P) particle features
        v:    (B, 4, P) (px, py, pz, energy) — fed to PairEmbed for the bias
        mask: (B, 1, P) boolean / 0-1; 1 = real particle, 0 = padding
        """
        # 1. trim padding columns
        x, v, mask, _ = self.trimmer(x, v, mask, None)
        B, _, P = x.shape
        device = x.device

        # 2. particle embedding: (P, B, embed_dim) then back to (B, P, embed_dim)
        tokens = self.embed(x)
        if tokens.dim() == 3 and tokens.size(0) == P and tokens.size(1) == B:
            tokens = tokens.permute(1, 0, 2).contiguous()
        # zero out padded particles
        tokens = tokens.masked_fill(~mask.squeeze(1).unsqueeze(-1), 0.0)

        # 3. prepend cls token
        cls = self.cls_token.expand(B, 1, self.embed_dim)
        seq = torch.cat([cls, tokens], dim=1)  # (B, 1+P, embed_dim)

        # 4. padding mask for the full (1+P) sequence: cls is never padded
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        # particles: True where padded
        particle_pad = ~mask.squeeze(1).bool()
        kp_mask = torch.cat([cls_mask, particle_pad], dim=1)  # (B, 1+P)

        # 5. pair-embedding attention bias (B*num_heads, 1+P, 1+P)
        attn_mask = self._build_attn_bias(v, mask, P) if v is not None else None

        # 6. T outer ticks of CTMAttention
        T = self.iterations
        predictions = torch.empty(B, self.num_classes, T, device=device, dtype=seq.dtype)
        certainties = torch.empty(B, 2, T, device=device, dtype=seq.dtype)

        attn_history = [] if track else None
        token_history = [] if track else None
        sync_o_history = [] if track else None

        state = None
        do_ckpt = self.use_grad_checkpoint and self.training and not track

        def _tick_fn(seq_in, *state_flat):
            st = _unflatten_state(state_flat)
            out_, _w, new_st = self.ctm_attention(
                seq_in, seq_in, seq_in, state=st,
                attn_mask=attn_mask, key_padding_mask=kp_mask,
                need_weights=False,
            )
            return (out_,) + _flatten_state(new_st)

        for t in range(T):
            if do_ckpt and state is not None:
                ret = _ckpt(_tick_fn, seq, *_flatten_state(state),
                            use_reentrant=False)
                out, new_state_flat = ret[0], ret[1:]
                state = _unflatten_state(new_state_flat)
                attn_w = None
            else:
                out, attn_w, state = self.ctm_attention(
                    seq, seq, seq, state=state,
                    attn_mask=attn_mask,
                    key_padding_mask=kp_mask,
                    need_weights=track,
                )
            cls_out = self.norm(out[:, 0])
            logits = self.head(cls_out)
            ne = compute_normalized_entropy(logits)
            certainty = torch.stack((ne, 1 - ne), dim=-1)

            predictions[..., t] = logits
            certainties[..., t] = certainty

            if track:
                attn_history.append(attn_w)
                token_history.append(out.detach().cpu().numpy())
                sync_o_history.append(state['decay_alpha_o'].mean(dim=1).detach().cpu().numpy())

        if track:
            return predictions, certainties, attn_history, token_history, sync_o_history
        return predictions, certainties


def get_loss(predictions, certainties, targets, use_most_certain=True):
    """Certainty-based loss from CTM examples 01/07.

    Picks two ticks per sample (argmin-loss and argmax-certainty) and averages
    cross-entropy at those ticks.
    """
    losses = nn.CrossEntropyLoss(reduction='none')(
        predictions,
        torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1),
    )
    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1
    bi = torch.arange(predictions.size(0), device=predictions.device)
    loss = (losses[bi, loss_index_1].mean() + losses[bi, loss_index_2].mean()) / 2
    return loss, loss_index_2


def calculate_accuracy(predictions, targets, where_most_certain):
    B = predictions.size(0)
    device = predictions.device
    idx = predictions.argmax(1)[torch.arange(B, device=device), where_most_certain]
    return (targets.detach().cpu().numpy() == idx.detach().cpu().numpy()).mean()
