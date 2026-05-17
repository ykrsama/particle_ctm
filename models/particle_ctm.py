"""ParticleCTM: every neuron pool is now NLM + Synchronization.

Four CTM pools are threaded through the same outer-tick loop:
    - CTMEmbed:     per-particle embedding (replaces ParT MLP).
    - CTMPairEmbed: per-pair attention bias (replaces ParT Conv1d stack).
    - CTMAttention: Q/K/V/O projections (already NLM+Sync).
    - CTMHead:      class logits (replaces the linear classification MLP).

Per outer tick:
    1. CTMEmbed(x) -> particle tokens (state threaded across ticks).
    2. CTMPairEmbed(v) -> (B, num_heads, 1+P, 1+P) additive bias.
    3. CTMAttention over (cls + particle) tokens.
    4. CTMHead on the cls slot -> logits.
    5. Normalised-entropy certainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm.ctm_attention import CTMAttention
from .ctm.ctm_embed import CTMEmbed
from .ctm.ctm_head import CTMHead
from .ctm.ctm_pair_embed import CTMPairEmbed
from .part_layers import SequenceTrimmer, trunc_normal_


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
                 # Pair-embed config
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 # Per-pool widths
                 embed_dim=128,
                 d_model_embed=128,
                 n_synch_embed=32,
                 d_model_pair=32,
                 n_synch_pair=8,
                 d_model_head=128,
                 n_synch_head=32,
                 # CTMAttention config
                 num_heads=8,
                 iterations=8,
                 memory_length=10,
                 d_model_qkv=128,
                 d_model_o=128,
                 n_synch_qkv=32,
                 n_synch_o=32,
                 dropout=0.0,
                 # misc
                 trim=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.iterations = iterations
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.trimmer = SequenceTrimmer(enabled=trim)

        self.embed = CTMEmbed(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_neurons=d_model_embed,
            memory_length=memory_length,
            n_synch=n_synch_embed,
            dropout=dropout,
        )

        self.pair_embed = CTMPairEmbed(
            pairwise_lv_dim=pair_input_dim,
            pairwise_input_dim=pair_extra_dim,
            num_heads=num_heads,
            n_neurons=d_model_pair,
            memory_length=memory_length,
            n_synch=n_synch_pair,
            dropout=dropout,
        ) if pair_input_dim > 0 else None

        self.ctm_attention = CTMAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            memory_length=memory_length,
            d_model_qkv=d_model_qkv,
            d_model_o=d_model_o,
            n_synch_qkv=n_synch_qkv,
            n_synch_o=n_synch_o,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = CTMHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            n_neurons=d_model_head,
            memory_length=memory_length,
            n_synch=n_synch_head,
            dropout=dropout,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

    def _expand_attn_bias(self, pair_bias):
        """Add a zero cls row/col to a (B, num_heads, P, P) pair bias and
        return (B, num_heads, 1+P, 1+P)."""
        B, H, P, _ = pair_bias.shape
        out = pair_bias.new_zeros(B, H, P + 1, P + 1)
        out[:, :, 1:, 1:] = pair_bias
        return out

    def forward(self, x, v=None, mask=None, track=False):
        """x: (B, C, P); v: (B, 4, P); mask: (B, 1, P)."""
        x, v, mask, _ = self.trimmer(x, v, mask, None)
        B, _, P = x.shape
        device = x.device

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        particle_pad = ~mask.squeeze(1).bool()
        kp_mask = torch.cat([cls_mask, particle_pad], dim=1)  # (B, 1+P)
        particle_keep = (~particle_pad).unsqueeze(-1)         # (B, P, 1)

        T = self.iterations
        predictions = torch.empty(B, self.num_classes, T, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, T, device=device, dtype=x.dtype)

        attn_history = [] if track else None
        token_history = [] if track else None
        sync_o_history = [] if track else None

        embed_state = None
        pair_state = None
        attn_state = None
        head_state = None

        for t in range(T):
            tokens, embed_state = self.embed(x, embed_state)        # (B, P, embed_dim)
            tokens = tokens.masked_fill(~particle_keep, 0.0)
            cls = self.cls_token.expand(B, 1, self.embed_dim)
            seq = torch.cat([cls, tokens], dim=1)                   # (B, 1+P, embed_dim)

            if self.pair_embed is not None and v is not None:
                pair_bias, pair_state = self.pair_embed(v, P, pair_state)
                attn_mask = self._expand_attn_bias(pair_bias)
            else:
                attn_mask = None

            out, attn_w, attn_state = self.ctm_attention(
                seq, seq, seq, state=attn_state,
                attn_mask=attn_mask,
                key_padding_mask=kp_mask,
                need_weights=track,
            )
            cls_out = self.norm(out[:, 0])
            logits, head_state = self.head(cls_out, head_state)
            ne = compute_normalized_entropy(logits)
            certainty = torch.stack((ne, 1 - ne), dim=-1)

            predictions[..., t] = logits
            certainties[..., t] = certainty

            if track:
                attn_history.append(attn_w)
                token_history.append(out.detach().cpu().numpy())
                sync_o_history.append(
                    attn_state['o']['decay_alpha'].mean(dim=1).detach().cpu().numpy())

        if track:
            return predictions, certainties, attn_history, token_history, sync_o_history
        return predictions, certainties


def get_loss(predictions, certainties, targets, use_most_certain=True):
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
