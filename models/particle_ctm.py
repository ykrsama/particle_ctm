"""ParticleCTM: single global CTM neuron pool, every module is a readout.

One `GlobalCTMPool` holds a (B, N_global) latent state and produces one sync
vector per outer tick. Seven readout Linears (embed / pair / Q / K / V / O /
head) all consume the SAME sync_{t-1} in parallel within each tick - none of
them have their own trace, NLM, or sync. At the end of the tick the mean-
pooled attention output drives a single pre-activation into the global pool
to produce sync_t for the next tick.

Per outer tick t (sync_{t-1} carried over from tick t-1, sync_0 from
`GlobalCTMPool.initial_sync`):
    1. tokens = embed_readout([x, sync_{t-1}])                   # (B, P, embed_dim)
    2. bias   = pair_readout([pair_geom_feats, sync_{t-1}])      # (B, H, P, P)
    3. Q/K/V  = q/k/v_readout([tokens, sync_{t-1}])              # (B, P, embed_dim)
    4. attn   = scaled_dot_product(Q, K, V) + bias               # standard attn
    5. o      = o_readout([attn, sync_{t-1}])                    # (B, P, embed_dim)
    6. logits = head_readout(sync_{t-1})                         # (B, num_classes)
    7. pool_in = masked_mean(o, particles); sync_t = global_pool.step(pool_in)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm.global_pool import GlobalCTMPool
from .part_layers import SequenceTrimmer, pairwise_lv_fts


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
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 embed_dim=128,
                 num_heads=8,
                 iterations=8,
                 # Global pool config
                 n_global=128,
                 n_synch_global=16,
                 memory_length=16,
                 dropout=0.0,
                 trim=True):
        super().__init__()
        if pair_extra_dim != 0:
            raise NotImplementedError(
                "pair_extra_dim != 0 not supported in the global-pool architecture")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pair_input_dim = pair_input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.iterations = iterations
        self.dropout_p = dropout

        self.trimmer = SequenceTrimmer(enabled=trim)
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.pair_bn = nn.BatchNorm1d(pair_input_dim) if pair_input_dim > 0 else None

        self.global_pool = GlobalCTMPool(
            d_pool_in=embed_dim,
            n_neurons=n_global,
            memory_length=memory_length,
            n_synch=n_synch_global,
            dropout=dropout,
        )
        ssize = self.global_pool.sync_size

        # All readouts: take their specific data input concatenated with the
        # shared sync_{t-1} vector and project to the target dimension.
        self.embed_readout = nn.Linear(input_dim + ssize, embed_dim)
        self.pair_readout = nn.Linear(pair_input_dim + ssize, num_heads) \
            if pair_input_dim > 0 else None
        self.q_readout = nn.Linear(embed_dim + ssize, embed_dim)
        self.k_readout = nn.Linear(embed_dim + ssize, embed_dim)
        self.v_readout = nn.Linear(embed_dim + ssize, embed_dim)
        self.o_readout = nn.Linear(embed_dim + ssize, embed_dim)
        self.head_readout = nn.Linear(ssize, num_classes)

        self.norm = nn.LayerNorm(embed_dim)

    def _compute_pair_geom(self, v, P):
        """Returns (B, num_pairs, pair_input_dim) pair features in row-major
        tril-with-diagonal order, plus the (i, j) indices used for scatter."""
        i, j = torch.tril_indices(P, P, offset=0, device=v.device)
        v_exp = v.unsqueeze(-1).expand(-1, -1, -1, P)
        vi = v_exp[:, :, i, j]
        vj = v_exp[:, :, j, i]
        feats = pairwise_lv_fts(vi, vj, num_outputs=self.pair_input_dim)
        return feats, i, j

    def _scatter_pair_bias(self, bias_flat, i, j, B, P):
        """bias_flat: (B, num_pairs, num_heads). Returns (B, num_heads, P, P)."""
        bias = bias_flat.transpose(1, 2).contiguous()  # (B, num_heads, num_pairs)
        y = bias.new_zeros(B, self.num_heads, P, P)
        y[:, :, i, j] = bias
        y[:, :, j, i] = bias
        return y

    def _attention(self, Q, K, V, bias, kp_mask, need_weights):
        """Standard scaled-dot-product attention. Q/K/V: (B, P, embed_dim)."""
        B, P, _ = Q.shape

        def split_heads(t):
            return t.reshape(B, P, self.num_heads, self.head_dim).transpose(1, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale  # (B, H, P, P)
        if bias is not None:
            logits = logits + bias
        if kp_mask is not None:
            kp_add = torch.zeros(B, 1, 1, P, device=Q.device, dtype=Q.dtype)
            kp_add = kp_add.masked_fill(kp_mask.view(B, 1, 1, P), float('-inf'))
            logits = logits + kp_add
        weights = F.softmax(logits, dim=-1)
        dropped = F.dropout(weights, p=self.dropout_p, training=self.training)
        out = torch.matmul(dropped, Vh)                          # (B, H, P, head_dim)
        out = out.transpose(1, 2).reshape(B, P, self.embed_dim)  # (B, P, embed_dim)
        return out, (weights if need_weights else None)

    def forward(self, x, v=None, mask=None, track=False):
        """x: (B, C, P); v: (B, 4, P); mask: (B, 1, P)."""
        x, v, mask, _ = self.trimmer(x, v, mask, None)
        B, _, P = x.shape
        device = x.device

        x_bn = self.input_bn(x).transpose(1, 2).contiguous()  # (B, P, C)
        particle_pad = ~mask.squeeze(1).bool()                 # (B, P) True at pad
        keep = (~particle_pad).to(x.dtype).unsqueeze(-1)       # (B, P, 1)
        keep_sum = keep.sum(dim=1).clamp(min=1.0)              # (B, 1)

        if self.pair_readout is not None and v is not None:
            with torch.no_grad():
                pair_geom, i_idx, j_idx = self._compute_pair_geom(v, P)
            pair_geom = self.pair_bn(pair_geom).transpose(1, 2).contiguous()
        else:
            pair_geom = i_idx = j_idx = None

        T = self.iterations
        predictions = torch.empty(B, self.num_classes, T, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, T, device=device, dtype=x.dtype)

        attn_history = [] if track else None
        token_history = [] if track else None
        sync_history = [] if track else None

        sync = self.global_pool.initial_sync(B, device, x.dtype)  # sync_0 (B, ssize)
        pool_state = None
        ssize = sync.size(-1)

        for t in range(T):
            sync_p = sync.unsqueeze(1).expand(B, P, ssize)        # (B, P, ssize)

            tokens = self.embed_readout(torch.cat([x_bn, sync_p], dim=-1))  # (B, P, embed_dim)
            tokens = tokens * keep

            if pair_geom is not None:
                sync_pp = sync.unsqueeze(1).expand(B, pair_geom.size(1), ssize)
                bias_flat = self.pair_readout(torch.cat([pair_geom, sync_pp], dim=-1))
                bias = self._scatter_pair_bias(bias_flat, i_idx, j_idx, B, P)
            else:
                bias = None

            tok_sync = torch.cat([tokens, sync_p], dim=-1)
            Q = self.q_readout(tok_sync)
            K = self.k_readout(tok_sync)
            V = self.v_readout(tok_sync)

            attn_out, attn_w = self._attention(Q, K, V, bias, particle_pad, track)
            o = self.o_readout(torch.cat([attn_out, sync_p], dim=-1)) * keep

            logits = self.head_readout(sync)                      # uses sync_{t-1}

            # Drive the global pool with the masked-mean of o → sync_t.
            pool_in = (o * keep).sum(dim=1) / keep_sum            # (B, embed_dim)
            pool_in = self.norm(pool_in)
            sync, pool_state = self.global_pool.step(pool_in, pool_state)

            ne = compute_normalized_entropy(logits)
            certainty = torch.stack((ne, 1 - ne), dim=-1)
            predictions[..., t] = logits
            certainties[..., t] = certainty

            if track:
                attn_history.append(attn_w)
                token_history.append(o.detach().cpu().numpy())
                sync_history.append(sync.detach().cpu().numpy())

        if track:
            return predictions, certainties, attn_history, token_history, sync_history
        return predictions, certainties


def summarize_parameters(model):
    """Per-top-level-child parameter counts plus model total."""
    def _fmt(n):
        return f'{n:>13,d}'

    children = list(model.named_children())
    name_w = max([len(n) for n, _ in children] + [5])

    lines = ['Parameter breakdown:']
    total = 0
    total_train = 0
    for name, child in children:
        n = sum(p.numel() for p in child.parameters())
        nt = sum(p.numel() for p in child.parameters() if p.requires_grad)
        total += n
        total_train += nt
        lines.append(f'  {name:<{name_w}}  total={_fmt(n)}  trainable={_fmt(nt)}')
    direct = sum(p.numel() for p in model.parameters(recurse=False))
    if direct:
        lines.append(f'  {"(direct)":<{name_w}}  total={_fmt(direct)}  trainable={_fmt(direct)}')
        total += direct
        total_train += direct
    lines.append('  ' + '-' * (name_w + 40))
    lines.append(f'  {"TOTAL":<{name_w}}  total={_fmt(total)}  trainable={_fmt(total_train)}')
    return '\n'.join(lines)


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
