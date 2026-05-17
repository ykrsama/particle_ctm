"""ParticleCTM: one global CTM pool, every projection is an NLMReadout.

High-level pipeline per outer tick (with `sync_{t-1}` carried over from the
previous tick, `sync_0` from `GlobalCTMPool.initial_sync`):

    1. particle_embed([x_norm,    sync_{t-1}]) -> (B, P, particle_embed_dim)
    2. pair_embed   ([pair_norm,  sync_{t-1}]) -> (B, num_pairs, pair_embed_dim)
    3. pair_bias    ([pair_emb,   sync_{t-1}]) -> (B, num_pairs, num_heads) -> scatter to (B, H, P, P)
    4. Q, K, V from particle_embed + sync_{t-1}
    5. attn = scaled_dot_product(Q, K, V, bias=pair_bias, kp_mask=particle_pad)
    6. o    = o_readout([attn, sync_{t-1}])
    7. logits = head(sync_{t-1})                      -- emits the tick's prediction
    8. pool_in = mean-pool(o, particles)
       sync_t, pool_state = global_pool.step(pool_in, pool_state)

Every parameterised projection (particle_embed, pair_embed, pair_bias, Q, K,
V, O, head, and the global pool's pre_proj) is an `NLMReadout` -- per-output-
neuron private weights with GLU gating. The only trace + NLM + sync
recurrence lives inside the single `global_pool`; everyone else is a
sync-readout that reads `sync_{t-1}` and produces its output.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctm.global_pool import GlobalCTMPool
from .ctm.nlm_readout import NLMReadout
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
                 particle_embed_dim=128,
                 pair_embed_dim=64,
                 num_heads=8,
                 iterations=8,
                 n_global=128,
                 n_synch_global=16,
                 memory_length=16,
                 dropout=0.0,
                 trim=True):
        super().__init__()
        if particle_embed_dim % num_heads != 0:
            raise ValueError(
                f"particle_embed_dim ({particle_embed_dim}) must be divisible "
                f"by num_heads ({num_heads})")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pair_input_dim = pair_input_dim
        self.particle_embed_dim = particle_embed_dim
        self.pair_embed_dim = pair_embed_dim
        self.num_heads = num_heads
        self.head_dim = particle_embed_dim // num_heads
        self.iterations = iterations
        self.dropout_p = dropout

        self.trimmer = SequenceTrimmer(enabled=trim)
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.pair_bn = nn.BatchNorm1d(pair_input_dim) if pair_input_dim > 0 else None

        # The single source of trace + NLM + sync recurrence in the model.
        self.global_pool = GlobalCTMPool(
            d_pool_in=particle_embed_dim,
            n_neurons=n_global,
            memory_length=memory_length,
            n_synch=n_synch_global,
            dropout=dropout,
        )
        ssize = self.global_pool.sync_size

        # All other projections are NLM-style sync-readouts. Each takes its
        # specific data input concatenated with the shared sync_{t-1}.
        self.particle_embed = NLMReadout(input_dim + ssize, particle_embed_dim, dropout=dropout)
        self.pair_embed = NLMReadout(pair_input_dim + ssize, pair_embed_dim, dropout=dropout) \
            if pair_input_dim > 0 else None
        self.pair_bias = NLMReadout(pair_embed_dim + ssize, num_heads, dropout=dropout) \
            if pair_input_dim > 0 else None
        self.q = NLMReadout(particle_embed_dim + ssize, particle_embed_dim, dropout=dropout)
        self.k = NLMReadout(particle_embed_dim + ssize, particle_embed_dim, dropout=dropout)
        self.v = NLMReadout(particle_embed_dim + ssize, particle_embed_dim, dropout=dropout)
        self.o = NLMReadout(particle_embed_dim + ssize, particle_embed_dim, dropout=dropout)
        self.head = NLMReadout(ssize, num_classes, dropout=dropout)

    def _compute_pair_geom(self, v, P):
        i, j = torch.tril_indices(P, P, offset=0, device=v.device)
        v_exp = v.unsqueeze(-1).expand(-1, -1, -1, P)
        vi = v_exp[:, :, i, j]
        vj = v_exp[:, :, j, i]
        feats = pairwise_lv_fts(vi, vj, num_outputs=self.pair_input_dim)
        return feats, i, j

    def _scatter_pair_bias(self, bias_flat, i, j, B, P):
        bias = bias_flat.transpose(1, 2).contiguous()  # (B, num_heads, num_pairs)
        y = bias.new_zeros(B, self.num_heads, P, P)
        y[:, :, i, j] = bias
        y[:, :, j, i] = bias
        return y

    def _attention(self, Q, K, V, bias, kp_mask, need_weights):
        B, P, _ = Q.shape

        def split_heads(t):
            return t.reshape(B, P, self.num_heads, self.head_dim).transpose(1, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale
        if bias is not None:
            logits = logits + bias
        if kp_mask is not None:
            kp_add = torch.zeros(B, 1, 1, P, device=Q.device, dtype=Q.dtype)
            kp_add = kp_add.masked_fill(kp_mask.view(B, 1, 1, P), float('-inf'))
            logits = logits + kp_add
        weights = F.softmax(logits, dim=-1)
        dropped = F.dropout(weights, p=self.dropout_p, training=self.training)
        out = torch.matmul(dropped, Vh)
        out = out.transpose(1, 2).reshape(B, P, self.particle_embed_dim)
        return out, (weights if need_weights else None)

    def forward(self, x, v=None, mask=None, track=False):
        x, v, mask, _ = self.trimmer(x, v, mask, None)
        B, _, P = x.shape
        device = x.device

        x_norm = self.input_bn(x).transpose(1, 2).contiguous()  # (B, P, input_dim)
        particle_pad = ~mask.squeeze(1).bool()                   # (B, P) True at pad
        keep = (~particle_pad).to(x.dtype).unsqueeze(-1)         # (B, P, 1)
        keep_sum = keep.sum(dim=1).clamp(min=1.0)                # (B, 1)

        if self.pair_embed is not None and v is not None:
            with torch.no_grad():
                pair_geom, i_idx, j_idx = self._compute_pair_geom(v, P)
            pair_norm = self.pair_bn(pair_geom).transpose(1, 2).contiguous()
            num_pairs = pair_norm.size(1)
        else:
            pair_norm = i_idx = j_idx = None
            num_pairs = 0

        T = self.iterations
        predictions = torch.empty(B, self.num_classes, T, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, T, device=device, dtype=x.dtype)

        attn_history = [] if track else None
        token_history = [] if track else None
        sync_history = [] if track else None

        sync = self.global_pool.initial_sync(B, device, x.dtype)  # sync_0
        pool_state = None
        ssize = sync.size(-1)

        for t in range(T):
            sync_p = sync.unsqueeze(1).expand(B, P, ssize)        # (B, P, ssize)

            p_emb = self.particle_embed(torch.cat([x_norm, sync_p], dim=-1)) * keep

            if pair_norm is not None:
                sync_pp = sync.unsqueeze(1).expand(B, num_pairs, ssize)
                pe_emb = self.pair_embed(torch.cat([pair_norm, sync_pp], dim=-1))
                bias_flat = self.pair_bias(torch.cat([pe_emb, sync_pp], dim=-1))
                bias = self._scatter_pair_bias(bias_flat, i_idx, j_idx, B, P)
            else:
                bias = None

            pq = torch.cat([p_emb, sync_p], dim=-1)
            Q = self.q(pq)
            K = self.k(pq)
            V = self.v(pq)

            attn_out, attn_w = self._attention(Q, K, V, bias, particle_pad, track)
            o_t = self.o(torch.cat([attn_out, sync_p], dim=-1)) * keep

            logits = self.head(sync)                              # reads sync_{t-1}

            pool_in = (o_t * keep).sum(dim=1) / keep_sum          # (B, particle_embed_dim)
            sync, pool_state = self.global_pool.step(pool_in, pool_state)

            ne = compute_normalized_entropy(logits)
            certainty = torch.stack((ne, 1 - ne), dim=-1)
            predictions[..., t] = logits
            certainties[..., t] = certainty

            if track:
                attn_history.append(attn_w)
                token_history.append(o_t.detach().cpu().numpy())
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
