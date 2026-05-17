"""Shared helpers for the global CTM pool.

In earlier revisions of this project every projection (Q/K/V/O, embed, pair,
head) had its own per-position `CTMPool`. The current architecture replaces
all of those with a single `GlobalCTMPool` and turns every projection into a
plain readout `Linear` of the shared sync vector. The two utilities below are
kept because the global pool still relies on them.
"""

import torch
import torch.nn as nn

from .modules import SuperLinear, Squeeze


def _build_nlm(memory_length, n_neurons, dropout=0.0):
    """Per-neuron MLP that maps a length-`memory_length` trace to a scalar
    post-activation. SuperLinear holds N independent weight matrices."""
    return nn.Sequential(
        SuperLinear(in_dims=memory_length, out_dims=2,
                    N=n_neurons, dropout=dropout),
        nn.GLU(),
        Squeeze(-1),
    )


def compute_sync_first_last(activated, n_synch, side, decay_alpha, decay_beta, r):
    """Pairwise-product synchronization recurrence over the first or last
    `n_synch` neurons of `activated`. Returns (sync, decay_alpha, decay_beta)."""
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
