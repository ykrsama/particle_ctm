"""GlobalCTMPool: a single per-sample neuron pool shared across the entire
ParticleCTM. Holds one (B, N) latent state per sample; updated once per
outer tick by a single pre-activation input (B, d_pool_in). Produces one
sync vector (B, sync_size) per tick that every downstream readout linear
(embed / pair / Q / K / V / O / head) consumes in parallel.

Unlike `CTMPool` (per-position pool), this module:
  * has no L (per-position) dimension - one neuron bank per sample;
  * has no built-in out_proj - each readout supplies its own linear;
  * exposes `initial_sync` to compute a non-trivial sync_0 from the learnable
    start_trace alone, so the first tick's readouts see meaningful sync.
"""

import math

import torch
import torch.nn as nn

from .ctm_pool import _build_nlm, compute_sync_first_last


class GlobalCTMPool(nn.Module):

    def __init__(self,
                 d_pool_in,
                 n_neurons,
                 memory_length,
                 n_synch,
                 side='first',
                 dropout=0.0):
        super().__init__()
        if n_synch > n_neurons:
            raise ValueError(
                f"n_synch ({n_synch}) cannot exceed n_neurons ({n_neurons})")
        if side not in ('first', 'last'):
            raise ValueError(f"side must be 'first' or 'last', got {side!r}")

        self.d_pool_in = d_pool_in
        self.n_neurons = n_neurons
        self.memory_length = memory_length
        self.n_synch = n_synch
        self.side = side
        self.sync_size = (n_synch * (n_synch + 1)) // 2

        self.pre_proj = nn.Linear(d_pool_in, n_neurons)
        self.nlm = _build_nlm(memory_length, n_neurons, dropout)

        bound = math.sqrt(1.0 / (n_neurons + memory_length))
        self.start_trace = nn.Parameter(
            torch.empty(n_neurons, memory_length).uniform_(-bound, bound))
        self.decay_params = nn.Parameter(torch.zeros(self.sync_size))

    def init_state(self, B, device, dtype):
        trace = self.start_trace.to(device=device, dtype=dtype) \
            .unsqueeze(0).expand(B, -1, -1).contiguous()
        return {'trace': trace, 'decay_alpha': None, 'decay_beta': None}

    def _decay_rate(self, B):
        return torch.exp(-self.decay_params.clamp(0.0, 15.0)) \
            .unsqueeze(0).expand(B, self.sync_size)

    def initial_sync(self, B, device, dtype):
        """sync_0 used by the very first tick's readouts. Run one NLM pass on
        the learnable start_trace alone (no input has been seen yet), then
        seed the sync recurrence as if this were the first observation.
        Returns (B, sync_size)."""
        state = self.init_state(B, device, dtype)
        activated = self.nlm(state['trace'])  # (B, n_neurons)
        r = self._decay_rate(B)
        sync, _, _ = compute_sync_first_last(
            activated, self.n_synch, self.side, None, None, r)
        return sync

    def step(self, pre_act, state=None):
        """Drive the global pool with one (B, d_pool_in) pre-activation input.

        Returns:
            sync:      (B, sync_size) - the new sync_t after this update.
            new_state: dict carrying trace + sync recurrence forward.
        """
        if pre_act.dim() != 2:
            raise ValueError(
                f"GlobalCTMPool.step expects (B, d_pool_in), got {pre_act.shape}")
        B = pre_act.size(0)
        if state is None:
            state = self.init_state(B, pre_act.device, pre_act.dtype)
        trace = state['trace']
        decay_alpha = state['decay_alpha']
        decay_beta = state['decay_beta']

        pre = self.pre_proj(pre_act)                          # (B, n_neurons)
        new_trace = torch.cat((trace[..., 1:], pre.unsqueeze(-1)), dim=-1)
        activated = self.nlm(new_trace)                       # (B, n_neurons)

        r = self._decay_rate(B)
        sync, new_alpha, new_beta = compute_sync_first_last(
            activated, self.n_synch, self.side,
            decay_alpha, decay_beta, r)

        new_state = {
            'trace': new_trace,
            'decay_alpha': new_alpha,
            'decay_beta': new_beta,
        }
        return sync, new_state
