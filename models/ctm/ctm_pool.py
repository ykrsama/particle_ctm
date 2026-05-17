"""CTMPool: one neuron pool's full NLM+Synchronization single-tick computation.

Encapsulates the "Linear pre-projection -> sliding trace -> SuperLinear NLM ->
pairwise-product Synchronization -> sync->out_dim Linear" pipeline shared by
the Q/K/V/O pools inside CTMAttention and by CTMEmbed / CTMPairEmbed / CTMHead.

State for one tick is a dict with keys `trace`, `decay_alpha`, `decay_beta`.
`trace` is always present (built from `start_trace` on the first tick), the
two decay tensors are None until the recurrence has stepped once.
"""

import math

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
    """Pairwise synchronization recurrence over the first or last `n_synch`
    neurons of `activated`. Identical logic to the original implementation in
    `CTMAttention`, but lifted out so it can be reused by every pool."""
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


class CTMPool(nn.Module):
    """One CTM neuron pool: Linear synapse -> trace -> NLM -> Sync -> Linear.

    Per-position neurons: each (B, L) slot carries its own private set of N
    neurons with its own trace; NLM weights are shared across positions.

    Inputs:
        x:     (B, L, d_in)
        state: optional dict with keys
               - 'trace':       (B, L, n_neurons, memory_length)
               - 'decay_alpha': (B, L, sync_size) or None on the first tick
               - 'decay_beta':  (B, L, sync_size) or None on the first tick

    Returns:
        out:       (B, L, d_out)
        new_state: dict with the same keys as above
    """

    def __init__(self,
                 d_in,
                 d_out,
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

        self.d_in = d_in
        self.d_out = d_out
        self.n_neurons = n_neurons
        self.memory_length = memory_length
        self.n_synch = n_synch
        self.side = side
        self.sync_size = (n_synch * (n_synch + 1)) // 2

        self.pre_proj = nn.Linear(d_in, n_neurons)
        self.nlm = _build_nlm(memory_length, n_neurons, dropout)
        self.out_proj = nn.Linear(self.sync_size, d_out)

        bound = math.sqrt(1.0 / (n_neurons + memory_length))
        self.start_trace = nn.Parameter(
            torch.empty(n_neurons, memory_length).uniform_(-bound, bound))
        self.decay_params = nn.Parameter(torch.zeros(self.sync_size))

    def init_state(self, B, L, device, dtype):
        trace = self.start_trace.to(device=device, dtype=dtype) \
            .unsqueeze(0).unsqueeze(0) \
            .expand(B, L, -1, -1).contiguous()
        return {'trace': trace, 'decay_alpha': None, 'decay_beta': None}

    def forward(self, x, state=None):
        if x.dim() != 3:
            raise ValueError(f"CTMPool expects 3D input (B, L, d_in), got {x.shape}")
        B, L, d_in = x.shape
        if d_in != self.d_in:
            raise ValueError(
                f"input last dim {d_in} != d_in {self.d_in}")

        if state is None:
            state = self.init_state(B, L, x.device, x.dtype)
        trace = state['trace']
        decay_alpha = state['decay_alpha']
        decay_beta = state['decay_beta']

        # 1. Pre-activation per position (B, L, n_neurons)
        pre = self.pre_proj(x)

        # 2. Slide the trace window: drop oldest column, append the new pre-act
        new_trace = torch.cat((trace[..., 1:], pre.unsqueeze(-1)), dim=-1)

        # 3. NLM: flatten (B, L) -> (B*L) for SuperLinear's neuron-batched einsum
        nlm_in = new_trace.reshape(B * L, self.n_neurons, self.memory_length)
        activated = self.nlm(nlm_in)  # (B*L, n_neurons)

        # 4. Sync recurrence (per (B, L) row of activated)
        r = torch.exp(-self.decay_params.clamp(0.0, 15.0)) \
            .unsqueeze(0).expand(B * L, self.sync_size)

        if decay_alpha is not None:
            decay_alpha_flat = decay_alpha.reshape(B * L, self.sync_size)
            decay_beta_flat = decay_beta.reshape(B * L, self.sync_size)
        else:
            decay_alpha_flat = None
            decay_beta_flat = None

        sync_flat, new_alpha_flat, new_beta_flat = compute_sync_first_last(
            activated, self.n_synch, self.side,
            decay_alpha_flat, decay_beta_flat, r)

        # 5. Sync -> d_out and reshape back to (B, L, d_out)
        out = self.out_proj(sync_flat).reshape(B, L, self.d_out)
        new_state = {
            'trace': new_trace,
            'decay_alpha': new_alpha_flat.reshape(B, L, self.sync_size),
            'decay_beta': new_beta_flat.reshape(B, L, self.sync_size),
        }
        return out, new_state
