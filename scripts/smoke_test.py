"""Smoke test: build the model, run a forward pass + backward on fake data.

No JetClass ROOT files needed. Run:
    python particle_ctm/scripts/smoke_test.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from particle_ctm.models.particle_ctm import ParticleCTM, get_loss, calculate_accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, P = 4, 17, 32
    x_feat = torch.randn(B, C, P, device=device)
    # plausible 4-vectors: keep energy > |p| so log(E - pz) is finite
    px = torch.randn(B, 1, P, device=device)
    py = torch.randn(B, 1, P, device=device)
    pz = torch.randn(B, 1, P, device=device)
    e = torch.sqrt(px ** 2 + py ** 2 + pz ** 2) + 1.0
    x_vec = torch.cat([px, py, pz, e], dim=1)
    mask = torch.ones(B, 1, P, device=device)
    mask[:, :, P // 2:] = 0  # half are padding
    y = torch.randint(0, 10, (B,), device=device)

    model = ParticleCTM(
        input_dim=C, num_classes=10,
        embed_dims=(64, 64),
        pair_embed_dims=(32, 32),
        num_heads=4,
        iterations=3,
        memory_length=4,
        d_model_qkv=32,
        d_model_o=32,
        n_synch_qkv=8,
        n_synch_o=8,
        trim=False,
    ).to(device)
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')

    preds, certs = model(x_feat, v=x_vec, mask=mask)
    loss, where = get_loss(preds, certs, y)
    acc = calculate_accuracy(preds, y, where)
    print(f'forward OK. preds {tuple(preds.shape)} certs {tuple(certs.shape)} '
          f'loss {loss.item():.4f} acc {acc:.3f}')
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f'backward OK. {has_grad} tensors have grads.')


if __name__ == '__main__':
    main()
