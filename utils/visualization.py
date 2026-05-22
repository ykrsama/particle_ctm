"""Saliency, certainty and neural-dynamics visualisations for ParticleCTM.

Adapted from `continuous-thought-machines/examples/07_imagenette_ctmattention.ipynb`,
but for the 1D particle sequence (P particles + 1 cls token) rather than a
14×14 patch grid. No spatial overlay; instead we plot per-particle saliency
bars and the multi-head attention paid by the cls token at each tick.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from scipy.special import softmax
from tqdm import tqdm

from particle_ctm.data.jetclass import LABELS


def compute_cls_saliency(model, x_feat, x_vec, mask):
    """Run a tracked forward pass and compute per-particle Grad-CAM-style
    saliency from the cls-token attention row at every tick.

    Returns:
        predictions  (B, C, T)
        certainties  (B, 2, T)
        attn_grid    np.ndarray (T, B, num_heads, P)   # cls row, particle keys
        token_acts   np.ndarray (T, B, 1+P, embed_dim)
        saliency     np.ndarray (T, B, P)              # per-particle, per-tick
    """
    model.eval()
    preds, certs, attn_tensors, tok_acts, _ = model(x_feat, v=x_vec, mask=mask, track=True)
    B = preds.size(0)
    T = preds.size(-1)
    pred_class = preds.argmax(dim=1).detach()

    saliencies = []
    for t in range(T):
        tgt = preds[torch.arange(B, device=preds.device), pred_class[:, t], t].sum()
        grad_t = torch.autograd.grad(tgt, attn_tensors[t], retain_graph=(t < T - 1))[0]
        # attn_tensors[t]: (B, num_heads, 1+P, 1+P) — take cls row, drop cls col.
        sal = attn_tensors[t][:, :, 0, 1:].clamp(min=0) * grad_t[:, :, 0, 1:].clamp(min=0)
        sal = sal.mean(dim=1)  # (B, P)
        saliencies.append(sal.detach().cpu())

    saliency = torch.stack(saliencies, dim=0).numpy()  # (T, B, P)
    attn_stack = torch.stack([a[:, :, 0, 1:].detach().cpu() for a in attn_tensors], dim=0).numpy()
    return preds.detach(), certs.detach(), attn_stack, np.array(tok_acts), saliency


def make_saliency_gif(predictions, certainties, targets,
                     attention_per_tick, saliency, masks,
                     class_names, out_path, batch_index=0,
                     top_k_particles=20):
    """Animate predictions + per-tick certainty + per-particle saliency bars.

    attention_per_tick: (T, num_heads, P) for the chosen sample (already squeezed)
    saliency:           (T, P) for the chosen sample
    masks:              (P,) 0/1
    """
    T, num_heads, P = attention_per_tick.shape
    print(f'[plot] saliency_gif: T={T}, heads={num_heads}, P={P}, '
          f'batch_index={batch_index}, top_k={top_k_particles}, out={out_path}')
    these_predictions = predictions[batch_index].detach().cpu().numpy()  # (C, T)
    these_certainties = certainties[batch_index].detach().cpu().numpy()  # (2, T)
    this_target = int(targets[batch_index])
    num_classes = these_predictions.shape[0]
    short_labels = [name.replace('label_', '')[:10] for name in class_names]
    mask_np = masks.detach().cpu().numpy() if hasattr(masks, 'detach') else np.asarray(masks)
    real_idx = np.where(mask_np > 0)[0]

    # Pick the top-K particles by mean saliency across ticks for a clean stacked plot.
    mean_sal = saliency.mean(axis=0)
    mean_sal_real = mean_sal.copy()
    mean_sal_real[mask_np == 0] = -np.inf
    top_idx = np.argsort(-mean_sal_real)[:min(top_k_particles, len(real_idx))]

    cmap_attn = sns.color_palette('viridis', as_cmap=True)
    frames = []

    for t in tqdm(range(T), desc='Saliency frames'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                                 gridspec_kw={'height_ratios': [1.2, 1]})

        # Class probabilities
        probs = softmax(these_predictions[:, t])
        colors = ['g' if i == this_target else 'b' for i in range(num_classes)]
        axes[0, 0].bar(np.arange(num_classes), probs, color=colors, alpha=0.6)
        axes[0, 0].set_xticks(np.arange(num_classes))
        axes[0, 0].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_title(f'Class probs (tick {t}/{T-1})')

        # Certainty curve
        axes[0, 1].plot(np.arange(T), these_certainties[1], 'k-', lw=2)
        axes[0, 1].axvline(t, color='red', alpha=0.5)
        axes[0, 1].set_title('Certainty (1 - normalised entropy)')
        axes[0, 1].set_xlim([0, T - 1])
        axes[0, 1].set_ylim([0, 1])

        # Per-particle saliency bars (top-K)
        ax = axes[1, 0]
        sal_t = saliency[t][top_idx]
        ax.bar(np.arange(len(top_idx)), sal_t, color='crimson', alpha=0.7)
        ax.set_xticks(np.arange(len(top_idx)))
        ax.set_xticklabels([str(int(i)) for i in top_idx], rotation=45, fontsize=7)
        ax.set_xlabel('particle index')
        ax.set_title(f'Top-{len(top_idx)} particle saliency')

        # Multi-head attention from cls to all real particles
        ax = axes[1, 1]
        attn_t = attention_per_tick[t][:, real_idx]  # (num_heads, |real|)
        # Per-head normalise so heatmap is visible.
        a_min = attn_t.min(axis=1, keepdims=True)
        a_max = attn_t.max(axis=1, keepdims=True)
        attn_norm = (attn_t - a_min) / (a_max - a_min + 1e-8)
        ax.imshow(attn_norm, aspect='auto', cmap=cmap_attn)
        ax.set_ylabel('head')
        ax.set_xlabel('real particle')
        ax.set_title('cls attention by head')

        fig.suptitle(f'ground truth: {class_names[this_target]}', fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(*reversed(fig.canvas.get_width_height()), 4)[:, :, :3]
        frames.append(img)
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    imageio.mimsave(out_path, frames, fps=4, loop=0)
    print(f'[plot] saliency_gif: saved {out_path} ({len(frames)} frames)')
    return out_path


def plot_neural_dynamics_simple(token_activations, out_path, n_to_plot=80,
                                n_per_row=10, title=None):
    """Per-neuron trace grid: overlay all token traces (sample 0) with one
    randomly-highlighted solid curve on top — mirrors the overlay idiom in
    continuous-thought-machines/tasks/image_classification/plotting.py."""
    th = np.asarray(token_activations)  # (T, B, 1+P, embed_dim)
    print(f'[plot] neural_dynamics: input shape={th.shape}, n_to_plot={n_to_plot}, '
          f'out={out_path}')
    if th.ndim == 3:
        th = th[:, :, None, :]  # treat as L=1
    T, B, L, D = th.shape
    n_to_plot = min(n_to_plot, D)
    n_to_plot = (n_to_plot // n_per_row) * n_per_row
    n_rows = n_to_plot // n_per_row

    palette = sns.color_palette('husl', 8)
    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(n_per_row * 1.4, n_rows * 0.8),
                             sharex=True)
    xs = np.arange(T)
    for i in range(n_to_plot):
        r, c = i // n_per_row, i % n_per_row
        ax = axes[r, c] if n_rows > 1 else axes[c]
        traces = th[:, 0, :, i].T  # (L, T)
        color = palette[np.random.randint(0, 8)]
        for tr in traces:
            ax.plot(xs, tr, lw=0.6, alpha=0.15, color=color)
        solid = traces[np.random.randint(0, L)]
        ax.plot(xs, solid, color='white', lw=2.5, alpha=1)
        ax.plot(xs, solid, color=color, lw=1.3, alpha=1)
        ax.plot(xs, solid, color='black', lw=0.3, alpha=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title or 'Neural dynamics (per-token overlay, sample 0)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] neural_dynamics: saved {out_path}')
    return out_path
