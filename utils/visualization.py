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
from matplotlib.collections import LineCollection
import seaborn as sns
import imageio
from scipy.special import softmax
from tqdm import tqdm

from particle_ctm.data.jetclass import LABELS


# Channel indices into x_feat (the 17-dim standardised PF features built by
# particle_ctm.data.jetclass._derive_features). deta/dphi are pass-through
# (sub=0, mul=1), so the values are usable as plotting coordinates directly.
# PID flags and charge also have sub=0, mul=1 so round-trip cleanly.
_FEAT_PT_LOG_IDX = 0
_FEAT_CHARGE_IDX = 5
_FEAT_ISCH_IDX = 6   # isChargedHadron
_FEAT_ISNH_IDX = 7   # isNeutralHadron
_FEAT_ISPH_IDX = 8   # isPhoton
_FEAT_ISE_IDX = 9    # isElectron
_FEAT_ISMU_IDX = 10  # isMuon
_FEAT_DETA_IDX = 15
_FEAT_DPHI_IDX = 16


def _draw_particle_cloud(ax, deta, dphi, pid_flags, *, sizes, colors=None,
                         alpha=None, edge_linewidth=0.6):
    """Plot particles in the `plot_particle_clouds` style on `ax`.

    Args:
        ax:        matplotlib axis to draw on.
        deta, dphi: (N,) float arrays of jet-relative coordinates.
        pid_flags: dict with keys 'ch', 'nh', 'ph', 'e', 'mu' — each a boolean
                   (N,) array. Anything not covered ends up in the 'other'
                   group with marker 'x'.
        sizes:     (N,) marker areas.
        colors:    (N, 4) RGBA array, or None for neutral dimgray.
        alpha:     scalar or (N,) array; if (N,) array, baked into the RGBA.
        edge_linewidth: outline thickness.

    Returns:
        dict[str, dict] mapping PID group key to {
            'collection': PathCollection returned by ax.scatter,
            'mask': bool (N,) selection mask for this group,
            'fill': 'full' or 'none' (does the group respect facecolor?),
        }. Keys present only for groups that contained at least one particle.
    """
    N = len(deta)
    if colors is None:
        rgba = np.tile(np.array([0.4, 0.4, 0.4, 1.0]), (N, 1))
    else:
        rgba = np.asarray(colors, dtype=float).copy()
        if rgba.shape[-1] == 3:
            rgba = np.concatenate([rgba, np.ones((N, 1))], axis=-1)
    if alpha is not None:
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim == 0:
            rgba[:, 3] = float(alpha_arr)
        else:
            rgba[:, 3] = alpha_arr

    ch = pid_flags['ch']
    nh = pid_flags['nh']
    ee = pid_flags['e']
    mu = pid_flags['mu']
    ph = pid_flags['ph']
    other = ~(ch | nh | ee | mu | ph)

    groups = [
        ('ch',    ch,    'o', 'full'),
        ('nh',    nh,    'o', 'none'),
        ('e',     ee,    'v', 'full'),
        ('mu',    mu,    '^', 'full'),
        ('ph',    ph,    'p', 'none'),
        ('other', other, 'x', 'full'),
    ]
    handles = {}
    for key, sel, marker, fill in groups:
        if not sel.any():
            continue
        sel_colors = rgba[sel]
        if fill == 'none':
            sc = ax.scatter(deta[sel], dphi[sel], s=sizes[sel],
                            marker=marker, facecolors='none',
                            edgecolors=sel_colors, linewidths=edge_linewidth)
        else:
            sc = ax.scatter(deta[sel], dphi[sel], s=sizes[sel],
                            marker=marker, facecolors=sel_colors,
                            edgecolors=sel_colors, linewidths=edge_linewidth)
        handles[key] = {'collection': sc, 'mask': sel, 'fill': fill}
    return handles


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
                     attention_per_tick, saliency, masks, x_feat,
                     class_names, out_path, batch_index=0,
                     top_k_particles=20, smooth_window=5, max_heads=8):
    """Animate predictions + per-tick certainty + two η-φ overlays.

    Bottom-left: particle cloud (PID-shape / charge-fill / pt-size / pt-alpha)
    underlay + per-head cumulative attention-focus arrow tracks (Spectral
    colormap by tick, white halo + colored stroke a la
    `make_classification_gif` in continuous-thought-machines).

    Bottom-right: same particle cloud, but each particle is colored by the
    smoothed mean-over-heads attention at the current tick (viridis).

    attention_per_tick: (T, num_heads, P) for the chosen sample (already squeezed)
    saliency:           (T, P) for the chosen sample (kept for back-compat; no
                        longer rendered — the new panels visualise attention)
    masks:              (P,) 0/1
    x_feat:             (C_feat, P) standardised PF features for the chosen
                        sample, sliced to the trimmed P. Used for deta/dphi,
                        pt_log, and PID flags.
    max_heads:          cap on number of head tracks rendered on the left panel.
    """
    del top_k_particles  # legacy
    del saliency         # kept in signature for back-compat
    T, num_heads, P = attention_per_tick.shape
    these_predictions = predictions[batch_index].detach().cpu().numpy()  # (C, T)
    these_certainties = certainties[batch_index].detach().cpu().numpy()  # (2, T)
    this_target = int(targets[batch_index])
    num_classes = these_predictions.shape[0]
    short_labels = [name.replace('label_', '')[:10] for name in class_names]
    mask_np = masks.detach().cpu().numpy() if hasattr(masks, 'detach') else np.asarray(masks)
    real_idx = np.where(mask_np > 0)[0]

    x_feat_np = x_feat.detach().cpu().numpy() if hasattr(x_feat, 'detach') else np.asarray(x_feat)
    if x_feat_np.shape[1] < P:
        raise ValueError(
            f'x_feat has P={x_feat_np.shape[1]} but attention has P={P}; '
            'caller must pass the trimmed slice')
    deta = x_feat_np[_FEAT_DETA_IDX, :P]
    dphi = x_feat_np[_FEAT_DPHI_IDX, :P]
    pt_log = x_feat_np[_FEAT_PT_LOG_IDX, :P]
    pid_flags = {
        'ch': x_feat_np[_FEAT_ISCH_IDX, :P].astype(bool),
        'nh': x_feat_np[_FEAT_ISNH_IDX, :P].astype(bool),
        'ph': x_feat_np[_FEAT_ISPH_IDX, :P].astype(bool),
        'e':  x_feat_np[_FEAT_ISE_IDX,  :P].astype(bool),
        'mu': x_feat_np[_FEAT_ISMU_IDX, :P].astype(bool),
    }

    # Sliding-window smoothing along T (matches make_classification_gif).
    def _smooth(arr):
        out = np.empty_like(arr, dtype=float)
        for tt in range(arr.shape[0]):
            lo = max(0, tt - (smooth_window - 1))
            out[tt] = arr[lo:tt + 1].mean(axis=0)
        return out

    attn_per_head_smooth = _smooth(attention_per_tick.astype(float))  # (T, H, P)
    attn_smooth = attn_per_head_smooth.mean(axis=1)                   # (T, P)

    # Per-head focus particle per tick: argmax over real particles. Pads scored
    # as -inf so they are never picked.
    pad_mask = np.ones(P, dtype=bool)
    pad_mask[real_idx] = False
    attn_for_argmax = attn_per_head_smooth.copy()
    attn_for_argmax[:, :, pad_mask] = -np.inf
    focus_idx = attn_for_argmax.argmax(axis=-1)  # (T, H)

    # Shared color scale across both bottom panels. Includes both per-particle
    # attention (right panel) and per-head focus-particle attention (left arrows).
    attn_real = attn_smooth[:, real_idx]
    focus_attn_all = np.take_along_axis(
        attn_per_head_smooth, focus_idx[:, :, None], axis=-1).squeeze(-1)  # (T, H)
    pool_for_scale = np.concatenate([attn_real.ravel(), focus_attn_all.ravel()])
    attn_vmin = float(pool_for_scale.min()) if pool_for_scale.size else 0.0
    attn_vmax = float(pool_for_scale.max()) if pool_for_scale.size else 1.0
    if attn_vmax - attn_vmin < 1e-12:
        attn_vmax = attn_vmin + 1e-12

    n_heads_show = int(min(num_heads, max_heads))
    n_arrows_final = n_heads_show * max(0, T - 1)
    print(f'[plot] saliency_gif: T={T}, heads={num_heads}, P={P}, '
          f'n_real={len(real_idx)}, batch_index={batch_index}, '
          f'out={out_path}')
    print(f'[plot] saliency_gif: smooth_window={smooth_window}, '
          f'max_heads={n_heads_show}, arrows@last_frame={n_arrows_final}, '
          f'shared attention range [{attn_vmin:.3g}, {attn_vmax:.3g}]')

    # Marker size + alpha from pt_log: standardised range is roughly [-3, 3].
    pt_norm = np.clip((pt_log + 3.0) / 6.0, 0.0, 1.0)
    sizes_all = 10 + 150 * pt_norm
    alpha_all = 0.35 + 0.55 * pt_norm

    # Symmetric η-φ axis limits, mirroring plot_particle_clouds.
    if real_idx.size:
        lim = max(float(np.abs(deta[real_idx]).max()),
                  float(np.abs(dphi[real_idx]).max()),
                  0.4) * 1.1
    else:
        lim = 0.5

    cmap_attn = sns.color_palette('viridis', as_cmap=True)
    attn_norm_obj = plt.Normalize(vmin=attn_vmin, vmax=attn_vmax)

    # Precompute per-tick rgba arrays.
    attn_smooth_norm = np.clip((attn_smooth[:, real_idx] - attn_vmin) /
                               (attn_vmax - attn_vmin), 0.0, 1.0)
    right_rgba_per_tick = cmap_attn(attn_smooth_norm)  # (T, n_real, 4)

    # Arrow segments + colors, accumulated as we walk ticks.
    focus_attn_norm = np.clip((focus_attn_all - attn_vmin) /
                              (attn_vmax - attn_vmin), 0.0, 1.0)
    arrow_rgba_full = cmap_attn(focus_attn_norm)  # (T, H, 4)

    pid_real = {k: v[real_idx] for k, v in pid_flags.items()}
    deta_real = deta[real_idx]
    dphi_real = dphi[real_idx]
    sizes_real = sizes_all[real_idx]
    alpha_real = alpha_all[real_idx]

    # ---- Build figure once. ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={'height_ratios': [1, 1.3]})

    # Top-left: class probabilities (bars updated per tick, axes static).
    bar_colors = ['g' if i == this_target else 'b' for i in range(num_classes)]
    bars = axes[0, 0].bar(np.arange(num_classes), np.zeros(num_classes),
                          color=bar_colors, alpha=0.6)
    axes[0, 0].set_xticks(np.arange(num_classes))
    axes[0, 0].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_ylim([0, 1])
    title_probs = axes[0, 0].set_title(f'Class probs (tick 0/{T-1})')

    # Top-right: certainty curve + moveable cursor.
    axes[0, 1].plot(np.arange(T), these_certainties[1], 'k-', lw=2)
    vline = axes[0, 1].axvline(0, color='red', alpha=0.5)
    axes[0, 1].set_title('Certainty (1 - normalised entropy)')
    axes[0, 1].set_xlim([0, T - 1])
    axes[0, 1].set_ylim([0, 1])

    # Bottom-left: particle cloud underlay + arrow LineCollections.
    axL = axes[1, 0]
    _draw_particle_cloud(axL, deta_real, dphi_real, pid_real,
                         sizes=sizes_real, colors=None, alpha=alpha_real,
                         edge_linewidth=0.6)
    lc_halo = LineCollection([], colors='white', linewidths=2.6, alpha=0.95,
                             zorder=3)
    lc_color = LineCollection([], colors=[], linewidths=1.4, alpha=0.95,
                              zorder=4)
    axL.add_collection(lc_halo)
    axL.add_collection(lc_color)
    axL.set_xlim(-lim, lim); axL.set_ylim(-lim, lim)
    axL.set_aspect('equal')
    axL.set_xlabel(r'$\Delta\eta$', fontsize=9)
    axL.set_ylabel(r'$\Delta\varphi$', fontsize=9)
    title_left = axL.set_title(f'cls attention tracks (tick 0, heads={n_heads_show})')
    axL.grid(alpha=0.2)
    sm_L = plt.cm.ScalarMappable(cmap=cmap_attn, norm=attn_norm_obj)
    sm_L.set_array([])
    fig.colorbar(sm_L, ax=axL, fraction=0.046, pad=0.04,
                 label='cls→particle attention')

    # Bottom-right: particle cloud whose facecolors are updated per tick.
    axR = axes[1, 1]
    initial_rgba = right_rgba_per_tick[0]
    right_handles = _draw_particle_cloud(
        axR, deta_real, dphi_real, pid_real,
        sizes=sizes_real, colors=initial_rgba, alpha=alpha_real,
        edge_linewidth=0.6)
    axR.set_xlim(-lim, lim); axR.set_ylim(-lim, lim)
    axR.set_aspect('equal')
    axR.set_xlabel(r'$\Delta\eta$', fontsize=9)
    axR.set_ylabel(r'$\Delta\varphi$', fontsize=9)
    title_right = axR.set_title(f'cls→particle attention (tick 0, mean over heads)')
    axR.grid(alpha=0.2)
    sm_R = plt.cm.ScalarMappable(cmap=cmap_attn, norm=attn_norm_obj)
    sm_R.set_array([])
    fig.colorbar(sm_R, ax=axR, fraction=0.046, pad=0.04,
                 label='cls→particle attention')

    fig.suptitle(f'ground truth: {class_names[this_target]}  '
                 f'(shape=PID, fill=charge, size/α ∝ pt)', fontsize=11)
    fig.tight_layout()

    # ---- Per-tick artist updates only. ----
    accum_segments = []
    accum_colors = []
    frames = []

    def _update_right_colors(rgba_all):
        # rgba_all is (n_real, 4). For each PID group handle, slice by its mask
        # (which is N-shaped) -> then index into rgba_all (which is n_real-shaped)
        # via the same group mask but applied within real indices.
        # The handles dict was built from pid_real flags (already n_real-shaped),
        # so the stored 'mask' is n_real-aligned.
        for grp in right_handles.values():
            sel = grp['mask']
            sub = rgba_all[sel]
            coll = grp['collection']
            if grp['fill'] == 'none':
                # Hollow markers: only edgecolor encodes color; keep face='none'.
                coll.set_edgecolors(sub)
            else:
                coll.set_facecolors(sub)
                coll.set_edgecolors(sub)

    for t in tqdm(range(T), desc='Saliency frames'):
        # Probabilities
        probs = softmax(these_predictions[:, t])
        for rect, h in zip(bars.patches, probs):
            rect.set_height(float(h))
        title_probs.set_text(f'Class probs (tick {t}/{T-1})')

        # Certainty cursor
        vline.set_xdata([t, t])

        # Arrow accumulation: add new segments for tt = t-1 -> t (none on t==0).
        if t > 0:
            tt = t - 1
            for h in range(n_heads_show):
                p_prev = focus_idx[tt, h]
                p_curr = focus_idx[tt + 1, h]
                x0, y0 = float(deta[p_prev]), float(dphi[p_prev])
                x1, y1 = float(deta[p_curr]), float(dphi[p_curr])
                if x0 == x1 and y0 == y1:
                    continue
                accum_segments.append([(x0, y0), (x1, y1)])
                accum_colors.append(arrow_rgba_full[tt, h])
        lc_halo.set_segments(accum_segments)
        lc_color.set_segments(accum_segments)
        if accum_colors:
            lc_color.set_colors(accum_colors)
        title_left.set_text(
            f'cls attention tracks (tick {t}, heads={n_heads_show}, '
            f'arrows={len(accum_segments)})')

        # Right-panel particle colors
        _update_right_colors(right_rgba_per_tick[t])
        title_right.set_text(f'cls→particle attention (tick {t}, mean over heads)')

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(*reversed(fig.canvas.get_width_height()), 4)[:, :, :3]
        frames.append(img.copy())

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
