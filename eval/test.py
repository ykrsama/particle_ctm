"""Standalone test/evaluation module for ParticleCTM.

Run:
    python particle_ctm/eval/test.py \
        --config particle_ctm/configs/test.yaml \
        --checkpoint runs/<run_name>/best.pt \
        --output-dir runs/<run_name>/test

Produces (all under --output-dir):
    metrics.json         test_acc, test_auc (macro), per-class AUC
    roc.png              one-vs-rest ROC for all 10 classes
    prc.png              one-vs-rest PRC
    confusion_matrix.png 2D heatmap
    particle_clouds.png  10 jet types as point clouds (η, φ, PID, q, displacement)
    certainty_vs_tick.png histogram: per-tick count of jets with certainty>0.8
    saliency.gif + neural_dynamics.png  from utils/visualization.py

Usage notes:
    - Loads config + checkpoint; both must match the model architecture.
    - Evaluates on cfg.data.test_glob.
    - Single-GPU only — keep things simple for offline analysis.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
_PROJ_ROOT = os.path.abspath(os.path.join(_PKG_ROOT, '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from particle_ctm.data.jetclass import (  # noqa: E402
    LABELS, NUM_CLASSES, _BASE_PARTICLE_VARS, _JET_FEATURES, _read_root,
    build_dataloader,
)
from particle_ctm.models.particle_ctm import (  # noqa: E402
    ParticleCTM, calculate_accuracy, get_loss,
)


CLASS_NAMES = [lbl.replace('label_', '') for lbl in LABELS]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def build_model_from_cfg(cfg, device):
    mcfg = cfg['model']
    model = ParticleCTM(
        input_dim=mcfg['input_dim'],
        num_classes=mcfg['num_classes'],
        pair_input_dim=mcfg['pair_input_dim'],
        pair_extra_dim=mcfg['pair_extra_dim'],
        embed_dims=tuple(mcfg['embed_dims']),
        pair_embed_dims=tuple(mcfg['pair_embed_dims']),
        use_pre_activation_pair=mcfg['use_pre_activation_pair'],
        num_heads=mcfg['num_heads'],
        iterations=mcfg['iterations'],
        memory_length=mcfg['memory_length'],
        memory_hidden_dims=mcfg.get('memory_hidden_dims', None),
        d_model_qkv=mcfg['d_model_qkv'],
        d_model_o=mcfg['d_model_o'],
        n_synch_qkv=mcfg['n_synch_qkv'],
        n_synch_o=mcfg['n_synch_o'],
        dropout=mcfg['dropout'],
        trim=mcfg['trim'],
        fc_params=tuple(tuple(x) for x in mcfg['fc_params']),
        activation=mcfg['activation'],
    ).to(device)
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict']
    # Strip DDP "module." prefix if present.
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    # strict=False so checkpoints saved before architectural pruning (e.g.
    # removed `prev_to_sync_q/k/v` after commit cb42bec) still load. Surface
    # missing/unexpected keys so silent drift remains visible.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[test] missing keys ({len(missing)}): {missing}')
    if unexpected:
        print(f'[test] unexpected keys ({len(unexpected)}): {unexpected}')
    return ckpt


# ---------------------------------------------------------------------------
# Inference: collect predictions, certainties, raw samples for plots
# ---------------------------------------------------------------------------
def count_total_batches(test_glob, batch_size):
    """Cheaply sum entries across all test ROOT files (num_entries is a header
    read, no array decoding). Returns floor(total_entries / batch_size) to
    match the DataLoader's drop_last=True."""
    import glob as _glob
    import uproot
    files = sorted(_glob.glob(test_glob))
    total = 0
    for fp in files:
        with uproot.open(fp) as f:
            total += int(f['tree'].num_entries)
    return total // batch_size, total


@torch.inference_mode()
def run_inference(model, loader, device, num_classes,
                  certainty_threshold=0.8, viz_per_class=1, max_batches=None,
                  total_batches=None):
    """Single sweep through the test loader.

    Returns:
        all_preds_softmax: (N_total, C) per-jet predicted probs at most-certain tick
        all_targets:       (N_total,)
        cert_above_per_tick: (T,) count of (jet, tick) where certainty > threshold
        per_class_samples: dict[class_idx → list of {x_feat, x_vec, mask, target}]
                           one or more per class for the particle-cloud plot.
    """
    model.eval()
    all_preds, all_tgts = [], []
    cert_above = None  # filled after we see T from the first batch
    per_class_samples = defaultdict(list)
    n_batches = 0
    n_seen = 0
    n_correct = 0

    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    elif max_batches is not None:
        total_batches = max_batches
    pbar = tqdm(loader, desc='infer', dynamic_ncols=True, unit='batch',
                total=total_batches)
    for x_feat, x_vec, mask, y in pbar:
        x_feat = x_feat.to(device, non_blocking=True)
        x_vec = x_vec.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        preds, certs = model(x_feat, v=x_vec, mask=mask)
        # preds: (B, C, T), certs: (B, 2, T)
        most_cert_idx = certs[:, 1].argmax(dim=-1)             # (B,)
        bi = torch.arange(preds.size(0), device=preds.device)
        logits = preds[bi, :, most_cert_idx]                   # (B, C)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_preds.append(probs)
        all_tgts.append(y.numpy())

        y_np = y.numpy()
        n_seen += y_np.shape[0]
        n_correct += int((probs.argmax(axis=1) == y_np).sum())
        pbar.set_postfix(n=n_seen, acc=f'{n_correct / max(n_seen, 1):.4f}')

        if cert_above is None:
            cert_above = torch.zeros(preds.size(-1), dtype=torch.long)
        cert_above += (certs[:, 1] > certainty_threshold).sum(dim=0).cpu()

        # Stash a few raw samples per class for the particle-cloud plot.
        for cls in range(num_classes):
            need = viz_per_class - len(per_class_samples[cls])
            if need <= 0:
                continue
            idxs = (y == cls).nonzero(as_tuple=True)[0][:need]
            for i in idxs.tolist():
                per_class_samples[cls].append({
                    'x_feat': x_feat[i].detach().cpu().numpy(),
                    'x_vec': x_vec[i].detach().cpu().numpy(),
                    'mask': mask[i].detach().cpu().numpy(),
                    'target': cls,
                })

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_tgts, axis=0),
        cert_above.numpy(),
        dict(per_class_samples),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(probs, targets, out_path, class_names):
    plt.figure(figsize=(7, 6))
    for c in range(probs.shape[1]):
        y_true = (targets == c).astype(int)
        if y_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, c])
        auc = roc_auc_score(y_true, probs[:, c])
        plt.plot(fpr, tpr, lw=1, label=f'{class_names[c]} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (one-vs-rest)')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_prc(probs, targets, out_path, class_names):
    plt.figure(figsize=(7, 6))
    for c in range(probs.shape[1]):
        y_true = (targets == c).astype(int)
        if y_true.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true, probs[:, c])
        ap = average_precision_score(y_true, probs[:, c])
        plt.plot(recall, precision, lw=1, label=f'{class_names[c]} (AP={ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (one-vs-rest)')
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_confusion(probs, targets, out_path, class_names):
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=cm_norm, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix (row-normalised)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_certainty_histogram(cert_above_per_tick, out_path, threshold=0.8):
    T = len(cert_above_per_tick)
    plt.figure(figsize=(9, 4))
    plt.bar(np.arange(T), cert_above_per_tick, color='steelblue', alpha=0.85)
    plt.xlabel('tick')
    plt.ylabel(f'# jets with certainty > {threshold}')
    plt.title(f'Confidence (>{threshold}) distribution across CTM ticks')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# Particle-cloud plot
# ---------------------------------------------------------------------------
def _read_raw_one_jet_per_class(test_glob, num_classes, max_num_particles=128):
    """Walk a few ROOT shards from test_glob until we have one jet per class.

    Returns dict[cls -> dict with raw arrays]: px, py, pz, energy, eta, phi,
    deta, dphi, charge, isChargedHadron, isNeutralHadron, isPhoton, isElectron,
    isMuon, d0val, dzval, mask.
    """
    import glob as _glob
    files = sorted(_glob.glob(test_glob))
    if not files:
        raise FileNotFoundError(f'No ROOT files matched: {test_glob}')

    needed = set(range(num_classes))
    out = {}
    for fp in files:
        x_part, x_jet, y = _read_root(
            fp, max_num_particles=max_num_particles,
            particle_features=_BASE_PARTICLE_VARS,
            jet_features=_JET_FEATURES, labels=LABELS,
        )
        idx = {n: i for i, n in enumerate(_BASE_PARTICLE_VARS)}
        labels_argmax = y.argmax(axis=1)
        for cls in list(needed):
            hits = np.where(labels_argmax == cls)[0]
            if len(hits) == 0:
                continue
            i = int(hits[0])
            px, py, pz, e = (x_part[i, idx[k]] for k in
                             ('part_px', 'part_py', 'part_pz', 'part_energy'))
            pt = np.hypot(px, py)
            mask = pt > 0
            # eta/phi from 4-vector (avoids divide-by-zero on padded slots)
            with np.errstate(divide='ignore', invalid='ignore'):
                p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
                eta = np.where(p > 0, 0.5 * np.log((p + pz) / (p - pz).clip(1e-9)), 0)
                phi = np.arctan2(py, px)
            out[cls] = {
                'px': px, 'py': py, 'pz': pz, 'energy': e,
                'eta': eta, 'phi': phi,
                'deta': x_part[i, idx['part_deta']],
                'dphi': x_part[i, idx['part_dphi']],
                'charge': x_part[i, idx['part_charge']],
                'isChargedHadron': x_part[i, idx['part_isChargedHadron']],
                'isNeutralHadron': x_part[i, idx['part_isNeutralHadron']],
                'isPhoton': x_part[i, idx['part_isPhoton']],
                'isElectron': x_part[i, idx['part_isElectron']],
                'isMuon': x_part[i, idx['part_isMuon']],
                'd0val': x_part[i, idx['part_d0val']],
                'dzval': x_part[i, idx['part_dzval']],
                'mask': mask,
            }
            needed.discard(cls)
        if not needed:
            break
    return out


def plot_particle_clouds(test_glob, num_classes, out_path, max_num_particles=128):
    """One jet per class, particles drawn at (η, φ) (jet-relative deta/dphi).

    Marker shape: hadron=circle, lepton (electron/muon)=triangle (down for e,
    up for mu), photon=pentagon. Filled = charged, hollow = neutral.
    Size ∝ energy. Color = displacement |d0|² + |dz|² (bluer = larger).
    """
    samples = _read_raw_one_jet_per_class(test_glob, num_classes,
                                          max_num_particles=max_num_particles)
    if not samples:
        print('[test] particle_clouds: no samples loaded, skipping')
        return None

    cols = 5
    rows = int(math.ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    cmap = sns.color_palette('Blues', as_cmap=True)
    # Compute global displacement scale so colors are comparable across panels.
    all_disp = []
    for s in samples.values():
        d = np.sqrt(s['d0val'][s['mask']] ** 2 + s['dzval'][s['mask']] ** 2)
        if len(d):
            all_disp.append(d)
    disp_max = float(np.percentile(np.concatenate(all_disp), 99)) if all_disp else 1.0
    disp_max = max(disp_max, 1e-3)

    for cls in range(num_classes):
        ax = axes[cls // cols, cls % cols]
        ax.set_title(CLASS_NAMES[cls], fontsize=10)
        if cls not in samples:
            ax.text(0.5, 0.5, 'no sample', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_xticks([]); ax.set_yticks([])
            continue
        s = samples[cls]
        m = s['mask']
        if not m.any():
            ax.text(0.5, 0.5, 'empty jet', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        # Use jet-relative coords (more compact, paper-style).
        x = s['deta'][m]
        y = s['dphi'][m]
        energy = np.clip(s['energy'][m], a_min=1e-3, a_max=None)
        sizes = 6 + 90 * (energy / max(energy.max(), 1e-3))
        disp = np.sqrt(s['d0val'][m] ** 2 + s['dzval'][m] ** 2)
        colors = cmap(np.clip(disp / disp_max, 0, 1))

        # 5 categories: charged hadron, neutral hadron, electron, muon, photon, other
        ch = s['isChargedHadron'][m].astype(bool)
        nh = s['isNeutralHadron'][m].astype(bool)
        ee = s['isElectron'][m].astype(bool)
        mu = s['isMuon'][m].astype(bool)
        ph = s['isPhoton'][m].astype(bool)
        other = ~(ch | nh | ee | mu | ph)

        groups = [
            (ch, 'o', 'full',  'charged hadron'),
            (nh, 'o', 'none',  'neutral hadron'),
            (ee, 'v', 'full',  'electron'),
            (mu, '^', 'full',  'muon'),
            (ph, 'p', 'none',  'photon'),
            (other, 'x', 'full', 'other'),
        ]
        for sel, marker, fill, _label in groups:
            if not sel.any():
                continue
            facecolor = colors[sel] if fill == 'full' else 'none'
            edgecolor = colors[sel]
            ax.scatter(x[sel], y[sel], s=sizes[sel],
                       marker=marker, facecolors=facecolor,
                       edgecolors=edgecolor, linewidths=1.0)

        lim = max(abs(x).max(), abs(y).max(), 0.5) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\Delta\eta$', fontsize=9)
        ax.set_ylabel(r'$\Delta\varphi$', fontsize=9)
        ax.grid(alpha=0.2)

    # Hide any leftover axes.
    for k in range(num_classes, rows * cols):
        axes[k // cols, k % cols].axis('off')

    # Legend (shape/fill = particle ID), color encodes displacement.
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color='dimgray', label='charged hadron'),
        plt.Line2D([], [], marker='o', linestyle='', markerfacecolor='none',
                   markeredgecolor='dimgray', label='neutral hadron'),
        plt.Line2D([], [], marker='v', linestyle='', color='dimgray', label='electron'),
        plt.Line2D([], [], marker='^', linestyle='', color='dimgray', label='muon'),
        plt.Line2D([], [], marker='p', linestyle='', markerfacecolor='none',
                   markeredgecolor='dimgray', label='photon'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Jets as particle clouds — size ∝ energy, color ∝ displacement', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Saliency / neural dynamics — delegate to utils/visualization.py
# ---------------------------------------------------------------------------
def run_visualization_module(model, per_class_samples, out_dir, device):
    """Build one small batch (≤8 jets) and produce the saliency GIF + neural-
    dynamics plot from particle_ctm.utils.visualization.
    """
    from particle_ctm.utils.visualization import (
        compute_cls_saliency, make_saliency_gif, plot_neural_dynamics_simple,
    )

    pool = []
    for cls in sorted(per_class_samples):
        for s in per_class_samples[cls]:
            pool.append(s)
            if len(pool) >= 8:
                break
        if len(pool) >= 8:
            break
    if not pool:
        print('[test] visualization: no samples available')
        return

    x_feat = torch.from_numpy(np.stack([s['x_feat'] for s in pool])).to(device)
    x_vec = torch.from_numpy(np.stack([s['x_vec'] for s in pool])).to(device)
    mask = torch.from_numpy(np.stack([s['mask'] for s in pool])).to(device)
    targets = torch.tensor([s['target'] for s in pool])

    # Need grad for saliency.
    x_feat.requires_grad_(False)
    x_vec.requires_grad_(False)
    model.train()  # enable autograd graph; we use no_grad inside utility helpers
    preds, certs, attn_stack, tok_acts, saliency = compute_cls_saliency(
        model, x_feat, x_vec, mask)
    model.eval()

    # SequenceTrimmer may have shrunk P inside the model. Saliency / attention
    # are shaped to the trimmed P; build a matching mask: first `real` slots
    # are 1, rest are padding.
    P_trimmed = saliency.shape[-1]
    real_count = int(mask[0, 0].sum().item())
    real_count = min(real_count, P_trimmed)
    viz_mask = np.zeros(P_trimmed, dtype=np.float32)
    viz_mask[:real_count] = 1.0

    make_saliency_gif(
        preds, certs, targets,
        attention_per_tick=attn_stack[:, 0],
        saliency=saliency[:, 0],
        masks=viz_mask,
        class_names=CLASS_NAMES,
        out_path=os.path.join(out_dir, 'saliency.gif'),
        batch_index=0,
        top_k_particles=20,
    )
    plot_neural_dynamics_simple(
        tok_acts, os.path.join(out_dir, 'neural_dynamics.png'),
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def run_test(cfg, ckpt_path, output_dir, device=None,
             max_batches=None, certainty_threshold=0.8, viz_per_class=1,
             batch_size=1024):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    print(f'[test] device {device}  ckpt {ckpt_path}')
    model = build_model_from_cfg(cfg, device)
    ck = load_checkpoint(model, ckpt_path, device)
    print(f"[test] loaded checkpoint @ step {ck.get('step', '?')} "
          f"val_acc {ck.get('val_acc', '?')}")

    # Build the test loader (single rank; the iterable dataset already handles
    # row striding so passing rank=0 world=1 just streams everything once).
    test_glob = cfg['data']['test_glob']
    test_loader = build_dataloader(
        test_glob,
        batch_size=batch_size,
        num_workers=max(1, cfg['data']['num_workers']),
        max_num_particles=cfg['data']['max_num_particles'],
        shuffle=True,
        rank=0, world_size=1, seed=cfg['train']['seed'],
    )

    try:
        total_batches, total_entries = count_total_batches(test_glob, batch_size)
        print(f'[test] eval set: {total_entries} jets -> {total_batches} batches '
              f'@ batch_size={batch_size} (drop_last=True)')
    except Exception as e:
        print(f'[test] could not pre-count entries ({e}); progress bar will be untotaled')
        total_batches = None

    probs, targets, cert_above, per_class_samples = run_inference(
        model, test_loader, device, NUM_CLASSES,
        certainty_threshold=certainty_threshold,
        viz_per_class=viz_per_class,
        max_batches=max_batches,
        total_batches=total_batches,
    )

    # Metrics
    preds_argmax = probs.argmax(axis=1)
    test_acc = float((preds_argmax == targets).mean())
    try:
        test_auc = float(roc_auc_score(targets, probs, multi_class='ovr',
                                       average='macro'))
    except ValueError:  # not all classes present
        test_auc = float('nan')
    per_class_auc = {}
    for c in range(NUM_CLASSES):
        y_true = (targets == c).astype(int)
        if y_true.sum() == 0:
            per_class_auc[CLASS_NAMES[c]] = float('nan')
        else:
            per_class_auc[CLASS_NAMES[c]] = float(roc_auc_score(y_true, probs[:, c]))

    metrics = {
        'test_accuracy': test_acc,
        'test_auc_macro_ovr': test_auc,
        'per_class_auc': per_class_auc,
        'n_eval_samples': int(targets.shape[0]),
        'certainty_threshold': certainty_threshold,
        'certainty_above_threshold_per_tick': cert_above.tolist(),
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[test] acc {test_acc:.4f}  AUC(macro,OvR) {test_auc:.4f}")

    # Plots
    plot_roc(probs, targets,        os.path.join(output_dir, 'roc.png'),               CLASS_NAMES)
    plot_prc(probs, targets,        os.path.join(output_dir, 'prc.png'),               CLASS_NAMES)
    plot_confusion(probs, targets,  os.path.join(output_dir, 'confusion_matrix.png'),  CLASS_NAMES)
    plot_certainty_histogram(cert_above, os.path.join(output_dir, 'certainty_vs_tick.png'),
                             threshold=certainty_threshold)
    try:
        plot_particle_clouds(test_glob, NUM_CLASSES,
                             os.path.join(output_dir, 'particle_clouds.png'),
                             max_num_particles=cfg['data']['max_num_particles'])
    except Exception as e:
        print(f'[test] particle_clouds failed: {e}')

    # Saliency + neural-dynamics from utils/visualization.py.
    try:
        run_visualization_module(model, per_class_samples, output_dir, device)
    except Exception as e:
        print(f'[test] visualization module failed: {e}')

    print(f'[test] outputs written to {output_dir}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description='ParticleCTM standalone tester')
    parser.add_argument('--config', required=True,
                        help='Path to the yaml config (same one used for training).')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to best.pt. If omitted, looks under '
                             'output.dir/output.run_name/output.ckpt_name.')
    parser.add_argument('--output-dir', default=None,
                        help='Where to write metrics + plots. Defaults to '
                             '<run_dir>/test.')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Cap eval to N batches (debugging).')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Eval batch size (overrides cfg.train.batch_size).')
    parser.add_argument('--certainty-threshold', type=float, default=0.8)
    parser.add_argument('--viz-per-class', type=int, default=1,
                        help='Raw jets per class to stash for viz module.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve data globs relative to the config's directory parent (same rule
    # as train.py — yaml lives in particle_ctm/configs/, data lives in
    # particle_ctm/datasets/, so '..' from configs/ is particle_ctm/).
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ('train_glob', 'val_glob', 'test_glob'):
        g = cfg['data'].get(key)
        if g and not os.path.isabs(g):
            cfg['data'][key] = os.path.abspath(os.path.join(cfg_dir, '..', g))

    out_cfg = cfg.get('output', {}) or {}
    out_dir = out_cfg.get('dir', 'runs')
    if not os.path.isabs(out_dir):
        out_dir = os.path.abspath(os.path.join(cfg_dir, '..', out_dir))
    run_name = out_cfg.get('run_name')

    if args.checkpoint:
        ckpt_path = os.path.abspath(args.checkpoint)
    else:
        if not run_name:
            raise SystemExit(
                'Must pass --checkpoint, or set output.run_name in the config.')
        ckpt_path = os.path.join(out_dir, run_name, out_cfg.get('ckpt_name', 'best.pt'))

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # Default to <run_dir>/test if we can infer the run dir from the ckpt.
        output_dir = os.path.join(os.path.dirname(ckpt_path), 'test')

    cfg['model']['input_dim'] = cfg['model'].get('input_dim', 17)
    cfg['model']['num_classes'] = cfg['model'].get('num_classes', NUM_CLASSES)

    run_test(cfg, ckpt_path, output_dir,
             max_batches=args.max_batches,
             certainty_threshold=args.certainty_threshold,
             viz_per_class=args.viz_per_class,
             batch_size=args.batch_size)


if __name__ == '__main__':
    main()
