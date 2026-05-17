"""Distributed training entrypoint for ParticleCTM on JetClass.

Uses Ray Train's TorchTrainer for multi-GPU (DDP). wandb logs from rank 0.

Run:
    python -m particle_ctm.train.train --config particle_ctm/configs/default.yaml
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import yaml

# Make sibling packages importable when launched directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from particle_ctm.data.jetclass import (  # noqa: E402
    LABELS, build_dataloader, NUM_FEATURES, NUM_CLASSES,
)
from particle_ctm.models.particle_ctm import (  # noqa: E402
    ParticleCTM, get_loss, calculate_accuracy, summarize_parameters,
)


# ---------------------------------------------------------------------------
# Label-distribution → wandb bar chart
# ---------------------------------------------------------------------------
def _label_hist(labels, num_classes, key_name):
    """Build a wandb bar chart of label counts. Discrete classes → categorical
    bars (so the eye can spot a single class dominating). Title is kept
    constant across steps so the wandb panel auto-refreshes in place."""
    import numpy as _np
    import wandb as _wandb
    counts = _np.bincount(_np.asarray(labels, dtype='int64'),
                          minlength=num_classes).tolist()
    class_names = [lbl.replace('label_', '') for lbl in LABELS][:num_classes]
    data = [[name, c] for name, c in zip(class_names, counts)]
    table = _wandb.Table(data=data, columns=['class', 'count'])
    return _wandb.plot.bar(table, 'class', 'count', title=key_name)


def _confusion_plot(true_labels, pred_labels, num_classes, title):
    """2D heatmap: true label (row) vs predicted label (col). Useful to see
    whether the model's mistakes are class-specific."""
    import wandb as _wandb
    class_names = [lbl.replace('label_', '') for lbl in LABELS][:num_classes]
    return _wandb.plot.confusion_matrix(
        y_true=list(map(int, true_labels)),
        preds=list(map(int, pred_labels)),
        class_names=class_names,
        title=title,
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
def evaluate(model, loader, device, max_batches=200):
    """Iterable val loader → fixed number of batches per eval pass.

    Returns (loss, acc, true_labels, pred_labels) where the last two are 1D
    numpy arrays for distribution diagnostics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    n_batches = 0
    true_lbls, pred_lbls = [], []
    with torch.inference_mode():
        for x_feat, x_vec, mask, y in loader:
            x_feat = x_feat.to(device, non_blocking=True)
            x_vec = x_vec.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            preds, certs = model(x_feat, v=x_vec, mask=mask)
            loss, where = get_loss(preds, certs, y)
            total_loss += loss.item()
            acc = calculate_accuracy(preds, y, where)
            correct += acc * y.size(0)
            seen += y.size(0)
            n_batches += 1
            # Capture per-sample predicted class at the most-certain tick.
            B = preds.size(0)
            bi = torch.arange(B, device=preds.device)
            pred_idx = preds.argmax(dim=1)[bi, where]
            true_lbls.append(y.detach().cpu().numpy())
            pred_lbls.append(pred_idx.detach().cpu().numpy())
            if n_batches >= max_batches:
                break
    model.train()
    import numpy as _np
    return (
        total_loss / max(n_batches, 1),
        correct / max(seen, 1),
        _np.concatenate(true_lbls) if true_lbls else _np.zeros(0, dtype='int64'),
        _np.concatenate(pred_lbls) if pred_lbls else _np.zeros(0, dtype='int64'),
    )


# ---------------------------------------------------------------------------
# Per-worker training function (called by ray.train.torch.TorchTrainer)
# ---------------------------------------------------------------------------
def train_worker(cfg):
    # Ray workers don't inherit the driver's sys.path. Re-inject the project
    # root (stored in cfg by main()) and re-import package modules locally.
    import sys as _sys
    proj_root = cfg.get('_proj_root')
    if proj_root and proj_root not in _sys.path:
        _sys.path.insert(0, proj_root)

    import ray.train
    import ray.train.torch
    import wandb

    from particle_ctm.data.jetclass import build_dataloader
    from particle_ctm.models.particle_ctm import (
        ParticleCTM, get_loss, calculate_accuracy, summarize_parameters,
    )

    rank = ray.train.get_context().get_world_rank()
    world = ray.train.get_context().get_world_size()
    device = ray.train.torch.get_device()

    torch.manual_seed(cfg['train']['seed'] + rank)

    # Output paths — main() has already resolved + created run_dir.
    out_cfg = cfg.get('output', {})
    run_dir = out_cfg.get('run_dir') or os.getcwd()
    os.makedirs(run_dir, exist_ok=True)
    run_name = out_cfg.get('run_name') or os.path.basename(run_dir)

    # wandb on rank 0; its local files go inside the run dir.
    use_wandb = rank == 0 and cfg['wandb']['mode'] != 'disabled'
    if use_wandb:
        wandb.init(
            project=cfg['wandb']['project'],
            entity=cfg['wandb']['entity'],
            mode=cfg['wandb']['mode'],
            name=run_name,
            config={k: v for k, v in cfg.items() if not k.startswith('_')},
            dir=run_dir,
        )
        # Use `step` from each log dict as the x-axis for every metric.
        # Without this, wandb auto-increments its own internal counter (the
        # 0,1,2,... we were seeing on the x-axis).
        wandb.define_metric('step')
        wandb.define_metric('*', step_metric='step')

    # Model
    mcfg = cfg['model']
    model = ParticleCTM(
        input_dim=mcfg['input_dim'],
        num_classes=mcfg['num_classes'],
        pair_input_dim=mcfg['pair_input_dim'],
        particle_embed_dim=mcfg['particle_embed_dim'],
        pair_embed_dim=mcfg['pair_embed_dim'],
        num_heads=mcfg['num_heads'],
        iterations=mcfg['iterations'],
        n_global=mcfg['n_global'],
        n_synch_global=mcfg['n_synch_global'],
        memory_length=mcfg['memory_length'],
        dropout=mcfg['dropout'],
        trim=mcfg['trim'],
    ).to(device)
    if rank == 0:
        print(summarize_parameters(model))
    model = ray.train.torch.prepare_model(model)

    # Clamp the global pool's decay_params before every forward (CTM stability).
    base_model = model.module if hasattr(model, 'module') else model
    _decay_params = base_model.global_pool.decay_params

    def clamp_decay_params(_module, _input):
        with torch.no_grad():
            _decay_params.data.clamp_(0, 15)
    model.register_forward_pre_hook(clamp_decay_params)

    # Data — file-level sharding by rank.
    shuffle_buf = cfg['data'].get('shuffle_buffer_size', 20000)
    num_concurrent = cfg['data'].get('num_concurrent_files', 10)
    rows_per_visit = cfg['data'].get('rows_per_file_visit', 10000)
    train_loader = build_dataloader(
        cfg['data']['train_glob'],
        batch_size=cfg['train']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        max_num_particles=cfg['data']['max_num_particles'],
        shuffle=True,
        rank=rank, world_size=world, seed=cfg['train']['seed'],
        shuffle_buffer_size=shuffle_buf,
        num_concurrent_files=num_concurrent,
        rows_per_file_visit=rows_per_visit,
    )
    val_loader = build_dataloader(
        cfg['data']['val_glob'],
        batch_size=cfg['train']['batch_size'],
        num_workers=max(1, cfg['data']['num_workers'] // 2),
        max_num_particles=cfg['data']['max_num_particles'],
        shuffle=False,
        rank=rank, world_size=world, seed=cfg['train']['seed'],
    )

    # Optim + scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg['train']['lr'],
                                  weight_decay=cfg['train']['weight_decay'],
                                  eps=1e-8)
    warmup = cfg['train']['warmup_steps']
    total = cfg['train']['total_steps']

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        # cosine decay
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler('cuda') if cfg['train']['use_amp'] else None

    best_val_acc = -float('inf')
    best_ckpt = os.path.join(run_dir, out_cfg.get('ckpt_name', 'best.pt'))
    last_ckpt = os.path.join(run_dir, 'last.pt')

    model.train()
    iterator = iter(train_loader)
    t0 = time.time()
    # Rolling stats for data-pipeline starvation diagnostics.
    data_wait_ms_acc = 0.0
    data_wait_ms_max = 0.0
    data_wait_n = 0
    for step in range(total):
        t_wait = time.time()
        try:
            x_feat, x_vec, mask, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x_feat, x_vec, mask, y = next(iterator)
        data_wait_ms = (time.time() - t_wait) * 1000.0
        data_wait_ms_acc += data_wait_ms
        data_wait_ms_max = max(data_wait_ms_max, data_wait_ms)
        data_wait_n += 1

        x_feat = x_feat.to(device, non_blocking=True)
        x_vec = x_vec.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds, certs = model(x_feat, v=x_vec, mask=mask)
                loss, where = get_loss(preds, certs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            # Only advance the scheduler when the GradScaler actually stepped
            # the optimiser (otherwise PyTorch warns about scheduler stepping
            # before optimiser).
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            preds, certs = model(x_feat, v=x_vec, mask=mask)
            loss, where = get_loss(preds, certs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            optimizer.step()
            scheduler.step()

        if rank == 0 and step % cfg['train']['log_every'] == 0:
            acc = calculate_accuracy(preds, y, where)
            ips = (step + 1) * cfg['train']['batch_size'] * world / max(time.time() - t0, 1e-6)
            data_wait_avg = data_wait_ms_acc / max(1, data_wait_n)
            data_wait_peak = data_wait_ms_max
            data_wait_ms_acc = 0.0
            data_wait_ms_max = 0.0
            data_wait_n = 0
            print(f'step {step:6d} loss {loss.item():.4f} acc {acc:.4f} '
                  f'lr {scheduler.get_last_lr()[0]:.2e} ips {ips:.0f} '
                  f'wait {data_wait_avg:.1f}/{data_wait_peak:.1f}ms', flush=True)
            if use_wandb:
                # Class distribution of the current batch (sanity check that
                # the shuffle buffer is producing diverse batches).
                B = preds.size(0)
                bi = torch.arange(B, device=preds.device)
                pred_idx_now = preds.argmax(dim=1)[bi, where]
                true_now = y.detach().cpu().numpy()
                pred_now = pred_idx_now.detach().cpu().numpy()
                wandb.log({
                    'train/loss': loss.item(),
                    'train/acc': float(acc),
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/ips': ips,
                    # Data-pipeline starvation: time spent blocked on the
                    # DataLoader queue. <5 ms = healthy prefetch; persistent
                    # >50 ms = workers can't keep up; periodic spikes = file-
                    # load stalls (see slot-exhaustion discussion in
                    # data/jetclass.py).
                    'data/wait_ms_avg': data_wait_avg,
                    'data/wait_ms_peak': data_wait_peak,
                    'train/true_label_dist':
                        _label_hist(true_now, mcfg['num_classes'], 'train true labels'),
                    'train/confusion':
                        _confusion_plot(true_now, pred_now, mcfg['num_classes'],
                                        'train true vs predicted'),
                    'step': step,
                })

        if step > 0 and step % cfg['train']['val_every'] == 0:
            val_loss, val_acc, val_true, val_pred = evaluate(model, val_loader, device)
            # Reduce across workers for a fair number on rank 0.
            if world > 1:
                t = torch.tensor([val_loss, val_acc, 1.0], device=device)
                torch.distributed.all_reduce(t)
                val_loss = (t[0] / t[2]).item()
                val_acc = (t[1] / t[2]).item()
            if rank == 0:
                print(f'[val @ {step}] loss {val_loss:.4f} acc {val_acc:.4f} '
                      f'best {best_val_acc:.4f}', flush=True)
                if use_wandb:
                    wandb.log({
                        'val/loss': val_loss, 'val/acc': val_acc,
                        'val/true_label_dist':
                            _label_hist(val_true, mcfg['num_classes'], 'val true labels'),
                        'val/pred_label_dist':
                            _label_hist(val_pred, mcfg['num_classes'], 'val predicted labels'),
                        'step': step,
                    })
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'model_state_dict': base_model.state_dict(),
                        'config': cfg,
                        'step': step,
                        'val_acc': best_val_acc,
                    }, best_ckpt)
                    if use_wandb:
                        wandb.run.summary['best_val_acc'] = best_val_acc
                        wandb.run.summary['best_step'] = step

    # Optionally save the final weights too (regardless of best-acc tracking).
    if rank == 0 and out_cfg.get('save_last', True):
        torch.save({
            'model_state_dict': base_model.state_dict(),
            'config': cfg,
            'step': cfg['train']['total_steps'],
            'val_acc': best_val_acc,
        }, last_ckpt)

    if rank == 0 and use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(_PROJ_ROOT, 'particle_ctm', 'configs', 'default.yaml'))
    parser.add_argument('--single-gpu', action='store_true',
                        help='Run train_worker directly (skip Ray). Useful for debugging.')
    parser.add_argument('--final-test', action='store_true',
                        help='After training, run particle_ctm.eval.test.run_test on '
                             'the best checkpoint and write metrics + plots into '
                             '<run_dir>/test.')
    parser.add_argument('--final-test-only', action='store_true',
                        help='Skip training; just run the final test using the '
                             'existing best.pt for this config.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Sanity-defaults from data module.
    cfg['model']['input_dim'] = cfg['model'].get('input_dim', NUM_FEATURES)
    cfg['model']['num_classes'] = cfg['model'].get('num_classes', NUM_CLASSES)
    # Pass the project root to train_worker so each Ray worker can put it on
    # sys.path before importing the `particle_ctm` package.
    cfg['_proj_root'] = _PROJ_ROOT

    # Resolve relative data globs against the config's directory so Ray workers
    # (which run in a staged tmp cwd) still find the ROOT files on the shared FS.
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    for key in ('train_glob', 'val_glob', 'test_glob'):
        g = cfg['data'].get(key)
        if g and not os.path.isabs(g):
            cfg['data'][key] = os.path.abspath(os.path.join(cfg_dir, '..', g))

    # Resolve output dir and freeze the run name now so every Ray worker (and
    # any subsequent restart) writes into the same directory.
    out_cfg = cfg.setdefault('output', {})
    out_dir = out_cfg.get('dir', 'runs')
    if not os.path.isabs(out_dir):
        out_dir = os.path.abspath(os.path.join(cfg_dir, '..', out_dir))
    run_name = out_cfg.get('run_name') or \
        f"particle-ctm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg['output']['dir'] = out_dir
    cfg['output']['run_name'] = run_name
    cfg['output']['run_dir'] = run_dir
    cfg['output'].setdefault('ckpt_name', 'best.pt')
    cfg['output'].setdefault('save_last', True)
    # Snapshot the resolved config next to the checkpoints.
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith('_')}, f)
    print(f'[output] run dir: {run_dir}', flush=True)

    # --final-test-only short-circuits training.
    if args.final_test_only:
        _run_final_test(cfg)
        return

    if args.single_gpu:
        train_worker(cfg)
        if args.final_test:
            _run_final_test(cfg)
        return

    import ray
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig

    # Stage just the source into a temp dir as `<tmp>/particle_ctm/` and ship
    # that as Ray's working_dir. Keeps the upload to a few MB and avoids
    # walking the 240 GB particle_transformer tree.
    import shutil
    import tempfile
    stage = tempfile.mkdtemp(prefix='particle_ctm_ray_')
    pkg_src = os.path.join(_PROJ_ROOT, 'particle_ctm')
    shutil.copytree(
        pkg_src,
        os.path.join(stage, 'particle_ctm'),
        ignore=shutil.ignore_patterns(
            'datasets', 'ckpts', 'runs', '__pycache__', '*.pyc', '*.root',
        ),
    )
    print(f'[ray] staged source at {stage}', flush=True)

    ray.init(
        ignore_reinit_error=True,
        runtime_env={'working_dir': stage},
        include_dashboard=True,
        dashboard_host='0.0.0.0',
        dashboard_port=8265,
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_worker,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=cfg['ray']['num_workers'],
            use_gpu=cfg['ray']['use_gpu'],
            resources_per_worker=cfg['ray']['resources_per_worker'],
        ),
    )
    result = trainer.fit()
    print('Ray run finished:', result)

    if args.final_test:
        _run_final_test(cfg)


def _run_final_test(cfg):
    """Invoke the standalone eval module on this run's best.pt."""
    from particle_ctm.eval.test import run_test
    run_dir = cfg['output']['run_dir']
    ckpt_name = cfg['output'].get('ckpt_name', 'best.pt')
    ckpt_path = os.path.join(run_dir, ckpt_name)
    output_dir = os.path.join(run_dir, 'test')
    if not os.path.isfile(ckpt_path):
        print(f'[final-test] no checkpoint at {ckpt_path}, skipping.')
        return
    print(f'[final-test] running on {ckpt_path} → {output_dir}')
    run_test(cfg, ckpt_path, output_dir)


if __name__ == '__main__':
    main()
