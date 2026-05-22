"""JetClass dataset loader.

Reads ROOT shards via `particle_transformer/dataloader.py::read_file` and
applies the standardisation specified in `JetClass_full.yaml`. Builds three
tensors per jet:

    x_features : (C_feat, P)  17 standardised particle features
    x_vectors  : (4, P)        (px, py, pz, energy) — fed to PairEmbed
    x_mask     : (1, P)        1 = real particle, 0 = padding
"""

import glob
import math
import os
import random
import sys

import awkward as ak
import numpy as np
import torch
import uproot
import vector
from torch.utils.data import IterableDataset

vector.register_awkward()


# ---------------------------------------------------------------------------
# read_file: inlined from particle_transformer/dataloader.py so this project is
# self-contained (no sibling-tree dependency at runtime — important for Ray's
# working_dir which can't ship the 240 GB particle_transformer/ tree).
# ---------------------------------------------------------------------------
def _read_root(filepath, max_num_particles=128, particle_features=None,
               jet_features=None, labels=None):
    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        if isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x

    table = uproot.open(filepath)['tree'].arrays()
    p4 = vector.zip({'px': table['part_px'], 'py': table['part_py'],
                     'pz': table['part_pz'], 'energy': table['part_energy']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi

    x_part = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles))
                       for n in particle_features], axis=1)
    x_jet = np.stack([ak.to_numpy(table[n]).astype('float32')
                      for n in jet_features], axis=1)
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)
    return x_part, x_jet, y


# ---------------------------------------------------------------------------
# Standardisation parameters straight out of JetClass_full.yaml (`inputs.pf_features`)
# Each row: (variable_name, subtract, multiply, clip_min, clip_max)
# `null` in YAML → no subtract / multiply 1.0 / default clip (-5, 5).
# ---------------------------------------------------------------------------
PF_FEATURES = [
    ('part_pt_log',       1.7,  0.7, -5.0, 5.0),
    ('part_e_log',        2.0,  0.7, -5.0, 5.0),
    ('part_logptrel',    -4.7,  0.7, -5.0, 5.0),
    ('part_logerel',     -4.7,  0.7, -5.0, 5.0),
    ('part_deltaR',       0.2,  4.0, -5.0, 5.0),
    ('part_charge',       0.0,  1.0, -5.0, 5.0),
    ('part_isChargedHadron', 0.0, 1.0, -5.0, 5.0),
    ('part_isNeutralHadron', 0.0, 1.0, -5.0, 5.0),
    ('part_isPhoton',     0.0,  1.0, -5.0, 5.0),
    ('part_isElectron',   0.0,  1.0, -5.0, 5.0),
    ('part_isMuon',       0.0,  1.0, -5.0, 5.0),
    ('part_d0',           0.0,  1.0, -5.0, 5.0),
    ('part_d0err',        0.0,  1.0,  0.0, 1.0),
    ('part_dz',           0.0,  1.0, -5.0, 5.0),
    ('part_dzerr',        0.0,  1.0,  0.0, 1.0),
    ('part_deta',         0.0,  1.0, -5.0, 5.0),
    ('part_dphi',         0.0,  1.0, -5.0, 5.0),
]

# Source variables that `read_file` must load from the ROOT tree so the
# JetClass_full derived features can be computed.
_BASE_PARTICLE_VARS = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi',
    'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon',
    'part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr',
]

_JET_FEATURES = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']

LABELS = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
    'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl',
]


def _derive_features(x_part, x_jet):
    """Build the 17-feature standardised array + the (px,py,pz,E) vectors.

    x_part: (N, len(_BASE_PARTICLE_VARS), P) — output of read_file
    x_jet:  (N, 4)
    """
    idx = {n: i for i, n in enumerate(_BASE_PARTICLE_VARS)}
    px = x_part[:, idx['part_px']]
    py = x_part[:, idx['part_py']]
    pz = x_part[:, idx['part_pz']]
    energy = x_part[:, idx['part_energy']]

    jet_pt = x_jet[:, 0:1]
    jet_energy = x_jet[:, 3:4]

    pt = np.hypot(px, py)
    # Avoid log(0) on padded particles by clamping; mask handles them downstream.
    safe_pt = np.where(pt > 0, pt, 1e-9)
    safe_e = np.where(energy > 0, energy, 1e-9)
    pt_log = np.log(safe_pt)
    e_log = np.log(safe_e)
    logptrel = np.log(safe_pt / np.where(jet_pt > 0, jet_pt, 1e-9))
    logerel = np.log(safe_e / np.where(jet_energy > 0, jet_energy, 1e-9))
    deta = x_part[:, idx['part_deta']]
    dphi = x_part[:, idx['part_dphi']]
    deltaR = np.hypot(deta, dphi)
    d0 = np.tanh(x_part[:, idx['part_d0val']])
    dz = np.tanh(x_part[:, idx['part_dzval']])

    derived = {
        'part_pt_log': pt_log, 'part_e_log': e_log,
        'part_logptrel': logptrel, 'part_logerel': logerel,
        'part_deltaR': deltaR,
        'part_charge': x_part[:, idx['part_charge']],
        'part_isChargedHadron': x_part[:, idx['part_isChargedHadron']],
        'part_isNeutralHadron': x_part[:, idx['part_isNeutralHadron']],
        'part_isPhoton': x_part[:, idx['part_isPhoton']],
        'part_isElectron': x_part[:, idx['part_isElectron']],
        'part_isMuon': x_part[:, idx['part_isMuon']],
        'part_d0': d0,
        'part_d0err': x_part[:, idx['part_d0err']],
        'part_dz': dz,
        'part_dzerr': x_part[:, idx['part_dzerr']],
        'part_deta': deta,
        'part_dphi': dphi,
    }

    feats = []
    for name, sub, mul, lo, hi in PF_FEATURES:
        f = (derived[name] - sub) * mul
        f = np.clip(f, lo, hi)
        feats.append(f.astype('float32'))
    x_features = np.stack(feats, axis=1)  # (N, C_feat, P)
    x_vectors = np.stack([px, py, pz, energy], axis=1).astype('float32')  # (N, 4, P)
    mask = (pt > 0).astype('float32')[:, None, :]  # (N, 1, P)
    return x_features, x_vectors, mask


class JetClassIterableDataset(IterableDataset):
    """Iterable dataset that streams jets out of a list of ROOT shards.

    File-level sharding across (num_workers × world_size) ensures each rank /
    worker sees a disjoint set of files. Within a shard we shuffle the rows
    before yielding.
    """

    def __init__(self,
                 file_glob,
                 max_num_particles=128,
                 shuffle_files=True,
                 shuffle_within_file=True,
                 seed=42,
                 rank=0,
                 world_size=1,
                 shuffle_buffer_size=20000,
                 num_concurrent_files=10,
                 rows_per_file_visit=10000,
                 shard_by_rows=False):
        """
        shuffle_buffer_size: rows held in the streaming shuffle buffer per
            worker. 0 disables shuffling entirely (eval mode).
        num_concurrent_files: open this many ROOT files at once and round-robin
            random-draw rows from them. Critical for JetClass because each
            file is class-pure (HToBB_*.root etc.); EVERY batch comes from a
            single DataLoader worker (PyTorch IterableDataset behaviour), so
            K must be ≥ num_classes if you want every batch to contain all
            classes. Default 10 matches JetClass's 10 classes.
        rows_per_file_visit: when a file is opened, only this many random rows
            are extracted and held in the slot; the parent ndarray is freed
            immediately. Required at K=10 to keep memory reasonable: with
            K=10 and full files, each worker holds K × 1.1 GB ≈ 11 GB and
            16 workers (4 ranks × 4 DataLoader workers) sum to ~176 GB. With
            rows_per_file_visit=10000 the slot holds 10k × 11 KB ≈ 110 MB →
            ~18 GB total. Set to None for full coverage per visit (heavy).
        """
        super().__init__()
        files = sorted(glob.glob(file_glob))
        if not files:
            raise FileNotFoundError(f'No ROOT files matched: {file_glob}')
        self.files = files
        self.max_num_particles = max_num_particles
        self.shuffle_files = shuffle_files
        self.shuffle_within_file = shuffle_within_file
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_concurrent_files = max(1, num_concurrent_files)
        self.rows_per_file_visit = rows_per_file_visit
        self.shard_by_rows = shard_by_rows

    def _shard_files(self, rank, world, worker_id, num_workers, epoch):
        """Return (files_for_this_shard, row_stride, row_offset).

        Two regimes:
          - len(files) >= total_shards: file-level sharding, every shard reads
            a disjoint set of files; row_stride=1 row_offset=0.
          - len(files) <  total_shards: every shard reads every file but takes
            rows at [offset::stride] so total_shards parallel streams cover
            all rows exactly once each pass.
        """
        rng = random.Random(self.seed + epoch)
        files = list(self.files)
        if self.shuffle_files:
            rng.shuffle(files)
        total_shards = world * num_workers
        shard_idx = rank * num_workers + worker_id

        # shard_by_rows: every shard reads every file but takes row stride —
        # required when files are class-pure and len(files)/total_shards <
        # num_classes (file-level sharding would otherwise drop classes).
        if self.shard_by_rows:
            return files, total_shards, shard_idx
        if len(files) >= total_shards:
            return files[shard_idx::total_shards], 1, 0
        # Few-file regime — replicate file list, slice rows instead.
        return files, total_shards, shard_idx

    def _load_block(self, fp, row_offset, row_stride, epoch, max_rows=None):
        """Load a ROOT file, apply row striding/shuffle, optionally subsample
        to `max_rows`, copy the slice out, free the source arrays.

        Returns (xf, xv, m, labels, order) where order indexes into xf/xv/m/
        labels (which are already the subsample). Returns None on read failure.
        """
        try:
            x_part, x_jet, y = _read_root(
                fp,
                max_num_particles=self.max_num_particles,
                particle_features=_BASE_PARTICLE_VARS,
                jet_features=_JET_FEATURES,
                labels=LABELS,
            )
        except Exception as e:
            print(f'[JetClass] skipping {fp}: {e}', flush=True)
            return None
        xf, xv, m = _derive_features(x_part, x_jet)
        labels_idx = y.argmax(axis=1).astype(np.int64)
        N = xf.shape[0]
        order = np.arange(N)
        if self.shuffle_within_file:
            rng = np.random.default_rng(self.seed + epoch * 7919 + (hash(fp) & 0x7fffffff))
            rng.shuffle(order)
        order = order[row_offset::row_stride]
        if max_rows is not None and len(order) > max_rows:
            order = order[:max_rows]
        # Extract only what we'll use and free the parent ndarrays.
        xf_sub = xf[order].copy()
        xv_sub = xv[order].copy()
        m_sub = m[order].copy()
        labels_sub = labels_idx[order].copy()
        del xf, xv, m, labels_idx, x_part, x_jet, y
        return xf_sub, xv_sub, m_sub, labels_sub, np.arange(len(order))

    @staticmethod
    def _make_row_tensors(xf_row, xv_row, m_row, label):
        return (
            torch.from_numpy(xf_row),
            torch.from_numpy(xv_row),
            torch.from_numpy(m_row),
            torch.tensor(label, dtype=torch.long),
        )

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        epoch = 0
        while True:
            files, row_stride, row_offset = self._shard_files(
                self.rank, self.world_size, worker_id, num_workers, epoch)
            if not files:  # never spin forever
                return

            do_shuffle = self.shuffle_files or self.shuffle_within_file
            buf_size = self.shuffle_buffer_size if do_shuffle else 0

            # ----- Eval / no-shuffle mode: yield rows in file order ----------
            if buf_size == 0:
                for fp in files:
                    # Eval: read full file (no subsample) for deterministic
                    # coverage of every test/val row.
                    blk = self._load_block(fp, row_offset, row_stride, epoch,
                                            max_rows=None)
                    if blk is None:
                        continue
                    xf, xv, m, labels_idx, order = blk
                    for i in order:
                        yield self._make_row_tensors(xf[i], xv[i], m[i], int(labels_idx[i]))
                if not self.shuffle_files and not self.shuffle_within_file:
                    return  # single pass; eval mode
                epoch += 1
                continue

            # ----- Train mode: stratified K-slot concurrent reader -----------
            # Pick K files at a time, BUT assign one class per slot so the
            # K-way mix is guaranteed to cover K distinct classes. Without
            # stratification, sampling K files at random from a balanced pool
            # of C classes gives only C*(1-(1-1/C)^K) unique classes (e.g.
            # 6.5 / 10 with K=10), which is why a batch ends up with 6–7
            # bars, not 10.
            rng = random.Random(self.seed + worker_id * 991 + epoch * 7919)
            files = list(files)
            rng.shuffle(files)

            # Group files by class prefix (everything before the first '_'
            # in the basename). JetClass files are HToBB_000.root etc.
            class_files = {}
            for fp in files:
                base = os.path.basename(fp)
                cls = base.split('_', 1)[0] if '_' in base else base
                class_files.setdefault(cls, []).append(fp)
            for cls in class_files:
                rng.shuffle(class_files[cls])
            classes = list(class_files.keys())
            rng.shuffle(classes)
            class_iters = {cls: iter(class_files[cls]) for cls in classes}

            def _new_slot(target_cls):
                """Load the next file from `target_cls`; fall back to any
                other class with remaining files if exhausted. Returns
                (slot_dict, class_used) or (None, None) when nothing left."""
                # Try the target class first, then any other class with files.
                lanes = [target_cls] + [c for c in classes if c != target_cls]
                for cls in lanes:
                    while True:
                        fp = next(class_iters[cls], None)
                        if fp is None:
                            break
                        blk = self._load_block(fp, row_offset, row_stride, epoch,
                                               max_rows=self.rows_per_file_visit)
                        if blk is None:
                            continue
                        xf, xv, m, labels_idx, order = blk
                        return ({
                            'xf': xf, 'xv': xv, 'm': m, 'labels': labels_idx,
                            'order': order, 'cursor': 0, 'n': len(order),
                            'cls': cls,
                        }, cls)
                return None, None

            # Initial fill: one slot per class, round-robin (capped at K).
            K = min(self.num_concurrent_files, len(files))
            slots = []
            for k in range(K):
                target = classes[k % len(classes)]
                s, _ = _new_slot(target)
                if s is None:
                    break
                slots.append(s)

            buffer = []
            while slots:
                j = rng.randrange(len(slots))
                s = slots[j]
                i = int(s['order'][s['cursor']])
                s['cursor'] += 1
                item = (s['xf'][i].copy(), s['xv'][i].copy(),
                        s['m'][i].copy(), int(s['labels'][i]))
                if len(buffer) < buf_size:
                    buffer.append(item)
                else:
                    k = rng.randrange(buf_size)
                    out = buffer[k]
                    buffer[k] = item
                    yield self._make_row_tensors(*out)
                if s['cursor'] >= s['n']:
                    # Refill the slot — prefer the same class lane to keep
                    # K classes resident at all times.
                    repl, _ = _new_slot(s['cls'])
                    if repl is None:
                        slots.pop(j)
                    else:
                        slots[j] = repl
            # Drain remaining buffer in random order.
            rng.shuffle(buffer)
            for out in buffer:
                yield self._make_row_tensors(*out)
            epoch += 1


def build_dataloader(file_glob, batch_size, num_workers=4,
                     max_num_particles=128, shuffle=True,
                     rank=0, world_size=1, seed=42,
                     shuffle_buffer_size=20000,
                     num_concurrent_files=10,
                     rows_per_file_visit=10000,
                     shard_by_rows=False,
                     drop_last=True):
    ds = JetClassIterableDataset(
        file_glob,
        max_num_particles=max_num_particles,
        shuffle_files=shuffle,
        shuffle_within_file=shuffle,
        rank=rank,
        world_size=world_size,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size if shuffle else 0,
        num_concurrent_files=num_concurrent_files,
        rows_per_file_visit=rows_per_file_visit,
        shard_by_rows=shard_by_rows,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=drop_last,
    )


NUM_CLASSES = len(LABELS)
NUM_FEATURES = len(PF_FEATURES)
