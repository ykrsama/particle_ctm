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
                 shuffle_buffer_size=20000):
        """
        shuffle_buffer_size: when shuffle is on, rows from many files are mixed
            in a buffer of this many rows before being yielded. Required for
            JetClass because each ROOT file is class-pure (HToBB_*.root, etc.);
            without inter-file mixing a single DataLoader batch contains only
            one class. 0 disables the buffer.
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

        if len(files) >= total_shards:
            return files[shard_idx::total_shards], 1, 0
        # Few-file regime — replicate file list, slice rows instead.
        return files, total_shards, shard_idx

    def _load_block(self, fp, row_offset, row_stride, epoch):
        """Load a ROOT file and return per-row arrays after row striding +
        optional within-file shuffle. Returns None on read failure.
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
        return xf, xv, m, labels_idx, order

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
                    blk = self._load_block(fp, row_offset, row_stride, epoch)
                    if blk is None:
                        continue
                    xf, xv, m, labels_idx, order = blk
                    for i in order:
                        yield self._make_row_tensors(xf[i], xv[i], m[i], int(labels_idx[i]))
                if not self.shuffle_files and not self.shuffle_within_file:
                    return  # single pass; eval mode
                epoch += 1
                continue

            # ----- Train mode: streaming shuffle buffer ----------------------
            # Buffer holds tuples (xf_row.copy(), xv_row.copy(), m_row.copy(),
            # label). We copy rows because the parent ndarrays for previous
            # files would otherwise stay alive as long as any slice references
            # them, blowing memory once we move on.
            rng = random.Random(self.seed + worker_id * 991 + epoch * 7919)
            buffer = []
            for fp in files:
                blk = self._load_block(fp, row_offset, row_stride, epoch)
                if blk is None:
                    continue
                xf, xv, m, labels_idx, order = blk
                for i in order:
                    item = (xf[i].copy(), xv[i].copy(), m[i].copy(), int(labels_idx[i]))
                    if len(buffer) < buf_size:
                        buffer.append(item)
                    else:
                        j = rng.randrange(buf_size)
                        out = buffer[j]
                        buffer[j] = item
                        yield self._make_row_tensors(*out)
            # Drain remaining buffer in random order.
            rng.shuffle(buffer)
            for out in buffer:
                yield self._make_row_tensors(*out)
            epoch += 1


def build_dataloader(file_glob, batch_size, num_workers=4,
                     max_num_particles=128, shuffle=True,
                     rank=0, world_size=1, seed=42,
                     shuffle_buffer_size=20000):
    ds = JetClassIterableDataset(
        file_glob,
        max_num_particles=max_num_particles,
        shuffle_files=shuffle,
        shuffle_within_file=shuffle,
        rank=rank,
        world_size=world_size,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size if shuffle else 0,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )


NUM_CLASSES = len(LABELS)
NUM_FEATURES = len(PF_FEATURES)
