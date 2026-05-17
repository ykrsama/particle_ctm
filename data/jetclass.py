"""JetClass dataset loader.

Reads ROOT shards via `particle_transformer/dataloader.py::read_file` and
applies the standardisation specified in `JetClass_full.yaml`. Builds three
tensors per jet:

    x_features : (C_feat, P)  17 standardised particle features
    x_vectors  : (4, P)        (px, py, pz, energy) — fed to PairEmbed
    x_mask     : (1, P)        1 = real particle, 0 = padding

We avoid weaver-core entirely so its numpy<2 pin doesn't conflict with the
CTM stack.
"""

import glob
import math
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import IterableDataset

# Import particle_transformer/dataloader.read_file without touching weaver.
_PT_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'particle_transformer'))
if _PT_REPO not in sys.path:
    sys.path.insert(0, _PT_REPO)
from dataloader import read_file as _read_root  # noqa: E402


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
                 world_size=1):
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

    def _shard_files(self, rank, world, worker_id, num_workers, epoch):
        rng = random.Random(self.seed + epoch)
        files = list(self.files)
        if self.shuffle_files:
            rng.shuffle(files)
        total_shards = world * num_workers
        shard_idx = rank * num_workers + worker_id
        return files[shard_idx::total_shards]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        # epoch-ish counter: workers reshuffle each pass.
        epoch = 0
        while True:
            files = self._shard_files(self.rank, self.world_size, worker_id, num_workers, epoch)
            for fp in files:
                try:
                    x_part, x_jet, y = _read_root(
                        fp,
                        max_num_particles=self.max_num_particles,
                        particle_features=_BASE_PARTICLE_VARS,
                        jet_features=_JET_FEATURES,
                        labels=LABELS,
                    )
                except Exception as e:  # corrupt shard: skip rather than crash the worker
                    print(f'[JetClass] skipping {fp}: {e}', flush=True)
                    continue

                x_features, x_vectors, mask = _derive_features(x_part, x_jet)
                N = x_features.shape[0]
                order = np.arange(N)
                if self.shuffle_within_file:
                    rng = np.random.default_rng(self.seed + epoch * 7919 + hash(fp) % (2**31))
                    rng.shuffle(order)

                for i in order:
                    yield (
                        torch.from_numpy(x_features[i]),
                        torch.from_numpy(x_vectors[i]),
                        torch.from_numpy(mask[i]),
                        torch.tensor(int(y[i].argmax()), dtype=torch.long),
                    )
            epoch += 1
            if not self.shuffle_files and not self.shuffle_within_file:
                break  # eval / single-pass mode


def build_dataloader(file_glob, batch_size, num_workers=4,
                     max_num_particles=128, shuffle=True,
                     rank=0, world_size=1, seed=42):
    ds = JetClassIterableDataset(
        file_glob,
        max_num_particles=max_num_particles,
        shuffle_files=shuffle,
        shuffle_within_file=shuffle,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )


NUM_CLASSES = len(LABELS)
NUM_FEATURES = len(PF_FEATURES)
