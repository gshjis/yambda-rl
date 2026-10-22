"""Sequence containers and torch datasets over the ETL artifacts."""

from __future__ import annotations

import numpy as np
import torch


class Split:
    """Flattened variable-length sequences with offset indexing."""

    def __init__(self, path: str):
        z = np.load(path)
        self.uids = z["uids"]
        self.offsets = z["offsets"]
        self.items = z["items"]
        self.ts = z["ts"] if "ts" in z else None
        self.ratio = z["ratio"] if "ratio" in z else None
        self.organic = z["organic"] if "organic" in z else None
        self.liked = z["liked"] if "liked" in z else None
        self.dt_next = z["dt_next"] if "dt_next" in z else None
        self.active7 = z["active7"] if "active7" in z else None

    def __len__(self) -> int:
        return len(self.uids)

    def seq(self, i: int) -> np.ndarray:
        return self.items[self.offsets[i] : self.offsets[i + 1]]

    def index_of(self, uid: int) -> int:
        pos = np.searchsorted(self.uids, uid)
        if pos >= len(self.uids) or self.uids[pos] != uid:
            return -1
        return int(pos)


class WindowDataset(torch.utils.data.Dataset):
    """Random next-item windows over user histories."""

    def __init__(self, split: Split, ctx: int, samples: int, seed: int = 0):
        self.s = split
        self.ctx = ctx
        self.samples = samples
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, _):
        n_users = len(self.s)
        for _ in range(8):
            u = int(self.rng.integers(n_users))
            lo, hi = int(self.s.offsets[u]), int(self.s.offsets[u + 1])
            if hi - lo >= 2:
                break
        else:
            raise RuntimeError("dataset has no user with >= 2 events")
        end = int(self.rng.integers(lo + 1, hi))
        start = max(lo, end - self.ctx)
        x = self.s.items[start:end]
        y = int(self.s.items[end])
        return x.astype(np.int64), y


def collate_windows(batch, pad_id: int):
    xs, ys = zip(*batch)
    length = max(len(x) for x in xs)
    x = np.full((len(xs), length), pad_id, dtype=np.int64)
    lengths = np.zeros(len(xs), dtype=np.int64)
    for i, seq in enumerate(xs):
        x[i, : len(seq)] = seq
        lengths[i] = len(seq)
    return torch.from_numpy(x), torch.tensor(ys, dtype=torch.long), torch.from_numpy(lengths)
