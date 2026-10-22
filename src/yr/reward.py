"""Reward model: predicts played ratio, like, session continuation and return.

Trained on logged train-period events only (`history < t`). The ensemble's
disagreement is used as a pessimistic penalty against reward overoptimisation
(Gao et al., 2022, https://arxiv.org/abs/2210.10760).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Split
from .model import Block
from .utils import json_dump

HEADS = ("ratio", "like", "session", "return")


def item_features(items: dict, device: str) -> torch.Tensor | None:
    counts = items["counts"].astype(np.float64)
    length = np.nan_to_num(items["mean_length"], nan=float(np.nanmedian(items["mean_length"])))
    feats = np.stack([
        np.log1p(counts) / np.log1p(counts.max()),
        np.clip(length, 0, 600) / 600.0,
        items["like_rate"].astype(np.float64),
    ], axis=1).astype(np.float32)
    return torch.from_numpy(feats).to(device)


class RewardModel(nn.Module):
    def __init__(self, vocab: int, d: int = 128, n_layer: int = 2, n_head: int = 4,
                 ctx: int = 100, n_feat: int = 6, dropout: float = 0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx + 1, d)
        self.blocks = nn.ModuleList([Block(d, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.feat = nn.Linear(n_feat, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, len(HEADS)))
        self.config = dict(vocab=vocab, d=d, n_layer=n_layer, n_head=n_head, ctx=ctx,
                           n_feat=n_feat, dropout=dropout)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x: torch.Tensor, cand: torch.Tensor, feats: torch.Tensor,
                lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t = x.shape
        ids = torch.cat([x, cand[:, None]], dim=1)
        pos = torch.arange(t + 1, device=x.device)
        h = self.tok(ids) + self.pos(pos)
        # history pads sit at [lengths, t); the candidate at index t must stay visible
        idx = torch.arange(t + 1, device=x.device)[None, :]
        mask = (idx >= lengths[:, None]) & (idx < t)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=mask)
        last = h[:, t]
        z = self.ln(last) + self.feat(feats)
        out = self.head(z)
        return {name: out[:, i] for i, name in enumerate(HEADS)}


class RMDataset(torch.utils.data.Dataset):
    """Samples (history, candidate) pairs; half positives, half random candidates."""

    def __init__(self, split: Split, counts: np.ndarray, ctx: int, n: int, seed: int):
        self.s = split
        self.counts = counts
        self.ctx = ctx
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.lengths = np.diff(split.offsets)
        self.valid_users = np.flatnonzero(self.lengths >= 2)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _):
        u = int(self.valid_users[self.rng.integers(len(self.valid_users))])
        lo, hi = int(self.s.offsets[u]), int(self.s.offsets[u + 1])
        pos = int(self.rng.integers(lo + 1, hi))
        hist = self.s.items[max(lo, pos - self.ctx): pos]
        cand = int(self.s.items[pos])
        ratio = float(self.s.ratio[pos]) / 100.0
        like = float(self.s.liked[pos])
        session = float(0 < self.s.dt_next[pos] <= 1800)
        ret = float(self.s.active7[pos])
        hist_ratio = float(self.s.ratio[max(lo, pos - self.ctx): pos].mean()) / 100.0
        hist_organic = float(self.s.organic[max(lo, pos - self.ctx): pos].mean())
        is_neg = self.rng.random() < 0.5
        if is_neg:
            cand = int(self.rng.integers(len(self.counts)))
            while self.counts[cand] == 0:
                cand = int(self.rng.integers(len(self.counts)))
            like, session, ret = 0.0, 0.0, 0.0
        return {
            "hist": hist.astype(np.int64),
            "cand": cand,
            "ratio": ratio,
            "like": like,
            "session": session,
            "return": ret,
            "mask_ratio": 0.0 if is_neg else 1.0,
            "feat": np.array([hist_ratio, hist_organic, min(len(hist), self.ctx) / self.ctx], dtype=np.float32),
        }


def collate(batch, n_feat_item: int, item_feats: torch.Tensor | None):
    bs = len(batch)
    t = max(len(b["hist"]) for b in batch)
    x = torch.zeros((bs, t), dtype=torch.long)
    lengths = torch.zeros(bs, dtype=torch.long)
    for i, b in enumerate(batch):
        x[i, : len(b["hist"])] = torch.from_numpy(b["hist"])
        lengths[i] = len(b["hist"])
    cand = torch.tensor([b["cand"] for b in batch], dtype=torch.long)
    feats = torch.from_numpy(np.stack([b["feat"] for b in batch]))
    if item_feats is not None:
        feats = torch.cat([feats, item_feats[cand].cpu()], dim=1)
    labels = {h: torch.tensor([b[h] for b in batch], dtype=torch.float32) for h in HEADS}
    mask_ratio = torch.tensor([b["mask_ratio"] for b in batch], dtype=torch.float32)
    return x, cand, feats, lengths, labels, mask_ratio


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    y = y[order]
    if y.sum() == 0:
        return float("nan")
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


@torch.no_grad()
def predict(models: list[RewardModel], x, cand, feats, lengths, device: str) -> dict[str, torch.Tensor]:
    out = {h: [] for h in HEADS}
    for m in models:
        o = m(x.to(device), cand.to(device), feats.to(device), lengths.to(device))
        for h in HEADS:
            out[h].append(o[h].float())
    return {h: torch.stack(v) for h, v in out.items()}


def evaluate(models: list[RewardModel], split: Split, items: dict, cfg: dict, device: str,
             n: int = 100_000, seed: int = 1) -> dict:
    rcfg = cfg["rm"]
    n = min(n, int(np.diff(split.offsets).sum()))
    rng = np.random.default_rng(seed)
    lengths = np.diff(split.offsets)
    users = np.flatnonzero(lengths >= 2)
    u = rng.choice(users, size=n)
    pos = np.floor(rng.random(n) * (lengths[u] - 1)).astype(np.int64) + split.offsets[u] + 1
    hist, cand, ratio, like, session, ret, organic, hratio = [], [], [], [], [], [], [], []
    for i in range(n):
        lo = int(split.offsets[u[i]])
        p = int(pos[i])
        hist.append(split.items[max(lo, p - rcfg["ctx"]): p])
        cand.append(int(split.items[p]))
        ratio.append(split.ratio[p] / 100.0)
        like.append(float(split.liked[p]))
        session.append(float(0 < split.dt_next[p] <= 1800))
        ret.append(float(split.active7[p]))
        organic.append(float(split.organic[max(lo, p - rcfg["ctx"]): p].mean()))
        hratio.append(float(split.ratio[max(lo, p - rcfg["ctx"]): p].mean()) / 100.0)
    item_feats = item_features(items, "cpu")
    neg = rng.integers(0, len(items["counts"]), size=n)
    hist2, cand2, like2 = [], [], []
    for i in range(n):
        hist2.append(hist[i])
        cand2.append(int(neg[i]))
        like2.append(0.0)
    all_hist = hist + hist2
    all_cand = cand + cand2
    all_like = np.array(like + like2)
    all_ratio = np.array(ratio + [0.0] * n)
    all_session = np.array(session + [0.0] * n)
    all_return = np.array(ret + [0.0] * n)
    mask_ratio = np.array([1.0] * n + [0.0] * n)
    all_feat = np.array([[hratio[i], organic[i], min(len(hist[i]), rcfg["ctx"]) / rcfg["ctx"]]
                         for i in range(n)] + [[0.0, 0.0, 0.0] for _ in range(n)], dtype=np.float32)
    preds = []
    for start in range(0, 2 * n, 2048):
        b = slice(start, min(start + 2048, 2 * n))
        t = max(len(h) for h in all_hist[b])
        x = torch.zeros((b.stop - b.start, t), dtype=torch.long)
        ln = torch.zeros(b.stop - b.start, dtype=torch.long)
        for j, h in enumerate(all_hist[b]):
            x[j, : len(h)] = torch.from_numpy(h.astype(np.int64))
            ln[j] = len(h)
        feats = torch.from_numpy(all_feat[b])
        feats = torch.cat([feats, item_feats[torch.tensor(all_cand[b])]], dim=1)
        o = predict(models, x, torch.tensor(all_cand[b]), feats, ln, device)
        preds.append({h: o[h].mean(0).cpu().numpy() for h in HEADS})
    pr = {h: np.concatenate([p[h] for p in preds]) for h in HEADS}
    r = pr["ratio"]
    m = mask_ratio == 1
    from scipy.stats import spearmanr

    res = {
        "n": int(2 * n),
        "ratio_mae": float(np.abs(r[m] - all_ratio[m]).mean()),
        "ratio_rmse": float(np.sqrt(((r[m] - all_ratio[m]) ** 2).mean())),
        "ratio_spearman": float(spearmanr(r[m], all_ratio[m]).statistic),
        "like_auc": _auc(all_like, pr["like"]),
        "like_ap": _ap(all_like, pr["like"]),
        "like_ece": _ece(all_like, 1 / (1 + np.exp(-pr["like"]))),
        "session_auc": _auc(all_session, pr["session"]),
        "return_auc": _auc(all_return, pr["return"]),
    }
    return res


def load_ensemble(cfg: dict, device: str) -> list[RewardModel]:
    import glob

    paths = sorted(glob.glob(os.path.join(cfg["out_dir"], "rm_*.pt")))
    if not paths:
        raise FileNotFoundError("no reward-model checkpoints; run the rm stage first")
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = RewardModel(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        models.append(m.to(device).eval())
    return models


def run(cfg: dict, log) -> dict:
    rcfg = cfg["rm"]
    data_dir = cfg["data_dir"]
    items = np.load(os.path.join(data_dir, "items.npz"))
    train = Split(os.path.join(data_dir, "train_seq.npz"))
    val = Split(os.path.join(data_dir, "val_seq.npz"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    item_feats = item_features(items, "cpu")
    n_models = int(rcfg["ensemble"])
    paths, metrics = [], []
    for k in range(n_models):
        torch.manual_seed(cfg["seed"] + k)
        model = RewardModel(vocab=len(items["counts"]), d=rcfg["d_model"], n_layer=rcfg["n_layer"],
                            n_head=rcfg["n_head"], ctx=rcfg["ctx"], n_feat=6).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=rcfg["lr"], weight_decay=0.01)
        ds = RMDataset(train, items["counts"], rcfg["ctx"], int(rcfg["max_events"]), cfg["seed"] + k)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=rcfg["batch_size"], shuffle=False, num_workers=0,
            collate_fn=lambda b: collate(b, 3, item_feats),
        )
        for epoch in range(int(rcfg["epochs"])):
            for step, (x, cand, feats, ln, labels, mask_ratio) in enumerate(loader):
                x, cand, feats, ln = x.to(device), cand.to(device), feats.to(device), ln.to(device)
                out = model(x, cand, feats, ln)
                loss = 0.0
                w = {"ratio": 1.0, "like": 1.0, "session": 0.5, "return": 0.5}
                loss = loss + w["ratio"] * (F.mse_loss(torch.sigmoid(out["ratio"]), labels["ratio"].to(device),
                                                       reduction="none")
                                            * mask_ratio.to(device)).sum() / mask_ratio.sum().clamp(min=1)
                for h in ("like", "session", "return"):
                    loss = loss + w[h] * F.binary_cross_entropy_with_logits(out[h], labels[h].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step % 200 == 0:
                    log.info("rm %d epoch %d step %d loss %.4f", k, epoch, step, loss.item())
        path = os.path.join(cfg["out_dir"], f"rm_{k}.pt")
        torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
        paths.append(path)
        m = evaluate([model], val, items, cfg, device, n=30_000, seed=cfg["seed"] + k)
        m["model"] = k
        metrics.append(m)
        log.info("rm %d val: %s", k, {key: round(v, 4) for key, v in m.items() if isinstance(v, float)})
    ens = [RewardModel(**torch.load(p, map_location="cpu", weights_only=False)["config"]).to(device) for p in paths]
    for m, p in zip(ens, paths):
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["state_dict"])
        m.eval()
    ens_metrics = evaluate(ens, val, items, cfg, device, n=50_000, seed=cfg["seed"] + 100)
    json_dump({"models": metrics, "ensemble": ens_metrics, "paths": paths},
              os.path.join(cfg["out_dir"], "rm_metrics.json"))
    log.info("rm ensemble: %s", {key: round(v, 4) for key, v in ens_metrics.items() if isinstance(v, float)})
    return ens_metrics
