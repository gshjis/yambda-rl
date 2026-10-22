"""Ranking metrics computed exactly like the Yambda benchmark protocol."""

from __future__ import annotations

import numpy as np


def _dcg(rel: np.ndarray) -> float:
    if rel.size == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float((rel * disc).sum())


def per_user_metrics(ranked: np.ndarray, targets: list[np.ndarray], catalog_size: int,
                     ks: tuple[int, ...] = (10, 50, 100)) -> dict:
    """ranked: (n_users, K) item indices, best first. targets: list of relevant item indices."""
    n = len(targets)
    out = {f"{m}@{k}": np.zeros(n, dtype=np.float64) for m in ("ndcg", "recall", "map") for k in ks}
    max_k = max(ks)
    for u in range(n):
        tgt = np.unique(targets[u])
        if tgt.size == 0:
            continue
        top = ranked[u, :max_k]
        for k in ks:
            hits = np.isin(top[:k], tgt)
            ideal = min(len(tgt), k)
            out[f"ndcg@{k}"][u] = _dcg(hits) / _dcg(np.ones(ideal))
            out[f"recall@{k}"][u] = hits.sum() / len(tgt)
        hits10 = np.isin(top[:10], tgt)
        if hits10.any():
            pos = np.flatnonzero(hits10)
            out["map@10"][u] = float((np.arange(1, len(pos) + 1) / (pos + 1)).sum() / min(len(tgt), 10))
    summary = {m: float(v.mean()) for m, v in out.items()}
    for k in ks:
        summary[f"coverage@{k}"] = float(np.unique(ranked[:, :k]).size / catalog_size)
    return {"summary": summary, "per_user": out}


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap over users for mean(a - b)."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "delta": float(d.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
    }
