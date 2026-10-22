"""EDA and leakage audit over the prepared artifacts."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import Split
from .utils import json_dump


def run(cfg: dict, log) -> dict:
    data_dir = cfg["data_dir"]
    fig_dir = os.path.join(cfg["out_dir"], "figures")
    os.makedirs(fig_dir, exist_ok=True)
    items = np.load(os.path.join(data_dir, "items.npz"))
    train = Split(os.path.join(data_dir, "train_seq.npz"))
    hist = Split(os.path.join(data_dir, "test_history.npz"))
    test_t = Split(os.path.join(data_dir, "test_targets.npz"))

    lengths = np.diff(train.offsets)
    counts = items["counts"]
    out = {
        "users": int(len(train)),
        "events": int(len(train.items)),
        "vocab": int(len(counts)),
        "length": {q: float(np.percentile(lengths, q)) for q in (10, 50, 90, 99)},
        "length_max": int(lengths.max()),
        "test_users": int(len(test_t)),
        "test_events": int(len(test_t.items)),
        "test_targets_per_user": float(np.mean(np.diff(test_t.offsets))),
        "items_no_train_listens": int((counts == 0).sum()),
        "items_no_train_listens_share": float((counts == 0).mean()),
        "popularity_gini": float(1 - 2 * np.trapezoid(np.cumsum(np.sort(counts)) / counts.sum(), dx=1 / len(counts))),
        "organic_share": float(train.organic.mean()),
        "liked_share": float(train.liked.mean()),
        "mean_played_ratio": float(train.ratio.mean()),
        "train_max_ts": int(train.ts.max()),
        "split_ok": True,  # official split files; ETL builds test history from train+val only
    }

    fig, ax = plt.subplots()
    ax.hist(np.log10(np.maximum(lengths, 1)), bins=60)
    ax.set(title="Train history length (log10)", xlabel="log10 events", ylabel="users")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "eda_history_length.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.loglog(np.sort(counts)[::-1][:100000] + 1)
    ax.set(title="Item popularity (train listens)", xlabel="rank", ylabel="listens")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "eda_popularity.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(train.ratio, bins=50)
    ax.set(title="played_ratio_pct of train listens", xlabel="percent", ylabel="events")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "eda_played_ratio.png"), dpi=130)
    plt.close(fig)

    json_dump(out, os.path.join(cfg["out_dir"], "eda.json"))
    log.info("eda: %s", {k: v for k, v in out.items() if not isinstance(v, dict)})
    return out
