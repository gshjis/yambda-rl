"""Popularity baselines on the official test split."""

from __future__ import annotations

import os

import numpy as np

from .data import Split
from .evaluate import EvalSet
from .metrics import per_user_metrics
from .utils import json_dump


def run(cfg: dict, log) -> dict:
    items = np.load(os.path.join(cfg["data_dir"], "items.npz"))
    counts = items["counts"].astype(np.float64)
    hist = Split(os.path.join(cfg["data_dir"], "test_history.npz"))
    tgt = Split(os.path.join(cfg["data_dir"], "test_targets.npz"))
    ev = EvalSet(hist, tgt, cfg["data"]["max_seq_len"])
    top = np.argsort(-counts)[: cfg["eval"]["topk"]]
    ranked = np.tile(top, (len(ev), 1))
    res = per_user_metrics(ranked, ev.targets, len(counts))
    res["n_users"] = len(ev)
    res["method"] = "popularity"
    json_dump(res, os.path.join(cfg["out_dir"], "metrics_popularity.json"))
    log.info("popularity: %s", {k: round(v, 4) for k, v in res["summary"].items()})
    return res
