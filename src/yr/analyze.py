"""Post-hoc analysis: relevance vs reward, popularity collapse, multi-step drift."""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import Split
from .evaluate import EvalSet, load_model
from .metrics import bootstrap_ci
from .reward import item_features, load_ensemble
from .rl import _pad, generate
from .utils import json_dump, load_config


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _behavior(ranked: np.ndarray, counts: np.ndarray, total_events: float, history_items: np.ndarray,
              lengths: np.ndarray, mean_length: np.ndarray) -> dict:
    top10 = ranked[:, :10]
    pop = counts[top10].astype(np.float64)
    novelty = -np.log2(np.maximum(pop, 1) / total_events)
    exposure = np.bincount(ranked[:, :10].ravel(), minlength=len(counts))
    p = np.sort(exposure / exposure.sum())
    gini = float(1 - 2 * np.trapezoid(np.cumsum(p), dx=1 / len(p)))
    pct = np.argsort(np.argsort(counts)) / max(1, len(counts) - 1)
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(ranked), size=(min(2000, len(ranked) * 2), 2))
    jacc = []
    for a, b in pairs:
        sa, sb = set(top10[a].tolist()), set(top10[b].tolist())
        jacc.append(len(sa & sb) / max(1, len(sa | sb)))
    repeats = []
    for i in range(len(ranked)):
        hist = set(history_items[i].tolist())
        repeats.append(np.isin(top10[i], list(hist)).mean() if hist else 0.0)
    return {
        "mean_popularity_percentile": float(pct[top10].mean()),
        "mean_log_popularity": float(np.log1p(pop).mean()),
        "mean_novelty": float(novelty.mean()),
        "exposure_gini": gini,
        "jaccard_top10": float(np.mean(jacc)),
        "repeat_from_history": float(np.mean(repeats)),
        "mean_track_length": float(np.nan_to_num(mean_length[top10], nan=0.0).mean()),
        "distinct_items_per_user": float(np.mean([len(set(r)) for r in top10])),
    }


def _score_topk(models, hist: Split, ev: EvalSet, ranked: np.ndarray, counts: np.ndarray, cfg: dict,
                device: str, k: int = 10) -> dict:
    rl_ctx = min(int(cfg["rm"]["ctx"]), int(cfg["data"]["max_seq_len"]))
    hists, cands, feats = [], [], []
    for i in range(len(ev)):
        seq = ev.hist_items(i)[-rl_ctx:]
        hists.extend([seq] * k)
        cands.extend(ranked[i, :k].tolist())
        feats.extend([[0.5, 0.5, min(len(seq), rl_ctx) / rl_ctx]] * k)
    item_feats = item_features(np.load(os.path.join(cfg["data_dir"], "items.npz")), "cpu")
    out = {"ratio": [], "like": []}
    for start in range(0, len(hists), 4096):
        part = slice(start, min(start + 4096, len(hists)))
        x, ln = _pad(hists[part], device)
        cand = torch.tensor(cands[part], dtype=torch.long, device=device)
        fs = torch.cat([torch.tensor(np.array(feats[part]), dtype=torch.float32, device=device),
                        item_feats[cand.cpu()].to(device)], dim=1)
        rs, ls = [], []
        for m in models:
            o = m(x, cand, fs, ln)
            rs.append(torch.sigmoid(o["ratio"]))
            ls.append(torch.sigmoid(o["like"]))
        out["ratio"].append(torch.stack(rs).mean(0).cpu().numpy())
        out["like"].append(torch.stack(ls).mean(0).cpu().numpy())
    res = {key: np.concatenate(v) for key, v in out.items()}
    return {f"rm_{key}_top{k}": float(v.mean()) for key, v in res.items()}


def _multistep(cfg, tag: str, ckpt: str, ev: EvalSet, models, items: dict, device: str, n: int = 200) -> dict:
    model = load_model(ckpt, device).eval()
    prompts = [ev.hist_items(i).tolist() for i in range(min(n, len(ev)))]
    seqs, actions = generate(model, prompts, 1, 5, cfg["data"]["max_seq_len"], 100, 1.0, device)
    counts = items["counts"]
    lengths = np.nan_to_num(items["mean_length"], nan=0.0)
    item_feats = item_features(items, device)
    steps = []
    for t in range(5):
        cands = [a[t] for a in actions]
        pops = counts[cands].astype(np.float64)
        rep = np.mean([a[t] in set(prompts[i]) or a[t] in set(actions[i][:t]) for i, a in enumerate(actions)])
        steps.append({
            "step": t,
            "mean_log_popularity": float(np.log1p(pops).mean()),
            "repeat_rate": float(rep),
            "mean_track_length": float(lengths[cands].mean()),
            "unique_items": int(len(set(cands))),
        })
    return {"tag": tag, "steps": steps, "actions": [a for a in actions[:500]]}


def run(cfg: dict, log) -> dict:
    out = cfg["out_dir"]
    data_dir = cfg["data_dir"]
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = np.load(os.path.join(data_dir, "items.npz"))
    counts = items["counts"].astype(np.float64)
    total_events = float(counts.sum())
    hist = Split(os.path.join(data_dir, "test_history.npz"))
    tgt = Split(os.path.join(data_dir, "test_targets.npz"))
    ev = EvalSet(hist, tgt, cfg["data"]["max_seq_len"])

    base = np.load(os.path.join(out, "base_ranked.npz")) if os.path.exists(os.path.join(out, "base_ranked.npz")) else None
    rl_files = sorted(glob.glob(os.path.join(out, "rl_*_ranked.npz")))
    rl = np.load(rl_files[0]) if rl_files else None

    result: dict = {"models": {}}
    for name, arr in (("base", base), ("rl", rl)):
        if arr is None:
            continue
        hist_items = [ev.hist_items(i) for i in range(len(ev))]
        result["models"][name] = _behavior(
            arr["ranked"], counts, total_events,
            [h for h in hist_items], np.array([len(h) for h in hist_items]),
            items["mean_length"],
        )

    if base is not None and rl is not None:
        n = min(len(base["uids"]), len(rl["uids"]))
        common = np.intersect1d(base["uids"][:n], rl["uids"][:n])
        bi = {int(u): i for i, u in enumerate(base["uids"])}
        ri = {int(u): i for i, u in enumerate(rl["uids"])}
        result["significance"] = {}
        for metric in ("ndcg@10", "recall@100", "map@10"):
            a = np.array([base[f"per_user_{metric}"][bi[int(u)]] for u in common])
            b = np.array([rl[f"per_user_{metric}"][ri[int(u)]] for u in common])
            result["significance"][metric] = bootstrap_ci(a, b, cfg["analysis"]["bootstrap"], cfg["seed"])

    try:
        models = load_ensemble(cfg, device)
        for name, arr in (("base", base), ("rl", rl)):
            if arr is not None:
                result["models"].setdefault(name, {}).update(
                    _score_topk(models, hist, ev, arr["ranked"], counts, cfg, device))
    except FileNotFoundError:
        log.warning("no reward-model checkpoints, skipping RM scoring")
        models = None

    ms = {}
    if base is not None and models is not None:
        base_ckpt = os.path.join(out, "ckpt", "base_best.pt")
        if not os.path.exists(base_ckpt):
            base_ckpt = os.path.join(out, "ckpt", "base.pt")
        if os.path.exists(base_ckpt):
            ms["base"] = _multistep(cfg, "base", base_ckpt, ev, models, items, device)
        rl_ckpt = sorted(glob.glob(os.path.join(out, "rl_*.pt")))
        if rl_ckpt:
            ms["rl"] = _multistep(cfg, "rl", rl_ckpt[0], ev, models, items, device)
    result["multistep"] = ms

    runs = []
    for path in sorted(glob.glob(os.path.join(out, "metrics_*.json"))):
        d = _load_json(path)
        if d and "summary" in d and "n_users" in d:
            runs.append({"name": os.path.basename(path), "ndcg@10": d["summary"].get("ndcg@10"),
                         "recall@100": d["summary"].get("recall@100")})
    result["runs"] = runs
    json_dump(result, os.path.join(out, "analysis.json"))

    if base is not None and rl is not None and "significance" in result:
        _figures(result, base, rl, cfg, fig_dir)
    log.info("analysis: %s", json.dumps(result["models"], ensure_ascii=False))
    return result


def _figures(result, base, rl, cfg, fig_dir):
    fig, ax = plt.subplots()
    for name, arr in (("base", base), ("rl", rl)):
        ax.plot(arr["per_user_ndcg@10"], label=name, alpha=0.7)
    ax.set(title="Per-user NDCG@10", xlabel="user", ylabel="ndcg@10")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_user_ndcg.png"), dpi=130)
    plt.close(fig)

    names = [n for n in result["models"] if result["models"][n]]
    keys = ["mean_novelty", "jaccard_top10", "repeat_from_history", "mean_log_popularity"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.5))
    for ax, key in zip(np.atleast_1d(axes), keys):
        ax.bar(names, [result["models"][n].get(key, 0) for n in names])
        ax.set(title=key)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "behavior.png"), dpi=130)
    plt.close(fig)

    if result["multistep"]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for name, d in result["multistep"].items():
            steps = [s["step"] for s in d["steps"]]
            for ax, key in zip(axes, ("mean_log_popularity", "repeat_rate", "mean_track_length")):
                ax.plot(steps, [s[key] for s in d["steps"]], marker="o", label=name)
                ax.set(title=key, xlabel="generation step")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "multistep.png"), dpi=130)
        plt.close(fig)
