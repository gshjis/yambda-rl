"""Loading checkpoints and full-catalog ranking evaluation."""

from __future__ import annotations

import numpy as np
import torch

from .data import Split
from .metrics import per_user_metrics
from .model import GPT, rank_all


class EvalSet:
    def __init__(self, history: Split, targets: Split, ctx: int):
        self.ctx = ctx
        tuids = targets.uids
        hidx = np.searchsorted(history.uids, tuids)
        valid = (hidx < len(history.uids)) & (history.uids[np.minimum(hidx, len(history.uids) - 1)] == tuids)
        self.uids = tuids[valid]
        self.hidx = hidx[valid]
        self.targets = [targets.items[targets.offsets[i] : targets.offsets[i + 1]] for i in range(len(targets)) if valid[i]]
        self.history = history

    def __len__(self) -> int:
        return len(self.uids)

    def hist_items(self, i: int) -> np.ndarray:
        j = int(self.hidx[i])
        seq = self.history.items[self.history.offsets[j] : self.history.offsets[j + 1]]
        return seq[-self.ctx :]

    def hist_lengths(self) -> np.ndarray:
        lo = self.history.offsets[self.hidx]
        hi = self.history.offsets[self.hidx + 1]
        return np.minimum(hi - lo, self.ctx)


def load_model(path: str, device: str) -> GPT:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def rank_users(model: GPT, hists: np.ndarray, lengths: np.ndarray, topk: int,
               batch_users: int, score_chunk: int, device: str) -> np.ndarray:
    out = np.zeros((len(hists), topk), dtype=np.int64)
    for start in range(0, len(hists), batch_users):
        end = min(start + batch_users, len(hists))
        n = end - start
        t = int(lengths[start:end].max())
        x = torch.zeros((n, t), dtype=torch.long, device=device)
        for i in range(n):
            seq = hists[start + i]
            x[i, : len(seq)] = torch.from_numpy(seq.astype(np.int64)).to(device)
        h = model(x)[torch.arange(n, device=device), torch.tensor(lengths[start:end] - 1, device=device)]
        out[start:end] = rank_all(h, model.tok.weight, topk=topk, chunk=score_chunk).cpu().numpy()
    return out


def evaluate(model: GPT, ev: EvalSet, cfg: dict, device: str, max_users: int | None = None) -> dict:
    n = len(ev)
    if max_users is not None and n > max_users:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(n, size=max_users, replace=False))
    else:
        keep = np.arange(n)
    hists = [ev.hist_items(int(i)) for i in keep]
    lengths = np.array([len(h) for h in hists], dtype=np.int64)
    ranked = rank_users(
        model, hists, lengths, cfg["eval"]["topk"], cfg["eval"]["batch_users"],
        cfg["eval"]["score_chunk"], device,
    )
    targets = [ev.targets[int(i)] for i in keep]
    res = per_user_metrics(ranked, targets, model.vocab)
    res["n_users"] = int(len(keep))
    return res


def run_test(cfg: dict, log, ckpt: str | None = None, tag: str = "base") -> dict:
    import os

    from .data import Split
    from .utils import json_dump

    out = cfg["out_dir"]
    if ckpt is None:
        ckpt = os.path.join(out, "ckpt", "base_best.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(out, "ckpt", "base.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)
    hist = Split(os.path.join(cfg["data_dir"], "test_history.npz"))
    tgt = Split(os.path.join(cfg["data_dir"], "test_targets.npz"))
    ev = EvalSet(hist, tgt, cfg["data"]["max_seq_len"])
    hists = [ev.hist_items(i) for i in range(len(ev))]
    lengths = np.array([len(h) for h in hists], dtype=np.int64)
    ranked = rank_users(model, hists, lengths, cfg["eval"]["topk"], cfg["eval"]["batch_users"],
                        cfg["eval"]["score_chunk"], device)
    res = per_user_metrics(ranked, ev.targets, model.vocab)
    res["n_users"] = len(ev)
    res["method"] = tag
    res["checkpoint"] = ckpt
    json_dump(res, os.path.join(out, f"metrics_{tag}.json"))
    np.savez(os.path.join(out, f"{tag}_ranked.npz"), uids=ev.uids, ranked=ranked,
             **{f"per_user_{k}": v for k, v in res["per_user"].items()})
    log.info("%s test: %s", tag, {k: round(v, 4) for k, v in res["summary"].items()})
    return res
