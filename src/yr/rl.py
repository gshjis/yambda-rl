"""GRPO / DPO / filtered-BC fine-tuning of the item-language policy.

The environment is simulated: the generated item is appended to the context and
the reward model scores the resulting (history, item) pair. This is open-loop RL
without real user feedback; it is the main limitation analysed in the report.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .data import Split
from .evaluate import load_model
from .reward import item_features, load_ensemble
from .utils import json_dump


def _pad(seqs: list[list[int]], device: str):
    t = max(1, max(len(s) for s in seqs))
    x = torch.zeros((len(seqs), t), dtype=torch.long, device=device)
    ln = torch.ones(len(seqs), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        ln[i] = max(1, len(s))
    return x, ln


@torch.no_grad()
def generate(model, prompts: list[list[int]], group: int, horizon: int, ctx: int,
             top_k: int, temperature: float, device: str):
    seqs = [list(p)[-ctx:] for p in prompts for _ in range(group)]
    actions = [[] for _ in seqs]
    for _ in range(horizon):
        x, ln = _pad([s[-ctx:] for s in seqs], device)
        h = model(x)[torch.arange(len(seqs), device=device), ln - 1]
        from .model import rank_all

        top = rank_all(h, model.tok.weight, top_k=top_k)
        scores = torch.einsum("bd,bkd->bk", h, model.tok.weight[top]) / temperature
        probs = torch.softmax(scores, dim=1)
        pick = torch.multinomial(probs, 1).squeeze(1)
        a = top[torch.arange(len(seqs), device=device), pick]
        for i in range(len(seqs)):
            seqs[i].append(int(a[i]))
            actions[i].append(int(a[i]))
    return seqs, actions


def sequence_logprobs(model, prompts: list[list[int]], actions: list[list[int]], ctx: int,
                      device: str, chunk: int = 50000) -> torch.Tensor:
    """Sum of per-action log-probs under the full-catalog softmax."""
    b = len(prompts)
    hz = len(actions[0])
    total = torch.zeros(b, device=device)
    for t in range(hz):
        seqs = [prompts[i] + actions[i][:t] for i in range(b)]
        x, ln = _pad([s[-ctx:] for s in seqs], device)
        h = model(x)[torch.arange(b, device=device), ln - 1]
        a = torch.tensor([actions[i][t] for i in range(b)], dtype=torch.long, device=device)
        act = (h * model.tok.weight[a]).sum(1)
        lse = torch.full((b,), -torch.inf, device=device)
        for start in range(0, model.vocab, chunk):
            s = h @ model.tok.weight[start : start + chunk].T
            lse = torch.logaddexp(lse, torch.logsumexp(s.float(), dim=1))
        total = total + act.float() - lse
    return total


def _returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    b, h = rewards.shape
    return (rewards * (gamma ** np.arange(h))[None, :]).sum(1)


def _advantages(r: np.ndarray, group: int, clip: float) -> np.ndarray:
    r = r.reshape(-1, group)
    adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)
    return np.clip(adv.reshape(-1), -clip, clip)


@torch.no_grad()
def score_actions(models, prompts: list[list[int]], actions: list[list[int]], histories_stats,
                  item_feats: torch.Tensor, items: dict, cfg: dict, device: str) -> dict:
    """Return RM components per (sequence, step)."""
    rl = cfg["rl"]
    ctx = min(int(cfg["rm"]["ctx"]), int(cfg["data"]["max_seq_len"]))
    b, h = len(prompts), len(actions[0])
    hists, cands, feats = [], [], []
    for i in range(b):
        for t in range(h):
            hists.append((prompts[i] + actions[i][:t])[-ctx:])
            cands.append(actions[i][t])
            feats.append(histories_stats[i])
    scores = {k: [] for k in ("ratio", "ratio_std", "like", "session", "return")}
    for start in range(0, len(hists), 2048):
        part = slice(start, min(start + 2048, len(hists)))
        x, ln = _pad(hists[part], device)
        cand = torch.tensor(cands[part], dtype=torch.long, device=device)
        fs = torch.tensor(np.stack(feats[part]), dtype=torch.float32, device=device)
        fs = torch.cat([fs, item_feats[cand]], dim=1)
        outs = {k: [] for k in ("ratio", "like", "session", "return")}
        for m in models:
            o = m(x, cand, fs, ln)
            for k in outs:
                outs[k].append(torch.sigmoid(o[k]))
        for k in outs:
            arr = torch.stack(outs[k])
            scores[k].append(arr.mean(0))
            if k == "ratio":
                scores["ratio_std"].append(arr.std(0))
    return {k: torch.cat(v).reshape(b, h).cpu().numpy() for k, v in scores.items()}


def _reward(scores: dict, prompts: list[list[int]], actions: list[list[int]], item_feats: np.ndarray,
            cfg: dict) -> np.ndarray:
    rl = cfg["rl"]
    kind = rl["reward"]
    if kind == "ratio":
        r = scores["ratio"] - rl["pessimistic"] * scores["ratio_std"]
    elif kind == "like":
        r = scores["like"]
    elif kind == "session":
        r = scores["session"]
    elif kind == "return":
        r = scores["return"]
    elif kind == "blend":
        r = scores["ratio"] + scores["like"] - rl["pessimistic"] * scores["ratio_std"]
    else:
        raise ValueError(kind)
    pop = np.log1p(item_feats[:, 0]) if item_feats is not None else 0.0
    for i in range(len(prompts)):
        seen = set(prompts[i])
        for t, a in enumerate(actions[i]):
            if rl["popularity_penalty"]:
                r[i, t] -= rl["popularity_penalty"] * float(item_feats[a, 0])
            if rl["repeat_penalty"] and a in seen:
                r[i, t] -= rl["repeat_penalty"]
            if rl["length_penalty"]:
                r[i, t] -= rl["length_penalty"] * float(item_feats[a, 1])
            seen.add(a)
    return r


def run(cfg: dict, log) -> dict:
    rl = cfg["rl"]
    data_dir = cfg["data_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = np.load(os.path.join(data_dir, "items.npz"))
    counts = items["counts"].astype(np.float64)
    item_feats_np = np.stack([
        np.log1p(counts) / np.log1p(counts.max()),
        np.nan_to_num(items["mean_length"], nan=0.0) / 600.0,
        items["like_rate"],
    ], axis=1).astype(np.float32)
    item_feats = torch.from_numpy(item_feats_np).to(device)

    base_ckpt = os.path.join(cfg["out_dir"], "ckpt", "base_best.pt")
    if not os.path.exists(base_ckpt):
        base_ckpt = os.path.join(cfg["out_dir"], "ckpt", "base.pt")
    policy = load_model(base_ckpt, device).train()
    ref = load_model(base_ckpt, device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    models = load_ensemble(cfg, device)
    log.info("rl: method=%s reward=%s beta=%s horizon=%d", rl["method"], rl["reward"],
             rl["kl_beta"], rl["horizon"])

    val = Split(os.path.join(data_dir, "val_seq.npz"))
    lengths = np.diff(val.offsets)
    users = np.flatnonzero(lengths >= 5)
    rng = np.random.default_rng(cfg["seed"])
    n_prompts = min(int(rl["prompts"]), len(users))
    chosen = rng.choice(users, size=n_prompts, replace=False)
    prompts = [val.items[val.offsets[u] : val.offsets[u + 1]][-cfg["data"]["max_seq_len"]:].astype(np.int64).tolist()
               for u in chosen]
    prompt_stats = []
    for u in chosen:
        lo = val.offsets[u]
        prompt_stats.append([float(val.ratio[lo:].mean()) / 100.0, float(val.organic[lo:].mean()),
                             min(lengths[u], cfg["data"]["max_seq_len"]) / cfg["data"]["max_seq_len"]])
    prompt_stats = np.array(prompt_stats, dtype=np.float32)

    opt = torch.optim.AdamW(policy.parameters(), lr=rl["lr"], betas=(0.9, 0.95))
    samples = []
    hist = []
    for epoch in range(int(rl["epochs"])):
        order = rng.permutation(n_prompts)
        for start in range(0, n_prompts, int(rl["batch_prompts"])):
            idx = order[start : start + int(rl["batch_prompts"])]
            batch_prompts = [prompts[i] for i in idx]
            seqs, actions = generate(policy, batch_prompts, int(rl["group_size"]), int(rl["horizon"]),
                                     cfg["data"]["max_seq_len"], int(rl["top_k"]), rl["temperature"], device)
            g = int(rl["group_size"])
            seq_prompts = [p for p in batch_prompts for _ in range(g)]
            seq_stats = np.repeat(prompt_stats[idx], g, axis=0)
            scores = score_actions(models, seq_prompts, actions, seq_stats, item_feats, items, cfg, device)
            rewards = _reward(scores, seq_prompts, actions, item_feats_np, cfg)
            ret = _returns(rewards, rl["gamma"])
            hz = int(rl["horizon"])
            if rl["method"] == "grpo":
                adv = torch.tensor(_advantages(ret, g, rl["clip_adv"]), device=device)
                lp = sequence_logprobs(policy, seq_prompts, actions, cfg["data"]["max_seq_len"], device) / hz
                with torch.no_grad():
                    lp_ref = sequence_logprobs(ref, seq_prompts, actions, cfg["data"]["max_seq_len"], device) / hz
                diff = lp - lp_ref
                kl = (diff.exp() - diff - 1).mean()
                loss = -(adv * lp).mean() + rl["kl_beta"] * kl
            elif rl["method"] == "dpo":
                best = ret.reshape(-1, g).argmax(1)
                worst = ret.reshape(-1, g).argmin(1)
                pw = [batch_prompts[i] for i in range(len(batch_prompts))]
                aw = [actions[i * g + int(best[i])] for i in range(len(batch_prompts))]
                al = [actions[i * g + int(worst[i])] for i in range(len(batch_prompts))]
                lp_w = sequence_logprobs(policy, pw, aw, cfg["data"]["max_seq_len"], device) / hz
                lp_l = sequence_logprobs(policy, pw, al, cfg["data"]["max_seq_len"], device) / hz
                with torch.no_grad():
                    ref_w = sequence_logprobs(ref, pw, aw, cfg["data"]["max_seq_len"], device) / hz
                    ref_l = sequence_logprobs(ref, pw, al, cfg["data"]["max_seq_len"], device) / hz
                margin = (lp_w - lp_l) - (ref_w - ref_l)
                loss = -F.logsigmoid(0.1 * margin).mean()
                kl = torch.tensor(0.0)
            elif rl["method"] == "filtered_bc":
                best = ret.reshape(-1, g).argmax(1)
                ba = [actions[i * g + int(best[i])] for i in range(len(batch_prompts))]
                lp = sequence_logprobs(policy, batch_prompts, ba, cfg["data"]["max_seq_len"], device) / hz
                loss = -lp.mean()
                kl = torch.tensor(0.0)
            else:
                raise ValueError(rl["method"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            row = {
                "step": len(hist),
                "loss": float(loss.detach()),
                "reward": float(ret.mean()),
                "reward_step0": float(rewards[:, 0].mean()),
                "reward_step_last": float(rewards[:, -1].mean()),
                "kl": float(kl.detach()) if torch.is_tensor(kl) else 0.0,
                "repeat_rate": float(np.mean([a in set(p) for p, acts in zip(seq_prompts, actions) for a in acts])),
                "unique_items": int(len({a for acts in actions for a in acts})),
            }
            hist.append(row)
            log.info("rl %s", row)
            for i in range(len(seq_prompts)):
                samples.append({
                    "uid": int(val.uids[idx[i // g]]),
                    "actions": actions[i],
                    "rewards": rewards[i].tolist(),
                    "return": float(ret[i]),
                })

    tag = f"rl_{rl['method']}"
    torch.save({"config": policy.config, "state_dict": policy.state_dict()},
               os.path.join(cfg["out_dir"], f"{tag}.pt"))
    json_dump({"history": hist, "method": rl["method"], "reward": rl["reward"]},
              os.path.join(cfg["out_dir"], "metrics_rl_train.json"))
    json_dump(samples[:5000], os.path.join(cfg["out_dir"], f"{tag}_samples.json"))
    log.info("rl done: mean reward %.4f", float(np.mean([h['reward'] for h in hist])) if hist else 0.0)
    return {"history": hist}
