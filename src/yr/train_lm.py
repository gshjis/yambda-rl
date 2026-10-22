"""Train the GPT-2 next-item recommender."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from .data import Split, WindowDataset, collate_windows
from .evaluate import EvalSet, evaluate
from .model import GPT, sampled_softmax_loss
from .utils import Timer, json_dump


def _lr(step: int, cfg: dict) -> float:
    warm = cfg["train"]["warmup"]
    base = cfg["train"]["lr"]
    if step < warm:
        return base * (step + 1) / warm
    progress = (step - warm) / max(1, cfg["train"]["steps"] - warm)
    return base * 0.1 + base * 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def run(cfg: dict, log, resume: bool = False) -> dict:
    os.makedirs(os.path.join(cfg["out_dir"], "ckpt"), exist_ok=True)
    data_dir = cfg["data_dir"]
    items = np.load(os.path.join(data_dir, "items.npz"))
    vocab = len(items["item_ids"])
    train = Split(os.path.join(data_dir, "train_seq.npz"))
    val_hist = train
    val_targets = Split(os.path.join(data_dir, "val_targets.npz"))
    ev = EvalSet(val_hist, val_targets, cfg["data"]["max_seq_len"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = cfg["model"]
    model = GPT(vocab=vocab, **{k: mcfg[k] for k in ("n_layer", "n_head", "n_embd", "ctx", "dropout")}).to(device)
    log.info("model params: %.1fM, vocab %d, device %s", sum(p.numel() for p in model.parameters()) / 1e6, vocab, device)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "tok" in name or "pos" in name else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["train"]["lr"], betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ds = WindowDataset(train, cfg["data"]["max_seq_len"], samples=10**9, seed=cfg["seed"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg["train"]["batch_size"],
        sampler=torch.utils.data.RandomSampler(
            ds, replacement=True,
            num_samples=cfg["train"]["steps"] * cfg["train"]["batch_size"] + cfg["train"]["batch_size"],
        ),
        collate_fn=lambda b: collate_windows(b, pad_id=0),
        num_workers=0, drop_last=True,
    )
    start_step = 0
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt", "base.pt")
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        log.info("resumed at step %d", start_step)

    t0 = time.time()
    best = -1.0
    history = []
    step = start_step
    while step < cfg["train"]["steps"]:
        for x, y, lengths in loader:
            if step >= cfg["train"]["steps"]:
                break
            if (time.time() - t0) / 60 > cfg["train"]["time_budget_min"]:
                log.warning("time budget reached at step %d", step)
                step = cfg["train"]["steps"]
                break
            x, y = x.to(device), y.to(device)
            lr = _lr(step, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                h = model(x)
                last = h[torch.arange(x.shape[0], device=device), lengths.to(device) - 1]
                loss = sampled_softmax_loss(last, y.unsqueeze(1), model.tok.weight,
                                            uniform_neg=cfg["train"]["negatives"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if not torch.isfinite(loss) or not torch.isfinite(gnorm):
                log.error("non-finite loss %.4f or grad norm %.4f at step %d", float(loss), float(gnorm), step)
                break
            if step % cfg["train"]["log_every"] == 0:
                log.info("step %d loss %.4f lr %.2e %.0f tok/s", step, loss.item(), lr,
                         cfg["train"]["batch_size"] * cfg["data"]["max_seq_len"] * cfg["train"]["log_every"] / (time.time() - t0))
                history.append({"step": step, "loss": float(loss.item()), "lr": lr})
            if step > 0 and step % cfg["train"]["eval_every"] == 0:
                res = evaluate(model, ev, cfg, device, max_users=cfg["train"]["val_users"])
                ndcg = res["summary"]["ndcg@10"]
                log.info("val ndcg@10 %.4f recall@100 %.4f (n=%d) max|emb|=%.3e", ndcg,
                         res["summary"]["recall@100"], res["n_users"],
                         float(model.tok.weight.abs().max()))
                history.append({"step": step, "val_ndcg@10": ndcg})
                if ndcg > best:
                    best = ndcg
                    torch.save({"config": model.config, "state_dict": model.state_dict(), "step": step},
                               os.path.join(cfg["out_dir"], "ckpt", "base_best.pt"))
                model.train()
            if step % 500 == 0:
                torch.save({"config": model.config, "state_dict": model.state_dict(), "opt": opt.state_dict(),
                            "step": step}, ckpt_path)
            step += 1

    torch.save({"config": model.config, "state_dict": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, ckpt_path)
    json_dump({"train_seconds": time.time() - t0, "history": history, "best_val_ndcg@10": best},
              os.path.join(cfg["out_dir"], "metrics_train.json"))
    log.info("training done in %.1f min", (time.time() - t0) / 60)
    return {"best_val_ndcg@10": best}
