#!/usr/bin/env python3
"""Single entry point for every pipeline stage."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from yr import analyze, baselines, eda, etl, evaluate, reward, rl, train_lm  # noqa: E402
from yr.utils import load_config, seed_everything, setup_logging  # noqa: E402


def _set(cfg: dict, expr: str) -> None:
    import yaml

    key, val = expr.split("=", 1)
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = yaml.safe_load(val)


def _quick(cfg: dict) -> None:
    cfg["train"].update(steps=40, eval_every=20, val_users=50, time_budget_min=10, warmup=5)
    cfg["rm"].update(max_events=4000, epochs=1, ensemble=1)
    cfg["rl"].update(prompts=16, batch_prompts=2, group_size=2, horizon=3, epochs=1)
    cfg["analysis"].update(bootstrap=200, multi_step_users=20)
    cfg["eval"].update(batch_users=8, score_chunk=50000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "etl", "eda", "baseline", "train", "eval", "rm", "rl", "analyze"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default="auto", choices=["auto", "kaggle", "hf"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="dotted config override, e.g. rl.kl_beta=0.01")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for expr in args.set:
        _set(cfg, expr)
    if args.quick:
        _quick(cfg)
    os.makedirs(cfg["data_dir"], exist_ok=True)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    log = setup_logging("yambda")
    fh = logging.FileHandler(os.path.join(cfg["out_dir"], "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    seed_everything(cfg["seed"])
    log.info("stage=%s quick=%s cfg=%s", args.stage, args.quick, json.dumps(cfg, ensure_ascii=False))

    def data():
        if not os.path.exists(os.path.join(cfg["data_dir"], "train_seq.npz")):
            etl.run(cfg, log, source=args.source)

    def base():
        data()
        if not os.path.exists(os.path.join(cfg["out_dir"], "ckpt", "base.pt")):
            train_lm.run(cfg, log, resume=args.resume)

    def rms():
        base()
        if not os.path.exists(os.path.join(cfg["out_dir"], "rm_0.pt")):
            reward.run(cfg, log)

    ok = False
    t0 = time.time()
    try:
        if args.stage in ("all", "etl"):
            etl.run(cfg, log, source=args.source)
        if args.stage in ("all", "etl", "eda"):
            eda.run(cfg, log)
        if args.stage in ("all", "etl", "baseline"):
            baselines.run(cfg, log)
        if args.stage in ("all", "train"):
            base()
            evaluate.run_test(cfg, log, tag="base")
        if args.stage in ("all", "eval"):
            evaluate.run_test(cfg, log, tag="base")
        if args.stage in ("all", "rm"):
            rms()
        if args.stage in ("all", "rl"):
            rms()
            rl.run(cfg, log)
            tag = cfg["rl"].get("tag") or f"rl_{cfg['rl']['method']}"
            evaluate.run_test(cfg, log, ckpt=os.path.join(cfg["out_dir"], f"{tag}.pt"), tag=tag)
            analyze.run(cfg, log)
        if args.stage == "analyze":
            analyze.run(cfg, log)
        ok = True
    finally:
        log.info("finished stage %s in %.1f min, ok=%s", args.stage, (time.time() - t0) / 60, ok)
        stats = os.path.join(cfg["data_dir"], "stats.json")
        if os.path.exists(stats):
            shutil.copy(stats, os.path.join(cfg["out_dir"], "stats.json"))
        if not cfg.get("keep_data", False):
            shutil.rmtree(cfg["data_dir"], ignore_errors=True)
            shutil.rmtree(os.path.join(cfg["out_dir"], "ckpt"), ignore_errors=True)
            for f in os.listdir(cfg["out_dir"]):
                if (f.startswith("rm_") or f.startswith("rl_")) and f.endswith(".pt"):
                    os.remove(os.path.join(cfg["out_dir"], f))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
