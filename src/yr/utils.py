"""Small shared helpers: config loading, seeding, logging, atomic JSON."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "configs", "default.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(tag: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(tag)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def json_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


class Timer:
    def __init__(self, log: logging.Logger, name: str):
        self.log, self.name = log, name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *exc):
        self.log.info("%s: %.1f s", self.name, time.time() - self.t)
