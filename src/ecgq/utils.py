"""Small shared helpers: config loading, seeding, run logging."""

from __future__ import annotations

import json
import logging
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (seeds third-party code that uses the legacy API)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for the signal-processing parts
        pass


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Logger that writes to the console and to reports/logs/<name>.log."""
    log_dir = Path(log_dir or REPO_ROOT / "reports" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_dir / f"{name}.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info("python %s on %s", platform.python_version(), platform.platform())
    return logger


def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
