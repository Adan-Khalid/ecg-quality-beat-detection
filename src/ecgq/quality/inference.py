"""Load the trained quality models and score windows of any ECG recording."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ecgq.quality.models import build_cnn, predict_cnn
from ecgq.utils import REPO_ROOT

MODEL_DIR = REPO_ROOT / "models"


def load_sklearn_models(model_dir: Path = MODEL_DIR) -> dict:
    with open(Path(model_dir) / "sklearn_models.pkl", "rb") as f:
        return pickle.load(f)  # noqa: S301 (file produced by scripts/train_final.py)


def load_cnn(name: str, model_dir: Path = MODEL_DIR):
    import torch

    ckpt = torch.load(Path(model_dir) / f"{name}.pt", map_location="cpu", weights_only=True)
    net = build_cnn(len(ckpt["channels"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["channels"]


def predict_windows(name: str, X_ecg: np.ndarray, X_acc: np.ndarray | None = None,
                    cnn_x: np.ndarray | None = None, models: dict | None = None) -> np.ndarray:
    """Predict quality classes (1, 2, 3) with one of the saved models."""
    if name.startswith("cnn"):
        net, ch = load_cnn(name)
        pred, _ = predict_cnn(net, cnn_x[:, ch].astype(np.float32))
        return pred
    models = models or load_sklearn_models()
    X = X_ecg if name in ("rule", "gbm_ecg") else np.hstack([X_ecg, X_acc])
    return np.asarray(models[name].predict(X)).astype(int)
