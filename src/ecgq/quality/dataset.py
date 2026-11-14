"""Load the processed BUT QDB windows into flat arrays for model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecgq.io import subject_of


@dataclass
class QualityData:
    X_ecg: np.ndarray
    X_acc: np.ndarray
    y: np.ndarray
    record: np.ndarray
    subject: np.ndarray
    start: np.ndarray
    cnn_x: np.ndarray  # (n, 4, 1000) float16: ECG + 3 ACC channels

    def __len__(self) -> int:
        return len(self.y)


def load_labelled(processed_dir: Path, records: list[str] | None = None) -> QualityData:
    files = sorted(Path(processed_dir).glob("*.npz"))
    if records:
        files = [f for f in files if f.stem in records]
    parts = {k: [] for k in ("X_ecg", "X_acc", "y", "record", "subject", "start", "cnn_x")}
    for f in files:
        d = np.load(f)
        idx = d["cnn_idx"]
        parts["X_ecg"].append(d["ecg_feats"][idx])
        parts["X_acc"].append(d["acc_feats"][idx])
        parts["y"].append(d["labels"][idx].astype(int))
        parts["record"].append(np.full(len(idx), f.stem))
        parts["subject"].append(np.full(len(idx), subject_of(f.stem)))
        parts["start"].append(d["starts"][idx])
        parts["cnn_x"].append(d["cnn_x"])
    return QualityData(**{k: np.concatenate(v) for k, v in parts.items()})
