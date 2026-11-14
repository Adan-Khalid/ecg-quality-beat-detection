"""Evaluation metrics: QRS detection (Se, +P) and quality classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


@dataclass
class BeatMatch:
    tp: int
    fp: int
    fn: int

    @property
    def sensitivity(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def ppv(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def f1(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else float("nan")

    def __add__(self, other: BeatMatch) -> BeatMatch:
        return BeatMatch(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(sensitivity=self.sensitivity, ppv=self.ppv, f1=self.f1)
        return d


def timing_errors(ref: np.ndarray, det: np.ndarray, tol: int) -> np.ndarray:
    """Signed offsets (det - ref, samples) of the matched pairs from ``match_beats``."""
    ref = np.sort(np.asarray(ref, dtype=np.int64))
    det = np.sort(np.asarray(det, dtype=np.int64))
    used = np.zeros(len(det), dtype=bool)
    out = []
    for r in ref:
        lo = np.searchsorted(det, r - tol, side="left")
        hi = np.searchsorted(det, r + tol, side="right")
        cand = [j for j in range(lo, hi) if not used[j]]
        if cand:
            j = min(cand, key=lambda k: abs(int(det[k]) - int(r)))
            used[j] = True
            out.append(int(det[j]) - int(r))
    return np.asarray(out, dtype=np.int64)


def match_beats(ref: np.ndarray, det: np.ndarray, tol: int) -> tuple[BeatMatch, np.ndarray]:
    """One-to-one matching of detections to reference beats within +-``tol`` samples.

    Greedy in time order: each reference beat takes the nearest unused detection inside
    its window. This is the matching rule of ANSI/AAMI EC57 with a 150 ms window when
    ``tol`` is 150 ms in samples. Returns the counts and a boolean array marking which
    reference beats were found.
    """
    ref = np.sort(np.asarray(ref, dtype=np.int64))
    det = np.sort(np.asarray(det, dtype=np.int64))
    used = np.zeros(len(det), dtype=bool)
    found = np.zeros(len(ref), dtype=bool)
    if len(det):
        for i, r in enumerate(ref):
            lo = np.searchsorted(det, r - tol, side="left")
            hi = np.searchsorted(det, r + tol, side="right")
            best, best_d = -1, None
            for j in range(lo, hi):
                if not used[j]:
                    d = abs(int(det[j]) - int(r))
                    if best_d is None or d < best_d:
                        best, best_d = j, d
            if best >= 0:
                used[best] = True
                found[i] = True
    tp = int(found.sum())
    return BeatMatch(tp=tp, fp=int(len(det) - used.sum()), fn=int(len(ref) - tp)), found


def detector_agreement(a: np.ndarray, b: np.ndarray, tol: int) -> float:
    """bSQI: matched beats over the union of beats from two detectors (Li et al. 2008).

    Returns 1 when the detectors agree on every beat, 0 when they share none. Defined as
    1.0 when both detectors find nothing (no evidence of disagreement), which the
    classifier sees together with the beat count.
    """
    if len(a) == 0 and len(b) == 0:
        return 1.0
    m, _ = match_beats(a, b, tol)
    return m.tp / (len(a) + len(b) - m.tp)


def quality_report(y_true: np.ndarray, y_pred: np.ndarray, labels=(1, 2, 3)) -> dict:
    """Macro-F1, per-class F1 and the confusion matrix for the 3 quality classes."""
    labels = list(labels)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "f1_per_class": {str(k): float(v) for k, v in zip(labels, per_class, strict=True)},
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n": int(len(y_true)),
    }
