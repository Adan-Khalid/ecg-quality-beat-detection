"""Turn a recording into fixed 10 s windows with features, labels and CNN inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgq.detectors import elgendi, pan_tompkins
from ecgq.filters import preprocess, resample
from ecgq.quality.features import (
    ACC_FEATURES,
    ECG_FEATURES,
    acc_features,
    beat_features,
    shape_features,
    spectral_features,
)


def run_blockwise(detector, x: np.ndarray, fs: float, block_s: float = 60.0,
                  overlap_s: float = 5.0) -> np.ndarray:
    """Run a detector on consecutive blocks and merge the results.

    Long recordings contain hours of artefact. Detectors with a global threshold (the
    Elgendi offset uses the mean energy of the whole input) or slowly adapting thresholds
    (Pan-Tompkins) would carry one bad stretch into the next clean one. Blocks of 60 s
    with 5 s overlap keep every decision local. Each block keeps only beats from its
    central part, so overlaps never produce duplicates.
    """
    n = len(x)
    block, ov = int(block_s * fs), int(overlap_s * fs)
    out = []
    for s in range(0, n, block):
        lo, hi = max(0, s - ov), min(n, s + block + ov)
        if hi - lo < int(2 * fs):
            continue
        b = detector(x[lo:hi], fs) + lo
        out.append(b[(b >= s) & (b < min(n, s + block))])
    return np.concatenate(out) if out else np.array([], dtype=int)


@dataclass
class WindowSet:
    starts: np.ndarray  # window start (samples at fs)
    labels: np.ndarray  # consensus class, 0 if unlabelled or mixed
    ecg_feats: np.ndarray  # (n, len(ECG_FEATURES))
    acc_feats: np.ndarray | None  # (n, len(ACC_FEATURES))
    fs: float
    win_s: float
    beats_pt: np.ndarray
    beats_el: np.ndarray
    label_counts: np.ndarray | None = None  # (n, 4) samples of class 0..3 in each window


def window_labels(labels: np.ndarray, starts: np.ndarray, w: int) -> np.ndarray:
    """A window gets a class only if every sample in it carries that same class."""
    out = np.zeros(len(starts), dtype=np.int8)
    for i, s in enumerate(starts):
        seg = labels[s : s + w]
        if len(seg) == w and seg[0] > 0 and np.all(seg == seg[0]):
            out[i] = seg[0]
    return out


def compute_windows(
    ecg_raw: np.ndarray,
    fs: float,
    labels: np.ndarray | None = None,
    acc: np.ndarray | None = None,
    fs_acc: float | None = None,
    win_s: float = 10.0,
    mains: float | None = 50.0,
) -> WindowSet:
    """Features for non-overlapping ``win_s`` windows over a whole recording."""
    ecg_raw = np.asarray(ecg_raw, dtype=np.float64)
    clean = preprocess(ecg_raw, fs, mains=mains)
    beats_pt = run_blockwise(pan_tompkins, clean, fs)
    beats_el = run_blockwise(elgendi, clean, fs)

    w = int(win_s * fs)
    n_win = len(clean) // w
    starts = np.arange(n_win) * w
    cw = clean[: n_win * w].reshape(n_win, w)
    rw = ecg_raw[: n_win * w].reshape(n_win, w)
    spec = spectral_features(cw, rw, fs)

    rows = np.zeros((n_win, len(ECG_FEATURES)))
    ipt = np.searchsorted(beats_pt, starts)
    jpt = np.searchsorted(beats_pt, starts + w)
    iel = np.searchsorted(beats_el, starts)
    jel = np.searchsorted(beats_el, starts + w)
    for i, s in enumerate(starts):
        d = beat_features(cw[i], fs, beats_pt[ipt[i] : jpt[i]] - s, beats_el[iel[i] : jel[i]] - s)
        d.update(shape_features(cw[i], fs))
        for k in ("psqi", "bassqi", "hf_ratio", "spec_entropy"):
            d[k] = spec[k][i]
        rows[i] = [d[k] for k in ECG_FEATURES]

    acc_feats = None
    if acc is not None and fs_acc is not None:
        wa = int(win_s * fs_acc)
        n_acc = min(n_win, len(acc) // wa)
        aw = acc[: n_acc * wa].reshape(n_acc, wa, 3).astype(np.float64)
        acc_feats = np.full((n_win, len(ACC_FEATURES)), np.nan)
        acc_feats[:n_acc] = acc_features(aw, fs_acc)

    lab = window_labels(labels, starts, w) if labels is not None else np.zeros(n_win, np.int8)
    counts = None
    if labels is not None:
        lw = np.asarray(labels[: n_win * w]).reshape(n_win, w)
        counts = np.stack([(lw == c).sum(axis=1) for c in range(4)], axis=1).astype(np.int32)
    return WindowSet(starts, lab, rows, acc_feats, fs, win_s, beats_pt, beats_el, counts)


def cnn_inputs(
    ecg_raw: np.ndarray, fs: float, starts: np.ndarray, win_s: float = 10.0,
    acc: np.ndarray | None = None, fs_acc: float | None = None, fs_cnn: float = 100.0,
    mains: float | None = 50.0,
) -> np.ndarray:
    """CNN input tensor (n, channels, win_s * fs_cnn): robust-scaled ECG plus optional ACC.

    ECG: band-limited, resampled to 100 Hz, centred on the median and divided by the
    interquartile range of the window, then clipped to +-20 so one spike cannot dominate.
    ACC: each axis centred on its window mean (gravity removed) and divided by 100 mg.
    """
    clean = preprocess(np.asarray(ecg_raw, dtype=np.float64), fs, mains=mains)
    w = int(win_s * fs)
    wc = int(win_s * fs_cnn)
    chans = 1 + (3 if acc is not None else 0)
    out = np.zeros((len(starts), chans, wc), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = resample(clean[s : s + w], fs, fs_cnn)[:wc]
        q75, q25 = np.percentile(seg, [75, 25])
        out[i, 0, : len(seg)] = np.clip((seg - np.median(seg)) / (q75 - q25 + 1e-9), -20, 20)
        if acc is not None and fs_acc is not None:
            a0 = int(round(s / fs * fs_acc))
            a = acc[a0 : a0 + int(win_s * fs_acc)].astype(np.float64)
            if len(a):
                a = resample(a.T, fs_acc, fs_cnn)[:, :wc]
                a = (a - a.mean(axis=1, keepdims=True)) / 100.0
                out[i, 1:, : a.shape[1]] = np.clip(a, -20, 20)
    return out
