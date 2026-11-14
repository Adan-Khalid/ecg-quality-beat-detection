"""Signal quality indices (SQIs) and accelerometer features for 10 s ECG windows.

Every ECG feature is scale-invariant (ratios, correlations, shape statistics or counts),
so a model trained on BUT QDB (microvolts, 1000 Hz, chest patch) can be applied to
MIT-BIH (millivolts, 360 Hz, Holter) without re-calibration.

Sources for the classic indices:
* bSQI (detector agreement), kSQI, sSQI, pSQI, basSQI: Li, Mark and Clifford (2008),
  Clifford et al. (2012), Behar et al. (2013).
* tSQI (template correlation): Orphanidou et al. (2015).
The remaining features (RR plausibility, spectral entropy, flat-line and high-frequency
fractions, QRS energy ratio) are standard descriptors added here; see docs/LEARNING.md.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

from ecgq.metrics import detector_agreement

ECG_FEATURES = [
    "bsqi", "hr_pt", "hr_el", "hr_diff", "rr_cv", "rr_max_s", "ksqi", "ssqi", "psqi",
    "bassqi", "hf_ratio", "spec_entropy", "tsqi", "max_to_r", "flat_frac", "zcr",
    "qrs_energy_ratio",
]  # fmt: skip
ACC_FEATURES = [
    "acc_mag_std", "acc_jerk", "acc_axis_std_max", "acc_p_low", "acc_p_high", "acc_range",
    "acc_tilt_change",
]  # fmt: skip


def _band_power(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f >= lo) & (f < hi)
    return pxx[..., m].sum(axis=-1)


def spectral_features(clean: np.ndarray, raw: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Vectorised spectral SQIs for a batch of windows, shape (n_windows, n_samples).

    ``clean`` is band-limited ECG (0.5 to 40 Hz); ``raw`` is the resampled ECG without the
    high-pass or low-pass (it still holds baseline wander and muscle noise).
    """
    nper = min(clean.shape[-1], int(2 * fs))
    f, p_clean = signal.welch(clean, fs=fs, nperseg=nper, axis=-1)
    _, p_raw = signal.welch(raw - raw.mean(axis=-1, keepdims=True), fs=fs, nperseg=nper, axis=-1)
    eps = 1e-12
    psqi = _band_power(f, p_clean, 5, 15) / (_band_power(f, p_clean, 5, 40) + eps)
    bassqi = 1 - _band_power(f, p_raw, 0, 1) / (_band_power(f, p_raw, 0, 40) + eps)
    hf_ratio = _band_power(f, p_raw, 40, fs / 2) / (_band_power(f, p_raw, 0, fs / 2) + eps)
    pn = p_clean[:, (f >= 0.5) & (f <= 40)]
    pn = pn / (pn.sum(axis=-1, keepdims=True) + eps)
    entropy = -(pn * np.log(pn + eps)).sum(axis=-1) / np.log(pn.shape[-1])
    return {"psqi": psqi, "bassqi": bassqi, "hf_ratio": hf_ratio, "spec_entropy": entropy}


def beat_features(
    win: np.ndarray, fs: float, beats_pt: np.ndarray, beats_el: np.ndarray
) -> dict[str, float]:
    """Beat-based SQIs for one window. Beat indices are relative to the window start."""
    dur = len(win) / fs
    tol = int(round(0.15 * fs))
    out = {
        "bsqi": detector_agreement(beats_pt, beats_el, tol),
        "hr_pt": 60.0 * len(beats_pt) / dur,
        "hr_el": 60.0 * len(beats_el) / dur,
    }
    out["hr_diff"] = abs(out["hr_pt"] - out["hr_el"])
    if len(beats_pt) >= 3:
        rr = np.diff(beats_pt) / fs
        out["rr_cv"] = float(rr.std() / rr.mean())
        out["rr_max_s"] = float(max(rr.max(), beats_pt[0] / fs, dur - beats_pt[-1] / fs))
    else:
        out["rr_cv"] = 1.0
        out["rr_max_s"] = dur

    # Template correlation: each beat (+-250 ms) against the median beat.
    half = int(0.25 * fs)
    ok = beats_pt[(beats_pt >= half) & (beats_pt < len(win) - half)]
    if len(ok) >= 3:
        seg = np.stack([win[b - half : b + half] for b in ok])
        tmpl = np.median(seg, axis=0)
        sc = seg - seg.mean(axis=1, keepdims=True)
        tc = tmpl - tmpl.mean()
        den = np.linalg.norm(sc, axis=1) * np.linalg.norm(tc) + 1e-12
        out["tsqi"] = float(np.mean(sc @ tc / den))
        r_amp = np.median(np.abs(win[ok]))
        out["max_to_r"] = float(np.percentile(np.abs(win), 99.5) / (r_amp + 1e-12))
        mask = np.zeros(len(win), dtype=bool)
        w = int(0.05 * fs)
        for b in ok:
            mask[max(0, b - w) : b + w] = True
        e_in = np.mean(win[mask] ** 2)
        e_out = np.mean(win[~mask] ** 2) + 1e-12
        out["qrs_energy_ratio"] = float(np.log10(e_in / e_out + 1e-12))
    else:
        out["tsqi"] = 0.0
        out["max_to_r"] = 10.0
        out["qrs_energy_ratio"] = 0.0
    return out


def shape_features(win: np.ndarray, fs: float) -> dict[str, float]:
    d = np.diff(win)
    mad = np.median(np.abs(d - np.median(d))) + 1e-12
    centred = win - np.median(win)
    return {
        "ksqi": float(stats.kurtosis(win, fisher=False)),
        "ssqi": float(abs(stats.skew(win))),
        "flat_frac": float(np.mean(np.abs(d) < 0.01 * mad)),
        "zcr": float(np.mean(np.signbit(centred[1:]) != np.signbit(centred[:-1])) * fs),
    }


def acc_features(acc: np.ndarray, fs_acc: float) -> np.ndarray:
    """Accelerometer features for a batch of windows, shape (n_windows, n_samples, 3)."""
    mag = np.linalg.norm(acc, axis=-1)
    nper = min(mag.shape[-1], int(4 * fs_acc))
    f, p = signal.welch(mag - mag.mean(axis=-1, keepdims=True), fs=fs_acc, nperseg=nper, axis=-1)
    # Tilt change: angle between the mean gravity direction in the first and last second.
    k = int(fs_acc)
    g0 = acc[:, :k].mean(axis=1)
    g1 = acc[:, -k:].mean(axis=1)
    cos = (g0 * g1).sum(-1) / (np.linalg.norm(g0, axis=-1) * np.linalg.norm(g1, axis=-1) + 1e-9)
    feats = np.column_stack([
        mag.std(axis=-1),
        np.abs(np.diff(mag, axis=-1)).mean(axis=-1) * fs_acc,
        acc.std(axis=1).max(axis=-1),
        np.log10(_band_power(f, p, 0.5, 3.0) + 1e-6),
        np.log10(_band_power(f, p, 3.0, 15.0) + 1e-6),
        np.percentile(mag, 99, axis=-1) - np.percentile(mag, 1, axis=-1),
        np.degrees(np.arccos(np.clip(cos, -1, 1))),
    ])
    return feats
