"""Synthetic ECG for unit tests and CI (no downloads needed).

A beat is a sum of Gaussian bumps for the P, Q, R, S and T waves, a simplified version
of the dynamical model of McSharry et al. (IEEE TBME 2003). It is only for testing code
paths, never for reporting results.
"""

from __future__ import annotations

import numpy as np

# (centre in seconds relative to R, width in seconds, amplitude in mV)
WAVES = {
    "P": (-0.20, 0.025, 0.12),
    "Q": (-0.03, 0.010, -0.10),
    "R": (0.00, 0.010, 1.00),
    "S": (0.03, 0.010, -0.25),
    "T": (0.25, 0.050, 0.30),
}


def synthetic_ecg(
    duration_s: float = 60.0,
    fs: float = 250.0,
    hr_bpm: float = 70.0,
    hrv_std_s: float = 0.03,
    noise_std: float = 0.02,
    baseline_amp: float = 0.0,
    mains_amp: float = 0.0,
    mains_hz: float = 50.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (signal in mV, true R-peak sample indices)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    rr_mean = 60.0 / hr_bpm
    r_times = []
    tr = 0.5
    while tr < duration_s - 0.5:
        r_times.append(tr)
        tr += max(0.3, rr_mean + rng.normal(0, hrv_std_s))
    x = np.zeros(n)
    for rt in r_times:
        for c, w, a in WAVES.values():
            lo, hi = int((rt + c - 5 * w) * fs), int((rt + c + 5 * w) * fs) + 1
            lo, hi = max(lo, 0), min(hi, n)
            tt = t[lo:hi]
            x[lo:hi] += a * np.exp(-0.5 * ((tt - rt - c) / w) ** 2)
    x += noise_std * rng.standard_normal(n)
    if baseline_amp:
        x += baseline_amp * np.sin(2 * np.pi * 0.25 * t)
    if mains_amp:
        x += mains_amp * np.sin(2 * np.pi * mains_hz * t)
    return x, np.round(np.asarray(r_times) * fs).astype(int)


def motion_burst(n: int, fs: float, start_s: float, dur_s: float, amp: float, seed: int = 1):
    """Low-frequency high-amplitude artefact resembling electrode motion."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    lo, hi = int(start_s * fs), min(n, int((start_s + dur_s) * fs))
    m = hi - lo
    if m > 0:
        walk = np.cumsum(rng.standard_normal(m))
        walk = (walk - walk.mean()) / (walk.std() + 1e-9)
        spikes = rng.standard_normal(m) * (rng.random(m) < 0.02) * 3
        out[lo:hi] = amp * (walk + spikes)
    return out
