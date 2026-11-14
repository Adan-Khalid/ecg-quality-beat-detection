"""Two-moving-average QRS detector (Elgendi).

Reference: M. Elgendi, "Fast QRS Detection with an Optimized Knowledge-Based Method:
Evaluation on 11 Standard ECG Databases", PLoS ONE 8(9): e73557, 2013.

Stages:
1. Zero-phase Butterworth band-pass, 8 to 20 Hz (third order).
2. Squaring, which emphasises the large QRS slopes.
3. Two moving averages: MA_qrs over W1 = 97 ms (about one QRS width) and MA_beat over
   W2 = 611 ms (about one heartbeat).
4. Blocks of interest where MA_qrs > MA_beat + alpha, with alpha = beta * mean(y^2) and
   beta = 0.08. Blocks narrower than W1 are rejected as noise.
5. The R peak is the largest absolute filtered value inside each block. A 300 ms
   refractory rule removes double detections.

This detector shares no state with Pan-Tompkins (no adaptive thresholds, no search-back),
which is what makes the agreement between the two a useful quality index.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ecgq.filters import bandpass


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average (zero phase), same length as ``x``."""
    width = max(1, int(width))
    return ndimage.uniform_filter1d(x, size=width, mode="nearest")


def elgendi(
    x: np.ndarray,
    fs: float,
    w1_s: float = 0.097,
    w2_s: float = 0.611,
    beta: float = 0.08,
    refractory_s: float = 0.3,
) -> np.ndarray:
    """Detect QRS complexes. Returns sample indices at the input rate ``fs``."""
    x = np.asarray(x, dtype=float)
    if len(x) < int(fs):
        return np.array([], dtype=int)
    y = bandpass(x, fs, 8.0, min(20.0, 0.45 * fs), order=3)
    sq = y**2
    w1, w2 = int(round(w1_s * fs)), int(round(w2_s * fs))
    ma_qrs = _moving_average(sq, w1)
    ma_beat = _moving_average(sq, w2)
    alpha = beta * float(np.mean(sq))
    blocks = ma_qrs > (ma_beat + alpha)

    # Rising and falling edges of the boolean block signal.
    edges = np.diff(blocks.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    abs_y = np.abs(y)
    peaks = []
    for s, e in zip(starts, ends, strict=True):
        if e - s >= w1:
            peaks.append(s + int(np.argmax(abs_y[s:e])))
    peaks_arr = np.asarray(peaks, dtype=int)
    if len(peaks_arr) < 2:
        return peaks_arr

    # Refractory period: within 300 ms keep the larger peak.
    refr = int(round(refractory_s * fs))
    kept = [int(peaks_arr[0])]
    for p in peaks_arr[1:]:
        if p - kept[-1] < refr:
            if abs_y[p] > abs_y[kept[-1]]:
                kept[-1] = int(p)
        else:
            kept.append(int(p))
    return np.asarray(kept, dtype=int)
