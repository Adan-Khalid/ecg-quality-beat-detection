"""Pan-Tompkins QRS detector, written from the original paper.

Reference: J. Pan and W. J. Tompkins, "A Real-Time QRS Detection Algorithm",
IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236, 1985.

The paper's filters are integer-coefficient difference equations designed for a
200 Hz sampling rate, so the input is resampled to 200 Hz first. Stages:

1. Low-pass  y(n) = 2y(n-1) - y(n-2) + x(n) - 2x(n-6) + x(n-12)     (gain 36, delay 5)
2. High-pass p(n) = x(n-16) - [y(n-1) + x(n) - x(n-32)] / 32          (delay 16)
   where y(n) = y(n-1) + x(n) - x(n-32) is a 32-point running sum.
3. Derivative y(n) = [2x(n) + x(n-1) - x(n-3) - 2x(n-4)] / 8          (delay 2)
4. Squaring.
5. Moving-window integration over 30 samples (150 ms).
6. Fiducial marks and two sets of adaptive thresholds (integrated and band-passed
   signals), RR-interval averages, search-back for missed beats and T-wave rejection.

Detected locations are mapped back to the input sampling rate and refined to the
largest absolute deflection of the band-passed ECG near the detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ecgq.filters import bandpass, resample

FS_PT = 200.0  # sampling rate the paper's integer filters were designed for
LP_DELAY = 5
HP_DELAY = 16
DER_DELAY = 2
MWI_WIDTH = 30  # 150 ms at 200 Hz
REFRACTORY = 40  # 200 ms at 200 Hz
T_WAVE_WINDOW = 72  # 360 ms at 200 Hz
WARMUP = 11 + 33 + 5 + MWI_WIDTH  # filter start-up transient in samples (about 0.4 s)


def pt_lowpass(x: np.ndarray) -> np.ndarray:
    """Paper eq. (1): H(z) = (1 - z^-6)^2 / (1 - z^-1)^2.

    The paper writes it recursively, with two poles on the unit circle cancelled by
    zeros. Over a 24-hour recording that recursion accumulates round-off error, so the
    identical transfer function is applied in its non-recursive form: (1 - z^-6)/(1 - z^-1)
    is a 6-point running sum, and the square of it is an 11-tap triangle.
    """
    b = np.convolve(np.ones(6), np.ones(6))  # [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
    return signal.lfilter(b, [1.0], x)


def pt_highpass(x: np.ndarray) -> np.ndarray:
    """Paper eq. (3): H(z) = z^-16 - (1/32)(1 - z^-32)/(1 - z^-1).

    An all-pass delay of 16 samples minus a 32-point moving average, applied in its
    non-recursive FIR form for the same round-off reason as the low-pass.
    """
    b = -np.ones(33) / 32.0
    b[32] = 0.0
    b[16] += 1.0
    return signal.lfilter(b, [1.0], x)


def pt_derivative(x: np.ndarray) -> np.ndarray:
    """Paper eq. (5): five-point derivative, scaled by 1/8 (T omitted, constant gain)."""
    b = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / 8.0
    return signal.lfilter(b, [1.0], x)


def pt_integrate(x: np.ndarray, width: int = MWI_WIDTH) -> np.ndarray:
    """Paper eq. (7): causal moving-window integration over ``width`` samples."""
    return signal.lfilter(np.ones(width) / width, [1.0], x)


@dataclass
class PanTompkinsTrace:
    """Intermediate signals, kept for plotting and teaching."""

    fs: float
    bandpassed: np.ndarray
    derivative: np.ndarray
    squared: np.ndarray
    integrated: np.ndarray
    qrs_200: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    thresholds: list[tuple[int, float, float]] = field(default_factory=list)


def _stages(x200: np.ndarray) -> tuple[np.ndarray, ...]:
    lp = pt_lowpass(x200)
    bp = pt_highpass(lp)
    der = pt_derivative(bp)
    sq = der**2
    mwi = pt_integrate(sq)
    return bp, der, sq, mwi


def _decide(bp: np.ndarray, mwi: np.ndarray, der: np.ndarray) -> tuple[np.ndarray, list]:
    """Adaptive-threshold decision logic (paper section III.B to III.D).

    Returns QRS locations as indices into the band-passed signal (200 Hz, still delayed
    by the low-pass and high-pass filters) and a threshold history for plotting.
    """
    # Candidate fiducial marks: local maxima of the integrated signal at least 200 ms
    # apart (the refractory period of the paper).
    peaks, _ = signal.find_peaks(mwi, distance=REFRACTORY)
    # Ignore the start-up transient of the causal filters (low-pass 10, high-pass 32,
    # derivative 4 and integrator 30 samples). A false "beat" there creates a short first
    # RR interval, which then triggers search-back on the T waves that follow.
    peaks = peaks[peaks >= WARMUP]
    if len(peaks) == 0:
        return np.array([], dtype=int), []

    abs_bp = np.abs(bp)
    abs_der = np.abs(der)

    # Learning phase 1: initialise thresholds from the first 2 s.
    init = mwi[: int(2 * FS_PT)]
    init_bp = abs_bp[: int(2 * FS_PT)]
    spki = 0.25 * float(init.max()) if init.size else 0.0
    npki = 0.5 * float(init.mean()) if init.size else 0.0
    spkf = 0.25 * float(init_bp.max()) if init_bp.size else 0.0
    npkf = 0.5 * float(init_bp.mean()) if init_bp.size else 0.0

    irregular = False  # paper section III.C: halve THRESHOLD I1 and F1 for irregular RR

    def thr_i() -> float:
        t = npki + 0.25 * (spki - npki)
        return 0.5 * t if irregular else t

    def thr_f() -> float:
        t = npkf + 0.25 * (spkf - npkf)
        return 0.5 * t if irregular else t

    qrs_mwi: list[int] = []  # positions of accepted MWI peaks
    qrs_bp: list[int] = []  # matching band-pass peak positions (QRS locations)
    qrs_slope: list[float] = []
    rr_recent: list[int] = []  # last 8 RR intervals
    rr_selected: list[int] = []  # last 8 RR intervals within the 92 to 116 percent limits
    history: list[tuple[int, float, float]] = []
    noise_peaks: list[int] = []  # candidate peaks rejected as noise since the last QRS

    def bp_peak_near(p: int) -> int:
        lo = max(0, p - MWI_WIDTH - 5)
        return lo + int(np.argmax(abs_bp[lo : p + 1]))

    def slope_near(p: int) -> float:
        lo = max(0, p - MWI_WIDTH)
        return float(abs_der[lo : p + 1].max())

    def looks_like_t_wave(p: int) -> bool:
        """Paper III.D: within 360 ms of the last QRS and under half its maximal slope."""
        return bool(qrs_mwi) and (p - qrs_mwi[-1]) < T_WAVE_WINDOW and (
            slope_near(p) < 0.5 * qrs_slope[-1]
        )

    def accept(p: int, q: int, peaki: float, peakf: float, searchback: bool) -> None:
        nonlocal spki, spkf, irregular
        if searchback:
            spki = 0.25 * peaki + 0.75 * spki
            spkf = 0.25 * peakf + 0.75 * spkf
        else:
            spki = 0.125 * peaki + 0.875 * spki
            spkf = 0.125 * peakf + 0.875 * spkf
        if qrs_mwi:
            rr = p - qrs_mwi[-1]
            rr_recent.append(rr)
            del rr_recent[:-8]
            if rr_selected:
                avg2 = float(np.mean(rr_selected))
                if 0.92 * avg2 < rr < 1.16 * avg2:
                    rr_selected.append(rr)
            else:
                rr_selected.append(rr)
            del rr_selected[:-8]
            if len(rr_recent) >= 8 and rr_selected:
                avg1, avg2 = float(np.mean(rr_recent)), float(np.mean(rr_selected))
                irregular = not (0.92 * avg2 < avg1 < 1.16 * avg2)
        qrs_mwi.append(p)
        qrs_bp.append(q)
        qrs_slope.append(slope_near(p))

    for p in peaks:
        peaki = float(mwi[p])
        q = bp_peak_near(p)
        peakf = float(abs_bp[q])

        # Search-back: if no QRS for 166 percent of the regular RR average, the largest
        # noise peak since the last QRS that clears the lower thresholds becomes a QRS.
        # T-wave candidates are excluded here too, otherwise a slow rhythm (long RR) makes
        # search-back pick up the T wave of every beat.
        if qrs_mwi and rr_selected:
            rr_avg2 = float(np.mean(rr_selected))
            if p - qrs_mwi[-1] > 1.66 * rr_avg2 and noise_peaks:
                cands = [c for c in noise_peaks
                         if c - qrs_mwi[-1] > REFRACTORY and not looks_like_t_wave(c)]
                if cands:
                    best = max(cands, key=lambda c: mwi[c])
                    qb = bp_peak_near(best)
                    if mwi[best] > 0.5 * thr_i() and abs_bp[qb] > 0.5 * thr_f():
                        accept(best, qb, float(mwi[best]), float(abs_bp[qb]), True)
                noise_peaks = []

        is_qrs = peaki > thr_i() and peakf > thr_f()

        # T-wave discrimination: a candidate within 360 ms of the previous QRS whose
        # maximal slope is under half that of the previous QRS is taken as a T wave.
        if is_qrs and looks_like_t_wave(p):
            is_qrs = False

        if is_qrs:
            accept(p, q, peaki, peakf, False)
            noise_peaks = []
        else:
            npki = 0.125 * peaki + 0.875 * npki
            npkf = 0.125 * peakf + 0.875 * npkf
            noise_peaks.append(int(p))

        history.append((int(p), thr_i(), thr_f()))

    return np.asarray(qrs_bp, dtype=int), history


def pan_tompkins(
    x: np.ndarray, fs: float, return_trace: bool = False
) -> np.ndarray | tuple[np.ndarray, PanTompkinsTrace]:
    """Detect QRS complexes. Returns sample indices at the input sampling rate ``fs``."""
    x = np.asarray(x, dtype=float)
    x200 = resample(x, fs, FS_PT)
    bp, der, sq, mwi = _stages(x200)
    qrs_bp, history = _decide(bp, mwi, der)

    # Undo the low-pass and high-pass delays, map to the input rate, then refine to the
    # largest absolute deflection of a zero-phase 5 to 25 Hz band-pass within +-60 ms.
    qrs_200 = qrs_bp - (LP_DELAY + HP_DELAY)
    qrs_200 = qrs_200[qrs_200 >= 0]
    qrs = np.round(qrs_200 * fs / FS_PT).astype(int)
    qrs = refine_peaks(x, fs, qrs, half_window_s=0.06)

    if return_trace:
        trace = PanTompkinsTrace(FS_PT, bp, der, sq, mwi, qrs_200, history)
        return qrs, trace
    return qrs


def refine_peaks(x: np.ndarray, fs: float, idx: np.ndarray, half_window_s: float) -> np.ndarray:
    """Move each detection to the largest absolute deflection of a 5 to 25 Hz band-pass."""
    if len(idx) == 0:
        return np.asarray(idx, dtype=int)
    hi = min(25.0, 0.45 * fs)
    ref = np.abs(bandpass(x, fs, 5.0, hi, order=2)) if len(x) > 30 else np.abs(x)
    w = max(1, int(round(half_window_s * fs)))
    out = np.empty(len(idx), dtype=int)
    for i, k in enumerate(idx):
        lo, up = max(0, k - w), min(len(x), k + w + 1)
        out[i] = lo + int(np.argmax(ref[lo:up])) if up > lo else k
    out = np.unique(out)
    # Two detections refined onto peaks closer than 200 ms: keep the first.
    if len(out) > 1:
        keep = np.concatenate([[True], np.diff(out) > int(0.2 * fs)])
        out = out[keep]
    return out
