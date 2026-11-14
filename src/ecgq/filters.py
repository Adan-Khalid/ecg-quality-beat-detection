"""Preprocessing filters for single-lead ECG.

All filters here are zero-phase (forward-backward) because they are used offline for
analysis and feature extraction. The causal Pan-Tompkins filters live in
``ecgq.detectors.pan_tompkins`` because that algorithm depends on their exact delays.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal


def highpass_baseline(x: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 2) -> np.ndarray:
    """Remove baseline wander with a zero-phase Butterworth high-pass filter.

    A 0.5 Hz cut-off removes breathing and movement drift while keeping the ST segment
    and T wave (the slowest diagnostic content sits around 0.67 Hz, a heart rate of 40 bpm).
    """
    sos = signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def notch_powerline(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove mains interference (50 Hz in Pakistan and Europe) with a zero-phase IIR notch.

    Q = 30 gives a -3 dB bandwidth of f0 / Q, about 1.7 Hz, narrow enough to leave the QRS
    spectrum (roughly 5 to 30 Hz) untouched. Harmonics up to Nyquist are also removed.
    """
    y = np.asarray(x, dtype=float)
    k = 1
    while k * f0 < fs / 2 - 1:
        b, a = signal.iirnotch(k * f0, q, fs=fs)
        y = signal.filtfilt(b, a, y, axis=-1)
        k += 1
    return y


def lowpass(x: np.ndarray, fs: float, cutoff: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (used before resampling and for display)."""
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Rational-factor polyphase resampling (anti-aliasing filter included)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, frac.numerator, frac.denominator, axis=-1)


def preprocess(x: np.ndarray, fs: float, mains: float | None = 50.0) -> np.ndarray:
    """Standard front end: baseline high-pass, optional mains notch, 40 Hz low-pass."""
    y = highpass_baseline(x, fs)
    if mains is not None and mains < fs / 2:
        y = notch_powerline(y, fs, f0=mains)
    if fs > 100:
        y = lowpass(y, fs, cutoff=min(40.0, 0.45 * fs))
    return y
