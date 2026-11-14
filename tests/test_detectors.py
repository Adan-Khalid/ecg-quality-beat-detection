import numpy as np
import pytest

from ecgq.detectors import elgendi, pan_tompkins
from ecgq.metrics import match_beats
from ecgq.synthetic import motion_burst, synthetic_ecg


@pytest.mark.parametrize("detector", [pan_tompkins, elgendi])
@pytest.mark.parametrize("fs", [250.0, 360.0, 1000.0])
def test_detects_clean_synthetic_beats(detector, fs):
    x, r = synthetic_ecg(120, fs, hr_bpm=72, noise_std=0.02, seed=3)
    det = detector(x, fs)
    m, _ = match_beats(r, det, int(0.15 * fs))
    assert m.sensitivity > 0.98
    assert m.ppv > 0.98


@pytest.mark.parametrize("detector", [pan_tompkins, elgendi])
@pytest.mark.parametrize("hr", [45, 120, 170])
def test_heart_rate_range(detector, hr):
    fs = 250.0
    x, r = synthetic_ecg(90, fs, hr_bpm=hr, hrv_std_s=0.01, noise_std=0.02, seed=4)
    det = detector(x, fs)
    m, _ = match_beats(r, det, int(0.15 * fs))
    assert m.sensitivity > 0.97 and m.ppv > 0.97


def test_detection_location_is_close_to_r_peak():
    fs = 360.0
    x, r = synthetic_ecg(60, fs, seed=5)
    det = pan_tompkins(x, fs)
    m, found = match_beats(r, det, int(0.15 * fs))
    # Located within 20 ms of the true R peak for matched beats.
    near, _ = match_beats(r, det, int(0.02 * fs))
    assert near.tp >= 0.97 * m.tp


def test_inverted_polarity():
    fs = 250.0
    x, r = synthetic_ecg(60, fs, seed=6)
    for det in (pan_tompkins(-x, fs), elgendi(-x, fs)):
        m, _ = match_beats(r, det, int(0.15 * fs))
        assert m.sensitivity > 0.97


def test_detectors_disagree_more_under_motion_noise():
    from ecgq.metrics import detector_agreement

    fs = 250.0
    x, _ = synthetic_ecg(60, fs, seed=7)
    tol = int(0.15 * fs)
    clean = detector_agreement(pan_tompkins(x, fs), elgendi(x, fs), tol)
    noisy_x = x + motion_burst(len(x), fs, 0, 60, amp=1.5, seed=2)
    noisy = detector_agreement(pan_tompkins(noisy_x, fs), elgendi(noisy_x, fs), tol)
    assert clean > 0.95
    assert noisy < clean


def test_empty_and_short_input():
    assert len(elgendi(np.zeros(10), 250.0)) == 0
    assert len(pan_tompkins(np.zeros(1000), 250.0)) <= 1
