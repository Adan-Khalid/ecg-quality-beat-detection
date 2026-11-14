import numpy as np
from scipy import signal

from ecgq.detectors.pan_tompkins import pt_derivative, pt_highpass, pt_lowpass
from ecgq.filters import highpass_baseline, notch_powerline, preprocess, resample
from ecgq.synthetic import synthetic_ecg

FS = 250.0


def _tone_gain(filt, f, fs=FS, n=20000):
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f * t)
    y = filt(x)
    core = slice(n // 4, 3 * n // 4)
    return np.std(y[core]) / np.std(x[core])


def test_highpass_removes_baseline_keeps_qrs_band():
    assert _tone_gain(lambda x: highpass_baseline(x, FS), 0.1) < 0.05
    assert _tone_gain(lambda x: highpass_baseline(x, FS), 10.0) > 0.99


def test_notch_removes_50hz_only():
    assert _tone_gain(lambda x: notch_powerline(x, FS), 50.0) < 0.01
    assert _tone_gain(lambda x: notch_powerline(x, FS), 20.0) > 0.98


def test_preprocess_improves_snr_on_contaminated_ecg():
    clean, _ = synthetic_ecg(60, FS, noise_std=0.0, seed=1)
    dirty, _ = synthetic_ecg(60, FS, noise_std=0.0, baseline_amp=0.8, mains_amp=0.3, seed=1)
    err_before = np.std(dirty - clean)
    err_after = np.std(preprocess(dirty, FS) - preprocess(clean, FS))
    # 0.25 Hz drift sits close to the 0.5 Hz cut-off, so about 93 percent is removed.
    assert err_after < 0.1 * err_before


def test_resample_preserves_length_ratio():
    x = np.random.default_rng(0).standard_normal(3600)
    assert len(resample(x, 360, 250)) == 2500


def test_pt_lowpass_matches_paper_recursive_form():
    # The FIR form must equal the paper's recursive difference equation.
    x = np.random.default_rng(0).standard_normal(500)
    b = np.zeros(13)
    b[0], b[6], b[12] = 1, -2, 1
    ref = signal.lfilter(b, [1, -2, 1], x)
    np.testing.assert_allclose(pt_lowpass(x), ref, atol=1e-9)


def test_pt_highpass_matches_paper_recursive_form():
    x = np.random.default_rng(1).standard_normal(500)
    b = np.zeros(33)
    b[0], b[16], b[17], b[32] = -1 / 32, 1, -1, 1 / 32
    ref = signal.lfilter(b, [1, -1], x)
    np.testing.assert_allclose(pt_highpass(x), ref, atol=1e-9)


def test_pt_bandpass_passband_centre():
    # The cascade peaks near 10 to 17 Hz (paper: 5 to 15 Hz pass band at 200 Hz).
    w, h = signal.freqz(np.convolve(np.convolve(np.ones(6), np.ones(6)),
                                    np.r_[-np.ones(16), 32 - 1, -np.ones(15)] / 32),
                        worN=4096, fs=200)
    peak_f = w[np.argmax(np.abs(h))]
    assert 5 <= peak_f <= 20


def test_derivative_of_ramp_is_constant():
    x = np.arange(100, dtype=float)
    d = pt_derivative(x)
    np.testing.assert_allclose(d[10:], 10 / 8)
