import numpy as np
import pytest

from ecgq.quality.features import ACC_FEATURES, ECG_FEATURES, acc_features
from ecgq.quality.models import RuleClassifier, balanced_weights, build_cnn, make_gbm
from ecgq.quality.windows import cnn_inputs, compute_windows, run_blockwise, window_labels
from ecgq.synthetic import motion_burst, synthetic_ecg

FS = 250.0


@pytest.fixture(scope="module")
def mixed_recording():
    """3 minutes: clean, then 60 s of heavy motion artefact, then clean again."""
    x, r = synthetic_ecg(180, FS, hr_bpm=75, noise_std=0.02, seed=11)
    x = x + motion_burst(len(x), FS, start_s=60, dur_s=60, amp=2.0, seed=3)
    labels = np.ones(len(x), dtype=np.int8)
    labels[int(60 * FS) : int(120 * FS)] = 3
    rng = np.random.default_rng(0)
    acc = rng.normal(0, 5, (int(180 * 100), 3)) + np.array([0, 0, 1000.0])
    acc[6000:12000] += rng.normal(0, 300, (6000, 3))  # motion during the artefact
    return x, r, labels, acc


def test_window_features_shapes(mixed_recording):
    x, _, labels, acc = mixed_recording
    ws = compute_windows(x, FS, labels=labels, acc=acc, fs_acc=100.0, mains=50.0)
    assert ws.ecg_feats.shape == (18, len(ECG_FEATURES))
    assert ws.acc_feats.shape == (18, len(ACC_FEATURES))
    assert np.all(np.isfinite(ws.ecg_feats))
    assert ws.label_counts.sum() == 18 * 10 * FS
    assert set(ws.labels.tolist()) == {1, 3}


def test_sqis_separate_clean_from_artefact(mixed_recording):
    x, _, labels, acc = mixed_recording
    ws = compute_windows(x, FS, labels=labels, acc=acc, fs_acc=100.0)
    f = {k: ws.ecg_feats[:, i] for i, k in enumerate(ECG_FEATURES)}
    clean, noisy = ws.labels == 1, ws.labels == 3
    assert f["bsqi"][clean].mean() > 0.95
    assert f["bsqi"][noisy].mean() < f["bsqi"][clean].mean() - 0.2
    assert f["tsqi"][clean].mean() > f["tsqi"][noisy].mean()
    a = {k: ws.acc_feats[:, i] for i, k in enumerate(ACC_FEATURES)}
    assert a["acc_mag_std"][noisy].mean() > 5 * a["acc_mag_std"][clean].mean()


def test_window_labels_require_purity():
    lab = np.array([1] * 10 + [2] * 5 + [1] * 5 + [0] * 10 + [3] * 10)
    out = window_labels(lab, np.array([0, 10, 20, 30]), 10)
    assert out.tolist() == [1, 0, 0, 3]


def test_blockwise_detection_has_no_duplicates():
    from ecgq.detectors import pan_tompkins

    x, r = synthetic_ecg(150, FS, seed=2)
    det = run_blockwise(pan_tompkins, x, FS, block_s=30, overlap_s=5)
    assert np.all(np.diff(det) > 0.25 * FS)
    assert abs(len(det) - len(r)) <= 2


def test_acc_features_static_vs_moving():
    rng = np.random.default_rng(1)
    still = rng.normal(0, 2, (1, 1000, 3)) + [0, 0, 1000]
    moving = still + rng.normal(0, 200, (1, 1000, 3))
    f = acc_features(np.concatenate([still, moving]), 100.0)
    assert f.shape == (2, len(ACC_FEATURES))
    assert f[1, 0] > 10 * f[0, 0]


def test_balanced_weights_equalise_classes():
    y = np.array([1] * 90 + [3] * 10)
    g = np.array(["a"] * 50 + ["b"] * 50)
    w = balanced_weights(y, g)
    assert np.isclose(w[y == 1].sum(), w[y == 3].sum(), rtol=0.2)
    assert np.isclose(w.sum(), len(y))


def test_rule_classifier_learns_thresholds():
    rng = np.random.default_rng(0)
    n = 300
    X = np.zeros((n, len(ECG_FEATURES)))
    y = rng.integers(1, 4, n)
    i = {k: ECG_FEATURES.index(k) for k in ("bsqi", "tsqi", "bassqi", "hr_pt")}
    X[:, i["hr_pt"]] = 70
    X[:, i["bsqi"]] = np.where(y == 3, 0.3, 0.97)
    X[:, i["tsqi"]] = np.where(y == 1, 0.97, 0.6)
    X[:, i["bassqi"]] = 0.99
    pred = RuleClassifier().fit(X, y).predict(X)
    assert np.mean(pred == y) == 1.0


def test_gbm_end_to_end(mixed_recording):
    x, _, labels, acc = mixed_recording
    ws = compute_windows(x, FS, labels=labels, acc=acc, fs_acc=100.0)
    clf = make_gbm().set_params(max_iter=20, min_samples_leaf=2).fit(ws.ecg_feats, ws.labels)
    assert set(clf.predict(ws.ecg_feats)) <= {1, 3}


def test_cnn_shapes_and_inputs(mixed_recording):
    torch = pytest.importorskip("torch")
    x, _, _, acc = mixed_recording
    starts = np.arange(0, int(170 * FS), int(10 * FS))
    xin = cnn_inputs(x, FS, starts, acc=acc, fs_acc=100.0)
    assert xin.shape == (len(starts), 4, 1000)
    assert np.abs(xin).max() <= 20
    net = build_cnn(4)
    out = net(torch.as_tensor(xin[:5]))
    assert tuple(out.shape) == (5, 3)
    assert sum(p.numel() for p in net.parameters()) < 80_000
