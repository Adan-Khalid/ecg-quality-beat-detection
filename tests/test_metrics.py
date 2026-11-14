import numpy as np

from ecgq.metrics import detector_agreement, match_beats, quality_report


def test_match_beats_counts():
    ref = np.array([100, 300, 500, 700])
    det = np.array([102, 290, 650, 900])  # 650 is 50 away from 700
    m, found = match_beats(ref, det, tol=60)
    assert (m.tp, m.fp, m.fn) == (3, 1, 1)
    assert found.tolist() == [True, True, False, True]


def test_match_is_one_to_one():
    m, _ = match_beats(np.array([100, 110]), np.array([105]), tol=20)
    assert (m.tp, m.fp, m.fn) == (1, 0, 1)


def test_matches_wfdb_compare_annotations():
    import wfdb.processing as wp

    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.integers(200, 400, 300))
    det = ref + rng.integers(-40, 40, len(ref))
    det = np.sort(np.concatenate([det[rng.random(len(det)) > 0.05], rng.integers(0, ref[-1], 10)]))
    det = np.unique(det)
    ours, _ = match_beats(ref, det, tol=54)
    cmp = wp.compare_annotations(ref, det, window_width=54)
    assert ours.tp == cmp.tp
    assert ours.fn == cmp.fn
    assert ours.fp == cmp.fp


def test_agreement_bounds():
    a = np.array([100, 300, 500])
    assert detector_agreement(a, a, 10) == 1.0
    assert detector_agreement(a, np.array([1000, 1200]), 10) == 0.0
    assert detector_agreement(np.array([]), np.array([]), 10) == 1.0


def test_quality_report_perfect():
    y = np.array([1, 2, 3, 1, 2, 3])
    rep = quality_report(y, y)
    assert rep["macro_f1"] == 1.0
    assert rep["confusion"][0][0] == 2
