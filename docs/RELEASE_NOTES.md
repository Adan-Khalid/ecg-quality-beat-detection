# v1.0.0: ECG signal quality and beat detection

First release. Numbers below are copied from `reports/results_tables.md`.

## What it does

- Filters single-lead ECG (0.5 Hz high-pass, 50 Hz notch, 40 Hz low-pass) and detects QRS complexes with two detectors written from their papers: Pan-Tompkins (1985) and Elgendi's two moving averages.
- Labels every 10 s of wearable ECG as class 1 (clean), class 2 (QRS only) or class 3 (unusable) with 17 ECG quality indices, optional accelerometer features, and three model types: rule-based thresholds, gradient boosting and a small 1D CNN.
- Uses the quality label as a gate before beat detection.
- Includes a Streamlit explorer, a day-long quality figure and a demo GIF, all from models that never saw the person shown.

## Key results

- QRS detection on all 48 MIT-BIH records (EC57 protocol, 150 ms window): Pan-Tompkins Se 99.68 %, +P 99.47 %; Elgendi Se 99.82 %, +P 99.89 %; WFDB XQRS library baseline Se 99.35 %, +P 99.74 %.
- Quality classification, leave-one-subject-out over 15 BUT QDB subjects: macro-F1 0.771 for gradient boosting with ECG and accelerometer features, against 0.634 for the rule-based baseline.
- Official BUT QDB test split, sample-level accuracy: 87.28 %, against 74.08 % reported for the SNR method in the dataset paper.
- Gate on the MIT-BIH noise stress test at 0 dB: Pan-Tompkins F1 0.910 with the gate (72 % of beats kept) against 0.814 without; no clean window rejected.
- 1D CNN (about 35k parameters, CPU): macro-F1 0.736 (ECG) and 0.723 (ECG + accelerometer); best mean per-subject score (0.860) but weakest on class 3.
- Candidate research question (does the accelerometer help on unseen subjects?): no consistent benefit. Gradient boosting gains +0.019 pooled macro-F1 (95 % CI +0.002 to +0.037) with 9 of 15 subjects better (Wilcoxon p = 0.23); the CNN gets worse for 10 of 15 subjects (p = 0.048).

## Known limitations

- 15 subjects, one device, one lead; class 3 has only 544 labelled windows.
- The ECG-only CNN confuses an unfamiliar arrhythmia (right bundle branch block) with noise.
- Research and education only, not for diagnosis.

## How to run

```bash
uv sync
uv run pytest
uv run python scripts/download_data.py --datasets mitdb nstdb
uv run python scripts/evaluate_qrs.py
```

The full pipeline (BUT QDB download, feature extraction, all evaluations and figures) is in the README Quickstart.
