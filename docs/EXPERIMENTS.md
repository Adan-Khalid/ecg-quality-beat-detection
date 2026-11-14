# Experiments

Every run that produced a number, in order. Config files are in `configs/`, seeds are fixed (0), logs are in `reports/logs/`, metrics in `reports/metrics/`. Machine: Intel i7-10610U (4 cores), 15.7 GB RAM, no CUDA GPU, Windows 11, Python 3.12.10.

## E1. QRS detectors on MIT-BIH, first run (superseded)

- Config: `configs/qrs_mitdb.yaml` (channel 0, 150 ms window), whole record, no exclusions.
- Result (gross, 48 records): Pan-Tompkins Se 0.9970, +P 0.9743, 2,875 false positives; Elgendi Se 0.9982, +P 0.9972; XQRS Se 0.9920, +P 0.9930.
- Source: `reports/metrics/history/qrs_mitdb_summary_before_ptfix.json`.
- Finding: 1,460 Pan-Tompkins false positives on record 117 alone, and 308 on 119. Traced to search-back accepting T waves after a start-up artefact (DECISIONS D2).

## E2. QRS detectors after the Pan-Tompkins fix, flutter excluded (superseded by E3)

- Changes: ignore peaks during filter warm-up; T-wave test applied to search-back candidates; ventricular flutter episodes excluded (EC57).
- Result (gross, 48 records, whole record): Pan-Tompkins Se 0.9968, +P 0.9940 (656 FP, 346 FN). Records 117 and 119: zero errors.

## E3. QRS detectors, final (EC57 protocol and whole record)

- Config: `configs/qrs_mitdb.yaml` with `learning_period_s: 300`.
- Command: `python scripts/evaluate_qrs.py`
- Results: `reports/metrics/qrs_mitdb_summary.json`, per record in `reports/metrics/qrs_mitdb_per_record.csv`.

| Detector | Protocol | Se | +P | F1 | Timing MAE | Run time, 48 records |
| --- | --- | --- | --- | --- | --- | --- |
| Pan-Tompkins (this repo) | EC57 | 0.9968 | 0.9947 | 0.9958 | 2.7 ms | 29.9 s |
| Elgendi (this repo) | EC57 | 0.9982 | 0.9989 | 0.9986 | 4.9 ms | 4.2 s |
| WFDB XQRS (library) | EC57 | 0.9935 | 0.9974 | 0.9955 | 4.0 ms | 343.4 s |
| Pan-Tompkins (this repo) | whole record | 0.9968 | 0.9940 | 0.9954 | 2.7 ms | |
| Elgendi (this repo) | whole record | 0.9982 | 0.9988 | 0.9985 | 4.9 ms | |
| WFDB XQRS (library) | whole record | 0.9920 | 0.9965 | 0.9942 | 3.9 ms | |

- Hardest records for Pan-Tompkins: 108 (large P waves, noise) and 114. See `reports/figures/qrs_failure_108.png`.
- For reference, the 1985 paper's abstract reports 99.3 % of QRS complexes correctly detected on the MIT-BIH database (quoted from the abstract, not reproduced; its protocol differs).
