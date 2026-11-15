# Experiments

Every run that produced a number, in order. Config files are in `configs/`, seeds are fixed (0), logs are in `reports/logs/`, metrics in `reports/metrics/`. Machine: Intel i7-10610U (4 cores), 15.7 GB RAM, no CUDA GPU, Windows 11, Python 3.12.10.

## E1. QRS detectors on MIT-BIH, first run (superseded)

- Config: `configs/qrs_mitdb.yaml` (channel 0, 150 ms window), whole record, no exclusions.
- Result (gross, 48 records): Pan-Tompkins Se 0.9970, +P 0.9743, 2,875 false positives; Elgendi Se 0.9982, +P 0.9972; XQRS Se 0.9920, +P 0.9930.
- Source: `reports/metrics/history/qrs_mitdb_summary_before_ptfix.json`.
- Finding: in the debugging session the per-record output showed 1,460 Pan-Tompkins false positives on record 117 and 308 on 119 (that per-record table was overwritten by the next run; only the summary above was kept). Traced to search-back accepting T waves after a start-up artefact (DECISIONS D2).

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

## E4. Quality classification, first LOSO run (superseded by E5)

- Labels: every consensus-labelled window, including the first 13.1 h of 105001 (31,560 windows, 5,224 class 3).
- Result: GBM ECG + ACC pooled macro-F1 0.880, class 3 F1 0.96. Rule baseline 0.754.
- Finding: 4,680 of the class 3 windows came from one stretch where the device was apparently off the body (static accelerometer, pure-noise ECG), which made class 3 trivial. The dataset authors' benchmark excludes it (DECISIONS D14).
- Source: `reports/metrics/history/quality_loso_summary_before_official_ranges.json`.

## E5. Quality classification, LOSO on the official annotated ranges (feature models)

- Config: `configs/quality_butqdb.yaml`, seed 0. Command: `python scripts/evaluate_quality.py --models rule gbm_ecg gbm_acc gbm_ecg_acc`
- Data: 26,880 windows (16,284 / 10,052 / 544 for classes 1 / 2 / 3), 15 subjects, 15 folds.
- Results (`reports/metrics/quality_loso_summary.json`, tables in `reports/results_tables.md`):

| Model | Pooled macro-F1 | Mean subject macro-F1 | F1 class 3 |
| --- | --- | --- | --- |
| Rule-based (baseline) | 0.634 | 0.675 | 0.610 |
| GBM, ACC only | 0.509 | 0.644 | 0.014 |
| GBM, ECG SQIs | 0.752 | 0.796 | 0.629 |
| GBM, ECG + ACC | 0.771 | 0.819 | 0.632 |

- Research question (GBM ECG + ACC vs GBM ECG): 9 subjects improve, 6 get worse; mean per-subject change +0.023; Wilcoxon p = 0.229; pooled macro-F1 change 95 % CI [+0.002, +0.037] from a subject bootstrap.
- Reading: a small pooled gain that is not consistent across people. The accelerometer mostly helps separate class 1 from class 2 (class 2 windows carry the most motion); expert class 3 windows have low motion (median close to class 1, `reports/figures/feature_distributions.png`), so motion alone cannot find them.

## E6. Official BUT QDB split (sample level)

- Command: `python scripts/evaluate_fixed_split.py`. Train 11 records, test 7 records of other subjects.
- Results (`reports/metrics/quality_fixed_split.json`): GBM ECG + ACC accuracy 87.28 % (F1 90.65 / 81.06 / 83.43 % for classes 1 / 2 / 3); GBM ECG 85.68 %; rule baseline 71.50 %. The SNR method in the dataset paper reports 74.08 % on the same split (quoted, not re-run).

## E7. Held-out predictions over whole recordings

- Command: `python scripts/predict_all.py` (one GBM ECG + ACC per held-out subject, scoring every window).
- Agreement with expert labels ranges from 70.7 % (125001) to 99.6 % (122001); see `reports/metrics/holdout_daylong_agreement.csv`.

## E8. Quality gate on the noise stress test database

- Config: `configs/rejection_nstdb.yaml`. Models: `scripts/train_final.py` (all 15 subjects, CNN 6 epochs x 8,000 samples). Command: `python scripts/evaluate_rejection.py`
- First run had a broken oracle: NSTDB files carry a different DC offset from MIT-BIH, so subtracting the clean record marked every window as noisy. Fixed by using the per-window standard deviation of the difference (D16); the documented schedule (noise from minute 5, 2 min on, 2 min off) is then recovered exactly.
- Results (`reports/metrics/rejection_nstdb_summary.json`): Pan-Tompkins F1 without a gate falls from 1.000 (clean) to 0.861 (6 dB), 0.814 (0 dB) and 0.743 (-6 dB). The GBM gate raises it to 0.914, 0.910 and 0.839 while keeping 82 %, 72 % and 73 % of beats; the rule gate to 0.951, 0.919 and 0.828 (72 %, 68 %, 73 % kept). Neither rejects any clean window.
- The ECG-only CNN gate rejects all of record 118 (right bundle branch block: a wide, notched QRS never seen in BUT QDB) including its clean minutes, and works as intended on record 119. It confuses unfamiliar morphology with noise.

## E9. 1D CNN, LOSO

- Config: `configs/quality_butqdb.yaml` (6 epochs x 8,000 weighted samples, batch 128, AdamW lr 1e-3, seed 0; budget fixed before results, DECISIONS D15). Command: `python scripts/evaluate_quality.py --models cnn_ecg cnn_ecg_acc` (about 4 min per fold for both models on the 4-core CPU).
- Results: CNN ECG pooled macro-F1 0.736 (mean subject 0.860, class 3 F1 0.505); CNN ECG + ACC 0.723 (0.816, 0.493).
- CNN ECG vs GBM ECG: 10 subjects better, 4 worse, Wilcoxon p = 0.198, pooled change 95 % CI [-0.053, +0.098]: no clear winner. The CNN is better on typical subjects and worse on class 3.
- CNN ECG + ACC vs CNN ECG: 5 better, 10 worse, Wilcoxon p = 0.048, pooled change 95 % CI [-0.115, +0.020]. Raw accelerometer channels hurt the CNN on unseen subjects.
- Conclusion for the candidate research question (with E5): no consistent benefit from the accelerometer on subjects not seen in training.
