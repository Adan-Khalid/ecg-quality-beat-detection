# Decisions

One entry per design decision: what was chosen, why, and what was rejected.

## D1. Pan-Tompkins filters in FIR form

The paper writes the low-pass and high-pass filters as recursive difference equations with poles on the unit circle that cancel zeros. Over a 24-hour recording (17 million samples at 200 Hz) the recursion accumulates floating-point round-off. The identical transfer functions are applied in non-recursive (FIR) form instead. A unit test checks that both forms give the same output.

## D2. Pan-Tompkins start-up transient and search-back

The first full-database run gave 1,460 false positives on record 117 alone. Cause: a false "beat" in the first 0.4 s (the causal filters' warm-up) created a short first RR interval. Every normal beat then looked late, search-back fired, and it accepted T waves because the T-wave rule was only applied to normal detections. Fix: ignore peaks during the warm-up and apply the T-wave slope test to search-back candidates too, as the paper's logic implies. Record 117 then had zero false positives. The summary of the run before the fix is kept in `reports/metrics/history/qrs_mitdb_summary_before_ptfix.json` (Pan-Tompkins +P 0.9743 with 2,875 false positives, before the flutter exclusion of D3).

## D3. EC57 scoring conventions

Ventricular flutter episodes (record 207) contain no beat labels, so any detection there would count as a false positive. ANSI/AAMI EC57 excludes them; so does this project. The "ec57" protocol also skips the first 5 minutes (the default learning period of WFDB `bxb`). The "whole_record" protocol is reported next to it.

## D4. Elgendi parameters

The 2010 paper defines the algorithm; the 2013 PLoS ONE paper tunes it (W1 = 97 ms, W2 = 611 ms, offset beta = 0.08, 8 to 20 Hz). This project uses the 2013 values and cites both.

## D5. Detectors run in 60 s blocks on long recordings

The Elgendi offset uses the mean energy of the whole input, and Pan-Tompkins thresholds adapt slowly. On a 24-hour recording one burst of motion artefact would raise thresholds for a long time. On BUT QDB both detectors therefore run on 60 s blocks with 5 s of overlap, keeping only beats from each block's centre. On MIT-BIH (30 min, mostly clean) they run on the whole record, as in the papers.

## D6. 10 s non-overlapping windows, pure labels only

10 s is long enough for 8 to 20 beats (needed for RR and template features) and short enough to localise artefact. Windows overlap nothing, so no two windows share samples. A window is used for training only when all its samples carry the same consensus class; mixed windows (at class boundaries) are still scored at sample level on the official split.

## D7. Scale-invariant ECG features

BUT QDB is in microvolts, MIT-BIH in millivolts, and electrode contact changes amplitude between people. Every ECG feature is a ratio, a correlation, a shape statistic or a count, so the same model can score both databases.

## D8. Class and record weighting

Three records are annotated for 24 h or more and fifteen for about 40 minutes, and class 1 dominates. Sample weights are 1 / (windows of that class) times 1 / sqrt(windows of that record), so rare classes count and the long records do not decide everything. The square root keeps some of the extra information in the long records.

## D9. Leave-one-subject-out as the main protocol

Published BUT QDB results often split windows at random, which puts the same person in training and test and inflates scores. This project holds out whole subjects (all of their records). The official fixed split from the dataset authors is reported too, for comparison with their baseline.

## D10. Rule-based baseline fitted per fold

Fixed literature thresholds (for example bSQI 0.9) were rejected because the thresholds depend on the detectors and the device. A grid search on the training subjects of each fold keeps the baseline fair and still interpretable: four thresholds.

## D11. Small CPU CNN

This laptop has no CUDA GPU (Intel UHD graphics, i7-10610U). The CNN has four conv blocks and about 35k parameters, takes 10 s at 100 Hz, and trains for 6 epochs of 8,000 weighted samples (D15). Augmentation: ECG polarity flip (lead placement differs between people), amplitude scaling and small noise.

## D12. Quality gate test on NSTDB, not BUT QDB

BUT QDB has no beat labels, so "how much does rejecting class 3 help beat detection" cannot be measured there. NSTDB adds known noise to MIT-BIH records that do have expert beat labels. The models are trained only on BUT QDB, so this also tests transfer to another device and lead.

## D13. MATLAB cross-check not run

The brief lists MATLAB for cross-checking the filter bench. MATLAB is not installed on this machine, so the cross-check uses Python only: the FIR form is tested against the paper's recursive form with `scipy.signal.lfilter`, and the frequency responses are plotted in `reports/figures/filter_responses.png`.

## D14. Only the dataset authors' annotated ranges are used

The consensus CSV of record 105001 labels its first 13.1 hours as class 3. In that stretch the accelerometer is static and the ECG is pure noise (see `reports/figures/day_quality_105001.png`), so the patch was apparently not on the body. Those hours held 4,680 of the 5,224 pure class 3 windows, which made class 3 look easy (pooled class 3 F1 0.96). The dataset authors' own benchmark (`choseDATA('all')` in `TESTrun_v01.m`, BUT-QDB-supplement repository) scores 105001 only from sample 46,800,001, and lists the annotated ranges of every record. This project now uses exactly those ranges everywhere (`ecgq.io.OFFICIAL_RANGES`): 26,880 labelled windows, of which 544 are class 3. The superseded results are kept in `reports/metrics/history/*_before_official_ranges.json`. Lesson: plot the whole recording before trusting a label count.

## D15. CNN compute budget

Each CNN is trained 15 times (once per held-out subject) on a 4-core laptop CPU. Training uses 6 epochs of 8,000 weighted samples (batch 128), fixed before seeing any result, so a full CNN evaluation runs in about an hour. More epochs might help the CNN; that is listed as a next step, not tuned on the test folds.

## D16. Noise location in NSTDB from the per-window standard deviation

NSTDB records are MIT-BIH 118 and 119 plus calibrated noise, but the stored signals also differ from the originals by a DC offset (about 5 mV) that changes between noise blocks. The oracle gate uses the per-window standard deviation of (NSTDB minus clean), which ignores the offset; it is exactly zero outside the noise blocks.

## D17. CNN gate reported, not plotted

The ECG-only CNN rejects all of MIT-BIH record 118 (right bundle branch block), clean minutes included, because that QRS shape never occurs in BUT QDB. Its gate numbers mix that failure with real noise rejection, so they are kept in the tables with this explanation and left out of `reports/figures/rejection_nstdb.png`. The feature-based gates do not have this problem.
