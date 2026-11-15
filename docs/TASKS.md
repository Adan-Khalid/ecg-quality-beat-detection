# Task list

What was done, roughly in order, and what is still open.

## Research
- [x] Check the three PhysioNet datasets: version, DOI, licence, size and citation (`DATA_CARD.md`)
- [x] Literature search: QRS detectors, signal quality indices, BUT QDB papers, standards, tools, products (`RESEARCH_LOG.md`)
- [x] Check whether the accelerometer question has already been answered (it appears open)
- [x] Find the dataset authors' official train/test split

## Design
- [x] Pipeline diagram, maths, baselines, metrics, split strategy and risks (`PROJECT_PLAN.md`)
- [x] Repository layout, pinned environment (uv, Python 3.12, CPU PyTorch), ruff, pytest, CI

## Implementation
- [x] Resumable, checksum-verified download script
- [x] Zero-phase high-pass, notch, low-pass and resampling, with frequency tests
- [x] Pan-Tompkins from the paper: filters, derivative, squaring, integration, thresholds, search-back, T-wave rejection
- [x] Elgendi two-moving-average detector
- [x] EC57 beat matching, cross-checked against WFDB `compare_annotations`
- [x] BUT QDB reader with chunked resampling and consensus labels
- [x] 17 ECG quality indices and 7 accelerometer features per 10 s window
- [x] Rule-based classifier with per-fold threshold search
- [x] Gradient boosting with class and record weighting
- [x] 1D CNN (ECG, and ECG + accelerometer)
- [x] Synthetic ECG generator and tests for every stage

## Evaluation
- [x] Detectors on all 48 MIT-BIH records (EC57 and whole-record protocols)
- [x] Find and fix the Pan-Tompkins search-back bug (record 117)
- [x] Features for all 18 BUT QDB records
- [x] Leave-one-subject-out evaluation of all six quality models
- [x] Accelerometer question: paired test and bootstrap interval
- [x] Official split, sample-level scores
- [x] Restrict labels to the authors' annotated ranges (105001 off-body stretch, DECISIONS D14)
- [x] Quality gate on the noise stress test database

## Demo
- [x] Held-out predictions for every window of every recording
- [x] Day-long figure, GIF and MP4
- [x] Streamlit explorer and screenshots

## Release
- [x] README with every number read from `reports/`
- [x] Social preview image, CITATION.cff, release notes
- [x] Fresh-environment install and test run
- [x] Public repository with CI on every push
- [x] Release v1.0.0

## Open
- [ ] Tune the CNN within the training folds
- [ ] Add arrhythmia examples so unusual rhythms are not mistaken for noise
- [ ] Test on a wrist ECG device
- [ ] Port the rule-based gate to fixed-point C for a microcontroller
