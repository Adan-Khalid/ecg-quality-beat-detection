# Tasks

Small, ordered tasks. Ticked items are done and committed.

## Phase 1: research
- [x] Open the three PhysioNet pages; record version, DOI, licence, size and citation in `DATA_CARD.md`
- [x] Literature search: detectors, SQIs, BUT QDB papers, standards, tools, products (`RESEARCH_LOG.md`)
- [x] Check the candidate research question against the literature (appears open)
- [x] Find the official BUT QDB train/test split (dataset authors' supplement repository)

## Phase 2: architecture
- [x] `PROJECT_PLAN.md` with pipeline diagram, maths, baselines, metrics, split strategy, risks
- [x] Repo layout, pinned environment (uv, Python 3.12, CPU PyTorch), ruff, pytest, CI

## Phase 3: implementation
- [x] Resumable, checksum-verified download script
- [x] Zero-phase high-pass, notch, low-pass and resampling, with frequency tests
- [x] Pan-Tompkins from the paper: filters, derivative, squaring, integration, thresholds, search-back, T-wave rejection
- [x] Elgendi two-moving-average detector
- [x] EC57 beat matching, cross-checked against WFDB `compare_annotations`
- [x] BUT QDB reader with chunked resampling and consensus labels
- [x] 17 ECG SQIs and 7 accelerometer features per 10 s window
- [x] Rule-based classifier with per-fold threshold search
- [x] Gradient boosting with class and record weighting
- [x] 1D CNN (ECG, and ECG + accelerometer)
- [x] Synthetic ECG generator and tests for every stage

## Phase 4: validation
- [x] Detectors on all 48 MIT-BIH records (EC57 and whole-record protocols)
- [x] Find and fix the Pan-Tompkins search-back bug (record 117)
- [x] Build features for all 18 BUT QDB records
- [x] LOSO evaluation of all six quality models
- [x] Research question: paired test and bootstrap interval
- [x] Official fixed split, sample-level scores
- [x] Restrict labels to the authors' annotated ranges (105001 off-body stretch, D14)
- [x] Quality gate on the noise stress test database

## Phase 5: demo
- [x] Held-out predictions for every window of every recording
- [x] Day-long figure, hero GIF and MP4
- [x] Streamlit explorer and screenshots

## Phase 6: release (preparation only, nothing uploaded)
- [x] README from the template with numbers read from `reports/`
- [x] Social preview image, CITATION.cff, release notes
- [x] Fresh-environment install and test run

## Release steps (not done yet)
- [ ] Refresh the GitHub CLI login with the `workflow` scope, then create the repo and push
- [ ] Tag and release v1.0.0
- [ ] Upload `assets/social-preview.png` in the repo settings and pin the repo
- [ ] LinkedIn release post (`/linkedin`)
