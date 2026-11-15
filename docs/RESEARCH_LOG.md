# Research log

Literature log for "ECG signal quality and beat detection", compiled 2026-11-14. Every link was accessed on 2026-11-14. Scope: single-lead wearable ECG quality classification on BUT QDB, QRS detection validated on MIT-BIH, rejection tested on NSTDB, and the candidate research question on accelerometer features.

How to read the verification notes: "opened" means the page or PDF was read for this log. "Index/abstract only" means the detail comes from a bibliographic record, abstract or search result, not the full text. "Not verified" means the detail could not be checked and should not be relied on.

## Problem and users

- Problem: free-living wearable ECG is often corrupted by motion, muscle noise and electrode problems. Beat detectors and rhythm algorithms produce false beats and false alarms on these segments. A quality gate that labels each segment as usable for full-wave analysis, usable for QRS detection only, or unusable (the 3 BUT QDB classes) lets downstream algorithms skip or flag bad data.
- Users: (1) analysts and clinicians reviewing long-term Holter or patch recordings, who need bad segments flagged; (2) developers of wearable ECG firmware or apps, who need a cheap gate before beat detection; (3) researchers who need reproducible, subject-independent benchmarks.
- Practical constraint: the gate has to work on people it was not trained on. This is why the project uses leave-one-subject-out (LOSO) evaluation.

## Key papers

| # | Reference | Link | What it does | Limits or relevance |
|---|---|---|---|---|
| 1 | Pan J, Tompkins WJ. A real-time QRS detection algorithm. IEEE Trans Biomed Eng. 1985;32(3):230-236. PMID 3997178. | https://doi.org/10.1109/TBME.1985.325532 | Real-time QRS detector: band-pass filter, derivative, squaring, moving-window integration, adaptive thresholds with search-back. Tested on the MIT/BIH arrhythmia database. The indexed abstract reports 99.3% of QRS complexes correctly detected (index/abstract only). Elgendi et al. 2010 describe its filter pass band as about 5-11 Hz. | Metadata verified via Europe PMC. The full text (a scanned PDF) was not read for this log, so check filter and threshold details against the paper while implementing. Under the JF stress test of Porr and Macfarlane 2024 (row 17), Pan and Tompkins had the lowest score of the detectors tested. |
| 2 | Hamilton PS, Tompkins WJ. Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database. IEEE Trans Biomed Eng. 1986;33(12):1157-1165. PMID 3817849. | https://doi.org/10.1109/TBME.1986.325695 | Feeds filtered MIT/BIH data to candidate decision rules and measures how each rule affects detection, then builds a final real-time detector. Elgendi 2013 cites its result as Se 99.69% and +P 99.77% over 109,267 beats. | Full text not read. One index gives +P as 98.77%, so the figure is not verified. This is the basis of the "Hamilton" detectors in py-ecg-detectors, BioSPPy and NeuroKit2 (those tools cite Hamilton 2002, EP Limited open-source software). |
| 3 | Elgendi M, Jonkman M, De Boer F. Frequency bands effects on QRS detection. BIOSIGNALS 2010, pp. 428-431. | https://doi.org/10.5220/0002742704280431 | Opened. Introduces the two-moving-average detector as a test detector for comparing band-pass ranges: second-order Butterworth band-pass, then MA_QRS (W1 = 44 samples) and MA_beat (W2 = 231 samples) at 360 Hz. Samples where MA_QRS > MA_beat form "blocks of interest", blocks narrower than W1 are rejected, and the maximum absolute value in each block is the R peak. With an 8-20 Hz band it reports Se 98.31% and +P 99.92% over 109,493 beats (48 MIT-BIH records). | This is where the algorithm is first defined, and it is the reference that py-ecg-detectors cites for `two_average_detector`. It has no threshold offset. |
| 4 | Elgendi M. Fast QRS detection with an optimized knowledge-based method: evaluation on 11 standard ECG databases. PLoS ONE. 2013;8(9):e73557. PMID 24066054. | https://doi.org/10.1371/journal.pone.0073557 | Opened. Optimized and justified version of the same detector: 8-20 Hz Butterworth band-pass, squaring, W1 = 97 ms, W2 = 611 ms, a threshold offset (beta times the mean of the squared signal), and accepted blocks of width >= W1. Tuned on MIT-BIH (124 FP, 247 FN reported in the text) and tested on 10 other databases, including NSTDB, without retuning. | Sole author (not "et al."). Cite this paper for the tuned parameters and paper 3 for the origin. The overall Se/+P table was not extracted for this log. |
| 5 | Li Q, Mark RG, Clifford GD. Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter. Physiol Meas. 2008;29(1):15-32. PMID 18175857. | https://doi.org/10.1088/0967-3334/29/1/002 | Fuses HR estimates from ECG and ABP with Kalman filters weighted by SQIs. Defines bSQI as the fraction of beats detected by two detectors (ep limited and wqrs) that agree within a window, using a 150 ms matching tolerance "as per" ANSI/AAMI EC57 1998 (search snippet of the full text). Evaluated on over 6000 h from 437 ICU patients (abstract). | This is where bSQI comes from. It is ICU multi-lead data, not wearable data. Definitions of the paper's other SQIs were not verified. |
| 6 | Clifford GD, Behar J, Li Q, Rezek I. Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms. Physiol Meas. 2012;33(9):1419-1433. PMID 22902749. | https://doi.org/10.1088/0967-3334/33/9/1419 | SQIs based on spectral energy distribution, higher-order moments, and inter-channel and inter-algorithm agreement, fused with machine learning. Abstract: 98% accuracy on training data and 97% on test data (PhysioNet/CinC Challenge 2011). | Binary acceptable/unacceptable labels for 12-lead recordings, so it does not map directly to 3-class single-lead wearable data. |
| 7 | Behar J, Oster J, Li Q, Clifford GD. ECG signal quality during arrhythmia and its application to false alarm reduction. IEEE Trans Biomed Eng. 2013;60(6):1660-1666. PMID 23335659. | https://doi.org/10.1109/TBME.2013.2240452 | SQIs fed to an SVM on more than 33,000 single-lead 10 s segments (Challenge 2011, MIT-BIH Arrhythmia, MIMIC II), plus about 12,000 noisy segments generated with NSTDB. Reports up to 99% accuracy for normal sinus rhythm and up to 95% for arrhythmias (abstract), and recommends rhythm-specific SQIs. | Shows that arrhythmia confounds quality labels. Uses NSTDB to synthesize noise, which is relevant to our rejection experiment. Full text not read. |
| 8 | Orphanidou C, Bonnici T, Charlton P, Clifton D, Vallance D, Tarassenko L. Signal-quality indices for the electrocardiogram and photoplethysmogram: derivation and applications to wireless monitoring. IEEE J Biomed Health Inform. 2015;19(3):832-838. PMID 25069129. | https://doi.org/10.1109/JBHI.2014.2338351 | SQI that decides whether a reliable HR can be obtained from wearable ECG and PPG. ECG: sensitivity 94%, specificity 97% against manual labels (abstract). Also used to trigger a power-saving scheme. | Binary, so there is no "QRS-only" middle class. NeuroKit2 implements a template-matching variant (`templatematch` / `orphanidou2015`). |
| 9 | Zhao Z, Zhang Y. SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation. Front Physiol. 2018;9:727. | https://doi.org/10.3389/fphys.2018.00727 | Rule-based fusion of pSQI, kSQI, basSQI and qSQI into 3 classes (excellent, barely acceptable, unacceptable). Found via search and the NeuroKit2 source; the paper itself was not opened. | A useful 3-class rule-based comparison. The paper's reported results were not read, so none are quoted here. |
| 10 | Smital L, et al. Real-time quality assessment of long-term ECG signals recorded by wearables in free-living conditions. IEEE Trans Biomed Eng. 2020;67(10):2721-2734. | https://doi.org/10.1109/TBME.2020.2969719 | BUT group method: a continuous SNR curve in a local window, with 3 quality levels (full-wave usable, QRS-only, unusable). Index records give thresholds of about 5 dB (QRS) and 18 dB (full wave). Developed on synthetic data and validated on wearable recordings (index/abstract only). | Same 3-level definition as BUT QDB. Used as the baseline in paper 11. Full author list from index (12 authors). |
| 11 | Smital L, Nemcova A, Saclova L, Smisek R, Opravilova K, Filipenska M, Vitek M. Real-world 24h+ ECG dataset with quality annotations and motion context. Sci Data. 2026;13:1567. PMID 42476988. | https://doi.org/10.1038/s41597-026-07905-w | Opened (via PMC). Data descriptor for BUT QDB: more than 86 h expert-labelled, about 2.2 million cardiac cycles. Standard protocol: one fixed subject-independent split (train: 11 recordings, 8 subjects; test: 7 recordings, 7 subjects), sample-level confusion matrix, per-class F1 and accuracy. SNR baseline (paper 10) on the test set: accuracy 74.08%, F1 Q1 83.04%, Q2 43.39%, Q3 81.91%. ACC data were hidden from the annotators. Code: https://github.com/BioSig-BUT/BUT-QDB-supplement | No accelerometer baseline is reported and LOSO is not discussed. Our LOSO numbers will not be directly comparable to this fixed split, so the fixed split should also be reported. |
| 12 | Liu F, Xia S, Wei S, et al. Wearable electrocardiogram quality assessment using wavelet scattering and LSTM. Front Physiol. 2022;13:905447. | https://doi.org/10.3389/fphys.2022.905447 | Opened. 3 classes, using BUT QDB plus the Challenge 2011 data, with NSTDB used for class balancing. 10-fold CV on randomly divided segments: mACC 98.56 +/- 0.39%, mF1 98.55 +/- 0.40%. Cross-database: BUT QDB to PhysioNet mACC 85.32%, PhysioNet to BUT QDB 82.73%. | Segment-level random split, so the same subjects can appear in train and test. No accelerometer. The cross-database drop shows a generalization gap. |
| 13 | Ma C, Wang Z, Zhao L, Long X, Vullings R, Aarts RM, Li J, Liu C. Deep learning-based signal quality assessment in wearable ECG monitoring. Computing in Cardiology 2023;50. | https://doi.org/10.22489/CinC.2023.017 | Opened. xResNet34 on BUT QDB: 10 s windows with 5 s overlap, downsampled to 200 Hz, 5-fold CV. The 3-class model's per-class accuracy is A 96.62%, B 93.66%, C 98.97% in the abstract, but the results text gives 98.86% for C. | The split unit (subject or segment) is not stated, and overlapping windows risk leakage. No accelerometer. |
| 14 | Tran KD. When can predictive uncertainty be trusted? A methodological evaluation in free-living wearable electrocardiogram signal-quality assessment. medRxiv preprint, posted 2026-08-28. | https://doi.org/10.64898/2026.08.25.26361304 | Abstract (via the medRxiv API): 32,224 non-overlapping 10 s windows of single-lead ECG plus 3-axis ACC from 15 BUT QDB subjects. Logistic regression and CART with Bayesian extensions, 9 features, uncertainty measures, and zero-shot external transfer. A search snippet of the full text states LOSO over 15 subjects with fold-wise standardisation. | Not peer reviewed. The full text returned 403, so the feature list and any ECG-only vs ECG+ACC comparison are not verified. This is the closest prior work to the candidate question. |
| 15 | Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. Computers in Cardiology. 1984;11:381-384. | https://physionet.org/content/nstdb/1.0.0/ | Origin of NSTDB: calibrated recorded noise added to clean ECG at set SNRs. | No DOI found. The citation was taken from the PhysioNet NSTDB page. |
| 16 | Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol Mag. 2001;20(3):45-50. PMID 11446209. | https://doi.org/10.1109/51.932724 | Describes MIT-BIH Arrhythmia, the reference publication PhysioNet requests for the database. | Citation verified on the PhysioNet MIT-BIH page. The paper itself was not read. |
| 17 | Porr B, Macfarlane PW. A new QRS detector stress test combining temporal jitter and F-score (JF) reveals significant performance differences amongst popular detectors. PLoS ONE. 2024;19(11):e0309739. | https://doi.org/10.1371/journal.pone.0309739 | Opened. JF combines F1 with a temporal-jitter score and uses no fixed tolerance window. Tested on the Glasgow University Database (GUDB: 25 subjects, 5 activities). EngZee, Elgendi and WQRS scored best, Pan and Tompkins lowest. | Argues that tolerance windows hide timing errors. Supports reporting timing error alongside Se/+P at 150 ms. An earlier bioRxiv version is by Porr and Howell (https://doi.org/10.1101/722397). |
| 18 | Ghaleb FA, et al. Two-stage motion artefact reduction algorithm for electrocardiogram using weighted adaptive noise cancelling and recursive Hampel filter. PLoS ONE. 2018;13(11):e0207176. | https://doi.org/10.1371/journal.pone.0207176 | Opened. Reports weak, not necessarily linear correlation between acceleration and the ECG motion artefact. Correlation was below 0.2 at lag zero in the sitting analysis, and accelerometer-referenced adaptive filtering showed no enhancement in the walking and running examples. | Evidence that ACC may be a weak proxy for ECG artefact, so a null result for the candidate question is plausible. This is a filtering study, not a quality-classification study. |
| 19 | Tabassum H, Mummidi S, An J, Xia X, Dzikowicz DJ, Couderc J-P. Evaluating ECG signal quality assessment methods on real-world wearable data. Computing in Cardiology 2026 (accepted abstract). | https://cinc.org/2026/Program/accepted/187.html | Opened (abstract only). Compares template matching, fuzzy, rule-based and CNN-LSTM methods on MIT-BIH with synthetic noise and on 3,403 segments from a 14-day patch. On real data the CNN-LSTM gave about 4% relative improvement over rule-based; the rule-based method had about 950 times higher throughput. | Preliminary abstract, with no BUT QDB and no accelerometer. Supports keeping a lightweight rule-based baseline. |

## Datasets

**BUT QDB (Brno University of Technology ECG Quality Database)**, opened at https://physionet.org/content/butqdb/1.0.0/
- Citation (dataset): Nemcova A, Smisek R, Opravilova K, Vitek M, Smital L, Marsanova L. Brno University of Technology ECG Quality Database (BUT QDB) (version 1.0.0). PhysioNet. 2020. https://doi.org/10.13026/kah4-0w24
- PhysioNet also asks users to cite the descriptor (Smital et al. 2026, Sci Data, key paper 11) and the standard PhysioNet citation (Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform for biomedical research. Nature Health. 2026;1(8):792-795. https://doi.org/10.1038/s44360-026-00096-z).
- Content: 18 recordings of at least 24 h from 15 subjects, recorded with a Bittium Faros 180 in free-living conditions. Single-lead ECG at 1000 Hz and 3-axis ACC at 100 Hz, in WFDB format (`*_ECG`, `*_ACC`). License CC BY 4.0. Version 1.0.0 was published 2020-07-22. A "latest" DOI (10.13026/qcnm-a155) is also listed; its version number was not verified.
- Labels: class 1 (all waveforms clear), class 2 (QRS reliable, other points not), class 3 (QRS not reliably detectable). The annotation CSV has 12 columns: start, end and class for each of 3 annotators, then the consensus. Only 3 records are fully annotated; the other 15 have two 20-min annotated segments each, and 5 extra poor-quality segments were added.
- Subject grouping for LOSO: the first three digits of a record name are the subject ID (e.g. 103001, 103002, 103003 are one subject). `subject-info.csv` holds sex, age, height, weight and smoking status.

**MIT-BIH Arrhythmia Database**, opened at https://physionet.org/content/mitdb/1.0.0/
- Citations: Moody GB, Mark RG (2001), key paper 16. Dataset DOI https://doi.org/10.13026/C2F305 (v1.0.0). PhysioNet lists a separate "latest" DOI (10.13026/dwn9-vt73).
- Content: 48 two-channel records of about 30 min from 47 subjects, 360 Hz, 11-bit over 10 mV, about 110,000 beat annotations.

**MIT-BIH Noise Stress Test Database (NSTDB)**, opened at https://physionet.org/content/nstdb/1.0.0/
- Citations: Moody GB, Muldrow WE, Mark RG (1984), key paper 15. Dataset DOI https://doi.org/10.13026/C2HS3T.
- Content: 3 half-hour noise records (baseline wander `bw`, muscle artifact `ma`, electrode motion `em`) and 12 half-hour test records. These are MIT-BIH 118 and 119 with `em` noise at SNR 24, 18, 12, 6, 0 and -6 dB. Noise starts after the first 5 min and alternates 2 min noisy / 2 min clean. Reference annotations are copied from the clean records.

## Evaluation standards

- Standard: ANSI/AAMI EC57, "Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms". BSI Knowledge lists ANSI/AAMI EC57:2012/(R)2020 as current (https://knowledge.bsigroup.com/products/testing-and-reporting-performance-results-of-cardiac-rhythm-and-st-segment-measurement-algorithms). It is paywalled, so the points below come from PhysioNet documentation (secondary source).
- Match window: test beat annotations must fall within 150 ms of the reference annotation to count as a match (WFDB Programmer's Guide, "Evaluation of ECG Analyzers": https://physionet.org/physiotools/wpg/wpg_67.htm).
- Reference implementation: `bxb` implements the EC38/EC57 beat-by-beat comparison. The default match window is 0.15 s, and changing it with `-w` is not permitted for EC57 compliance (https://physionet.org/physiotools/wag/bxb-1.htm).
- Learning period: by default, comparison starts 5 min after the record start. The first 5 min are treated as the detector's learning period and are not scored (both PhysioNet pages above).
- Databases: EC38/EC57 specify the MIT DB and the AHA DB, plus the ESC DB for ST analysis (wpg_67 page).
- Metrics: Se = TP/(TP+FN) and +P = TP/(TP+FP). These are the conventional definitions used by the papers above; the standard's exact wording was not read.
- Implications for this project:
  - 150 ms is 54 samples at 360 Hz. WFDB `compare_annotations` takes `window_width` in samples.
  - State whether the first 5 min are excluded, and report gross (pooled) and per-record Se/+P.
  - Following key paper 17, also report the timing error of matched beats.
- Quality classification: no formal standard was found. The BUT QDB descriptor (key paper 11) defines a fixed subject-independent split with sample-level F1 and accuracy. This project uses LOSO over 15 subjects and also reports results on the fixed split for comparison.

## Open-source implementations

| Tool | Citation and link | What it offers | Limits |
|---|---|---|---|
| NeuroKit2 | Makowski D et al. NeuroKit2: a Python toolbox for neurophysiological signal processing. Behav Res Methods. 2021;53(4):1689-1696. https://doi.org/10.3758/s13428-020-01516-y. Docs: https://neuropsychology.github.io/NeuroKit/functions/ecg.html | `ecg_peaks` with many methods, including `pantompkins1985`, `hamilton2002`, `elgendi2010`, `christov2004`, `engzeemod2012`, `kalidas2017`, `nabian2018`, `rodrigues2021`, `emrich2023`, `promac` and the default `neurokit`, plus optional artifact correction (`correct_artifacts`). `ecg_quality` offers `averageQRS` (default), `templatematch` (orphanidou2015), `dissimilarity` (sabeti2019), `zhao2018` (simple or fuzzy, 3 categories) and `ho2025`. | `averageQRS` is relative to the record's own average beat, so a value of 1 does not mean good quality. `zhao2018` drops qSQI and changes the weights. `dissimilarity` omits DTW. The `neurokit` and `promac` methods are unpublished, and the docs ask for help verifying implementations. |
| WFDB Python | Xie C, McCullum L, Johnson A, Pollard T, Gow B, Moody B. Waveform Database Software Package (WFDB) for Python (version 4.1.0). PhysioNet. 2023. https://doi.org/10.13026/9njx-6322. Docs (v4.3.1): https://wfdb.readthedocs.io/en/latest/processing.html | Reads PhysioNet records. `xqrs_detect` (XQRS): 5-20 Hz band-pass, Ricker-wavelet moving-wave integration, learning-based initialisation, and back-search at 1.66 times the recent RR. `gqrs_detect`: a port of WFDB GQRS. `compare_annotations` with `window_width` in samples, returning a Comparitor with `print_summary`. `benchmark_mitdb` reports sensitivity and positive predictivity. | The docs say GQRS "is not being developed/supported" and has known bugs and no amplitude normalisation, so it is not used here. `compare_annotations` is not `bxb`, and its docs (as read) do not describe the EC57 learning-period convention. |
| py-ecg-detectors | Howell L, Porr B. Zenodo DOI https://doi.org/10.5281/zenodo.3353396. Repo: https://github.com/berndporr/py-ecg-detectors (GPL-3.0) | 8 detectors: Hamilton, Christov, Engelse-Zeelenberg (modified), Pan-Tompkins, SWT, two moving average (cites Elgendi 2010), matched filter and WQRS. Related benchmark: JF (key paper 17, https://github.com/berndporr/JF-ECG-Benchmark). | The README states that most detectors introduce a delay and do not locate the true R peak. The matched filter needs a template file. No release versions are published, so pin a commit hash. |
| BioSPPy | Bota P, Silva R, Carreiras C, Fred A, da Silva HP. BioSPPy: a Python toolbox for physiological signal processing. SoftwareX. 2024. https://doi.org/10.1016/j.softx.2024.101712. Repo: https://github.com/scientisst/BioSPPy (BSD-3). Docs: https://biosppy.readthedocs.io/en/stable/biosppy.signals.html | `ecg.ecg()` pipeline (filtered signal, R peaks, templates, HR). Segmenters: Hamilton, Christov, Engzee, Gamboa, SSF and ASI. Quality functions: `bSQI`, `fSQI`, `kSQI`, `pSQI`, `sSQI` and `ZZ2018`. | BioSPPy's `pSQI` is a flatline percentage, not the 5-15 Hz / 5-40 Hz power ratio used elsewhere, so define SQIs explicitly. The `ZZ2018` docstring is empty, and no version number was visible in the docs. |

SQI definitions used in this project, as implemented in the NeuroKit2 `zhao2018` source (https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_quality.html):
- pSQI = P(5-15 Hz) / P(5-40 Hz)
- basSQI = 1 - P(0-1 Hz) / P(0-40 Hz)
- kSQI = kurtosis
- bSQI follows Li et al. 2008 (key paper 5).

## Products

- **AliveCor KardiaMobile.** The official IFU found is for the KardiaMobile Card (AC-021), document 21LB01.6, August 2025: https://cdn.alivecor.com/ifus/km-card/21LB01.6_KardiaMobile_Card_Instructions_for_Use_(IFU)_locked.pdf (opened).
  - Recordings are 30 s. Recordings shorter than 30 s get "Too short".
  - "Unreadable" means the recording has too much interference. The user is told to re-record, relax, rest the arms and move away from electronics. "Unclassified" is a separate result for recordings that do not fit the algorithm's classes.
  - The IFU warns that ventricular flutter, bigeminy and trigeminy may be wrongly reported as unreadable.
  - Not checked: IFUs for the original KardiaMobile model.
- **Apple Watch ECG app.** "Take an ECG with the ECG app on Apple Watch": https://support.apple.com/en-us/120278 (opened).
  - Recordings are 30 s. Possible results are sinus rhythm, AFib, low or high heart rate, Inconclusive, and Poor Recording.
  - Inconclusive covers pacemakers or ICDs, other arrhythmias, and insufficient signal, plus (version 1 only) HR of 100-120 BPM or a loose fit.
  - Poor Recording is unique to ECG version 2 and means the recording cannot be classified. The suggested fixes are rest, a snug fit, a clean dry wrist and distance from plugged-in electronics.
- **Relevance.** Both products use a binary "cannot classify" gate for 30 s spot checks, and that gate also absorbs out-of-scope rhythms (the confound noted by Behar et al. 2013). This project addresses continuous free-living recordings with 3 levels, so the products are context, not comparators.

## Baselines chosen for this project

- **Detector baseline 1: Pan-Tompkins, implemented from the paper.**
  - It is the most-cited reference algorithm, and most toolkits include it (NeuroKit2 `pantompkins1985`, py-ecg-detectors).
  - Comparing our implementation with those versions on MIT-BIH checks correctness.
  - Expect it to be weaker in noise (key paper 17).
- **Detector baseline 2: WFDB XQRS.**
  - An independently maintained PhysioNet implementation from a different design family (wavelet integration with a learning phase).
  - Shares tooling with the dataset format and the comparison function.
  - GQRS is excluded because its own docs mark it as unsupported.
- **Second detector under study: Elgendi two moving averages.** Implement the 2013 parameters (key paper 4) and cite 2010 (key paper 3) for the origin.
- **Quality baseline: rule-based bSQI threshold.**
  - bSQI is the original detector-agreement index (key paper 5).
  - It needs no training data except one threshold, so it is naturally subject-independent and interpretable.
  - It reuses the same 150 ms tolerance as the beat evaluation.
  - Pair two detectors with different failure modes. Li et al. used ep limited and wqrs; here, Pan-Tompkins and Elgendi or XQRS.
  - Tune the threshold on training subjects only within each LOSO fold.
  - kSQI, skewness, pSQI and basSQI are added as rule-based features. Gradient boosting and the 1D CNN must beat this baseline to justify their cost (key paper 19 found small real-world gains for deep learning).
- **External reference point:** the SNR baseline on the BUT QDB fixed split (key paper 11: accuracy 74.08%), reported on that split only.

## Candidate research question: literature check

Candidate research question: do accelerometer features improve 3-class ECG quality classification for subjects not seen in training (LOSO on BUT QDB)?

What was found:
- BUT QDB was explicitly designed to allow ACC-informed or ACC-only quality assessment (PhysioNet page; Sci Data 2026 abstract). The Sci Data protocol provides no ACC baseline. ACC was hidden from annotators, so labels are ECG-based and ACC is an independent input.
- Peer-reviewed BUT QDB classifiers found (key papers 12 and 13) use ECG only and random or unspecified segment-level CV, so they do not address unseen subjects or ACC.
- One preprint (key paper 14) uses ECG plus 3-axis ACC with LOSO over the 15 BUT QDB subjects. Its question is uncertainty reliability, and its abstract reports no ECG-only vs ECG+ACC comparison. The full text could not be opened (HTTP 403), so it is unknown whether it contains such an ablation.
- Evidence on ACC as an artefact reference is mixed to weak: correlation with ECG artefact was weak and ACC-referenced adaptive filtering did not help in the walking and running examples (key paper 18). This is a filtering result, not a classification result.
- A 2026 multimodal ECG+ACC study (Sensors 26(4):1135, https://doi.org/10.3390/s26041135) targets arrhythmia screening on MIT-BIH with added noise, not quality classification on BUT QDB. Search index only; the full text returned 403.

Status: **appears open, based only on what was found.** No peer-reviewed source reports an ECG-only vs ECG+ACC comparison for BUT QDB quality classification on unseen subjects. One adjacent preprint (key paper 14) may partly address it. Re-check its full text before finalising the question, and cite it either way.

## Search log

All searches and fetches were run on 2026-11-14 with web search and direct page fetches.

Web search queries:
1. Pan Tompkins "A real-time QRS detection algorithm" IEEE Transactions on Biomedical Engineering 1985 doi
2. Hamilton Tompkins 1986 "Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database"
3. Elgendi "Fast QRS detection with an optimized knowledge-based method" PLoS ONE two moving averages
4. Li Mark Clifford 2008 "Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter" Physiological Measurement
5. Clifford Behar Li Rezek 2012 "Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms" Physiological Measurement doi
6. Behar Oster Li Clifford 2013 "ECG signal quality during arrhythmia and its application to false alarm reduction" IEEE Transactions on Biomedical Engineering
7. Orphanidou 2015 "Signal-quality indices for the electrocardiogram and photoplethysmogram: derivation and applications to wireless monitoring" IEEE JBHI
8. Smital Haider Vitek "Real-time quality assessment of long-term ECG signals recorded by wearables in free-living conditions" IEEE Transactions on Biomedical Engineering 2020
9. BUT QDB ECG quality classification accelerometer deep learning
10. "Real-world 24h+ ECG dataset with quality annotations and motion context" Scientific Data
11. BUT QDB ECG quality accelerometer "leave-one-subject-out" OR "subject-independent" wearable signal quality
12. ECG signal quality assessment accelerometer features motion artifact wearable classification improvement
13. "BUT QDB" accelerometer ECG quality classification
14. Elgendi Jonkman De Boer "Frequency Bands Effects on QRS Detection" BIOSIGNALS 2010
15. Li Mark Clifford 2008 bSQI "wqrs" "eplimited" agreement 150 ms kSQI pSQI basSQI definition
16. "basSQI" "pSQI" "kSQI" "bSQI" ECG signal quality index definition 5-15 Hz relative power baseline
17. Makowski NeuroKit2 "A Python toolbox for neurophysiological signal processing" Behavior Research Methods 2021 doi
18. AliveCor Kardia "Unreadable" result what does it mean support
19. alivecor.com support "Unreadable" EKG result tips avoid
20. Apple support "Take an ECG with the ECG app on Apple Watch" inconclusive poor recording
21. ANSI/AAMI EC57 "150 ms" beat-by-beat comparison sensitivity positive predictivity QRS detection standard
22. "EC57:2012" AAMI "Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms"
23. ECG signal quality classification wearable accelerometer features cross-subject evaluation motion artifact detection Holter
24. "From Motion Artifacts to Clinical Insight" multi-modal deep learning arrhythmia screening ambulatory ECG Sensors 2026 accelerometer dataset
25. Brno ECG quality accelerometer Smital Nemcova motion "ECG quality" classification Faros 2021 OR 2022 OR 2023 OR 2024
26. "Deep Learning-based Signal Quality Assessment in Wearable ECG Monitoring" Computing in Cardiology 2023 10.22489
27. Porr Howell "R-peak detector stress test with a new noisy ECG database reveals significant performance differences amongst popular detectors" bioRxiv
28. PhysioNet "Waveform Database Software Package (WFDB) for Python" citation version doi

Metadata lookups: Europe PMC REST queries by DOI (papers 1, 2, 4 to 8 and 11) and the bioRxiv/medRxiv API (paper 14).

Fetches that failed: HTTP 403 for medRxiv full text, MDPI, ANSI webstore, GlobalSpec and core.ac.uk; cookie or verification walls on PubMed and one PMC page; 404 for lcp.mit.edu/pdf/LiPM08.pdf; empty response from IEEE Xplore. Details from these sources are marked as index/abstract only or not verified.
