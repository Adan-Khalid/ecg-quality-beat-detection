# Data card

All three datasets are open access on PhysioNet and were checked on 2026-11-14 (pages opened, licence and size read, SHA-256 checksums verified on download). Raw data is not in this repository; `data/README.md` gives the download command.

## 1. BUT QDB: Brno University of Technology ECG Quality Database

| Field | Value |
| --- | --- |
| Source | https://physionet.org/content/butqdb/1.0.0/ |
| Version | 1.0.0, published 2020-07-22 |
| DOI | https://doi.org/10.13026/kah4-0w24 |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Size | 2.2 GB zip, 4.2 GB uncompressed (95 files) |
| Access | Open, no registration |
| Device | Bittium Faros 180 (single-lead ECG patch with integrated accelerometer) |
| Content | 18 recordings of at least 24 h from 15 subjects (9 female, 6 male, aged 21 to 83), free-living conditions |
| Signals | ECG at 1000 Hz, stored in microvolts (header gain 1.996 adu/uV); 3-axis accelerometer at 100 Hz in milli-g |
| Labels | Sample-wise quality classes from three experts plus a consensus (12-column `*_ANN.csv`). Class 1: P, QRS and T clearly visible. Class 2: noisy, but QRS reliably detectable. Class 3: QRS not reliably detectable. 0 = not annotated. |
| Annotated part | 3 recordings fully annotated (100001, 105001 from sample 46,800,001, 111001); the other 15 have two 20-minute annotated segments, plus 5 extra noisy segments |
| Subject ID | First three digits of the record name (103001, 103002 and 103003 are one subject) |

**Citation (dataset):** Nemcova A, Smisek R, Opravilova K, Vitek M, Smital L, Marsanova L. Brno University of Technology ECG Quality Database (BUT QDB) (version 1.0.0). PhysioNet. 2020. https://doi.org/10.13026/kah4-0w24

**Citation (descriptor, requested by PhysioNet):** Smital L, Nemcova A, Saclova L, Smisek R, Opravilova K, Filipenska M, Vitek M. Real-world 24h+ ECG dataset with quality annotations and motion context. Scientific Data. 2026;13:1567. https://doi.org/10.1038/s41597-026-07905-w

**How this project uses it**
- ECG is resampled to 250 Hz (anti-aliased polyphase filter) in one-hour chunks, because one 24 h record at 1000 Hz is 87 million samples.
- Windows are 10 s and non-overlapping. A window gets a label only if every sample in it carries the same consensus class; mixed windows are kept for prediction but not for training. Window counts per record and class: `reports/metrics/butqdb_window_counts.csv`.
- Accelerometer windows are the same 10 s spans at 100 Hz.
- Splits are by subject (leave-one-subject-out), never by window, so no person appears in both training and test data. The official fixed split from the dataset authors (11 training records, 7 test records, from https://github.com/BioSig-BUT/BUT-QDB-supplement, MIT licence) is also evaluated.

**Known issues and limits**
- Three long, fully annotated records contribute most labelled windows. Training uses per-record and per-class weights so they do not dominate (see `docs/DECISIONS.md`).
- Class 3 is rare in most records. Per-class F1 for class 3 depends on a few subjects and has wide uncertainty.
- One device, one lead position and 15 people from one country. Results may not transfer to wrist ECG, other patches or other populations.
- Annotators labelled the ECG without seeing the accelerometer, so accelerometer features are an independent input, not a label leak.

## 2. MIT-BIH Arrhythmia Database

| Field | Value |
| --- | --- |
| Source | https://physionet.org/content/mitdb/1.0.0/ |
| Version | 1.0.0 |
| DOI | https://doi.org/10.13026/C2F305 |
| Licence | Open Data Commons Attribution License v1.0 (ODC-By 1.0) |
| Size | 73.5 MB zip, 104.3 MB uncompressed |
| Content | 48 half-hour two-channel ambulatory records from 47 subjects, 360 Hz, about 110,000 expert beat labels |
| Use here | Validating the QRS detectors on channel 0 (MLII in 46 records, V5 in 102 and 104) |

**Citations:** Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng Med Biol Mag. 2001;20(3):45-50. https://doi.org/10.1109/51.932724 and the dataset DOI above.

**Scoring conventions used:** 150 ms match window; ventricular flutter episodes (between `[` and `]` annotations, record 207) excluded; the "ec57" protocol also skips the first 5 minutes as the learning period (as WFDB `bxb` does by default); paced records 102, 104, 107 and 217 are reported both included and excluded.

## 3. MIT-BIH Noise Stress Test Database (NSTDB)

| Field | Value |
| --- | --- |
| Source | https://physionet.org/content/nstdb/1.0.0/ |
| Version | 1.0.0 |
| DOI | https://doi.org/10.13026/C2HS3T |
| Licence | Open Data Commons Attribution License v1.0 (ODC-By 1.0) |
| Size | 67.7 MB zip |
| Content | MIT-BIH records 118 and 119 with electrode-motion noise added at SNRs of 24, 18, 12, 6, 0 and -6 dB (12 records), plus the three noise recordings (baseline wander, muscle, electrode motion) |
| Use here | Measuring how much rejecting class 3 windows improves beat detection. Reference beat labels are the original clean-record labels. The exact noisy stretches are found by subtracting the clean MIT-BIH record. |

**Citation:** Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia detectors. Computers in Cardiology. 1984;11:381-384.

## Standard PhysioNet citation

Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform for biomedical research. Nature Health. 2026. https://doi.org/10.1038/s44360-026-00096-z

## Intended use

Research and education on signal quality assessment. Not for diagnosis or any clinical decision.
