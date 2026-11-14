# Data

Raw data is not stored in git. Download it with the script below, which reads each PhysioNet project's `SHA256SUMS.txt`, downloads every file, verifies its checksum and resumes interrupted transfers.

```bash
python scripts/download_data.py --datasets mitdb nstdb butqdb --out data/raw
```

| Dataset | Folder | Size | Licence |
| --- | --- | --- | --- |
| MIT-BIH Arrhythmia Database 1.0.0 | `data/raw/mitdb` | 104 MB | ODC-By 1.0 |
| MIT-BIH Noise Stress Test Database 1.0.0 | `data/raw/nstdb` | 68 MB | ODC-By 1.0 |
| BUT QDB 1.0.0 | `data/raw/butqdb` | 4.2 GB | CC BY 4.0 |

To run only the detector validation, download `mitdb` alone (about two minutes).

Processed features are written to `data/processed/butqdb/*.npz` by `scripts/build_features.py` (about 1 GB, also not in git).

Sources, licences, citations and known issues: [docs/DATA_CARD.md](../docs/DATA_CARD.md).
