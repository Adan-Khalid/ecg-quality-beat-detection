"""Download the PhysioNet datasets used by this project and verify their checksums.

Each PhysioNet project publishes SHA256SUMS.txt, which doubles as the file list.
Files that already exist with the right checksum are skipped, so the script can be
re-run after an interrupted download.

Usage:
    python scripts/download_data.py --datasets mitdb nstdb butqdb --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet.org/files"
DATASETS = {
    "mitdb": "1.0.0",  # MIT-BIH Arrhythmia Database, ODC-By 1.0
    "nstdb": "1.0.0",  # MIT-BIH Noise Stress Test Database, ODC-By 1.0
    "butqdb": "1.0.0",  # Brno University of Technology ECG Quality Database, CC BY 4.0
}
USER_AGENT = "ecg-quality-beat-detection/0.1 (research download script)"


def _get(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, expected: str, retries: int = 20) -> str:
    if dest.exists() and _sha256(dest) == expected:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            # Resume a partial file with an HTTP Range request (PhysioNet supports it), so a
            # dropped connection on a 300 MB file does not restart it from zero.
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with tmp.open(mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            if _sha256(tmp) != expected:
                tmp.unlink()  # corrupt: start this file again on the next attempt
                raise OSError(f"checksum mismatch for {dest.name}")
            tmp.replace(dest)
            return "ok"
        except Exception as exc:  # network errors are retried, keeping the partial file
            if attempt == retries:
                raise
            print(f"  retry {attempt} for {dest.name}: {exc}", flush=True)
            time.sleep(5 * attempt)
    return "fail"


def download_dataset(name: str, out_root: Path, workers: int = 4) -> None:
    version = DATASETS[name]
    base_url = f"{BASE}/{name}/{version}"
    sums = _get(f"{base_url}/SHA256SUMS.txt").decode("utf-8").splitlines()
    entries = []
    for line in sums:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            entries.append((parts[1].strip(), parts[0].strip()))
    target = out_root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"[{name}] {len(entries)} files -> {target}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_file, f"{base_url}/{rel}", target / rel, digest): rel
            for rel, digest in entries
        }
        for fut in as_completed(futures):
            rel = futures[fut]
            status = fut.result()
            done += 1
            if status != "skip" or done % 50 == 0:
                print(f"[{name}] {done}/{len(entries)} {status} {rel}", flush=True)
    print(f"[{name}] complete", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["mitdb", "nstdb", "butqdb"])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    for name in args.datasets:
        if name not in DATASETS:
            parser.error(f"unknown dataset {name}; choose from {sorted(DATASETS)}")
        download_dataset(name, args.out, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
