#!/usr/bin/env python3
"""
Hard-coded filter:
  - Reads nist_train_disjoint.mgf
  - Loads library.json (array of objects)
  - Keeps only records whose 'spectrum_id' appears as SPECTRUMID in the MGF
  - Writes filtered_library.json
"""

import json
import gzip
import re
from pathlib import Path

# ---------- Hard-coded paths (edit these as needed) ----------
MGF_PATH = Path("data/spec_datasets/nist20/nist_train_disjoint.mgf")  
LIB_PATH = Path("data/spec_datasets/nist20/library.json")
OUT_PATH = Path("data/spec_datasets/nist20/filtered_library.json")
# -------------------------------------------------------------

SPECTRUMID_RE = re.compile(r'^\s*SPECTRUMID\s*=\s*(\S+)\s*$', re.IGNORECASE)

def open_text(path: Path):
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")

def collect_ids(mgf_path: Path):
    ids = set()
    with open_text(mgf_path) as fh:
        for line in fh:
            m = SPECTRUMID_RE.match(line)
            if m:
                ids.add(m.group(1))
    return ids

def main():
    if not MGF_PATH.exists():
        raise SystemExit(f"[ERROR] MGF not found: {MGF_PATH.resolve()}")
    if not LIB_PATH.exists():
        raise SystemExit(f"[ERROR] library.json not found: {LIB_PATH.resolve()}")

    mgf_ids = collect_ids(MGF_PATH)
    print(f"[INFO] Collected {len(mgf_ids)} SPECTRUMID(s) from {MGF_PATH}")

    with open(LIB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("[ERROR] library.json must be a JSON array")

    kept = [rec for rec in data if isinstance(rec, dict) and rec.get("spectrum_id") in mgf_ids]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(f"[INFO] Input records: {len(data)}")
    print(f"[INFO] Kept (present in MGF): {len(kept)}")
    print(f"[INFO] Wrote: {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
