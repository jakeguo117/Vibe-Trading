#!/usr/bin/env python3
"""Upload the QM run to Drive, read it back, verify hashes, and report links."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

RESULT_ROOT = "1kRID4peUueFv10wFC2wrkrICL5hw40eI"   # 已有正式回测结果正本
STUDY_FOLDER = "187atRsVfZn_WAr0kxskPvuyMioeYSKrU"    # QM-RESEARCH-001 研究目录
RUN = "runs/qm-001-20261022-0041"
RUN_TAG = "qm-research-001-20261022"
DATE_TAG = "2026-10-22_qm-research-001"

KEY = [
    "00_summary_zh.md", "data_readiness.md", "spec_frozen.md", "test_evidence.md",
    "reproduction.md", "statistics.md", "attribution.md",
    f"{RUN}/config/config.json", f"{RUN}/config/manifest.json",
    f"{RUN}/deliverables/comparisons.csv", f"{RUN}/deliverables/acceptance.csv",
    f"{RUN}/deliverables/comparison_table.csv", f"{RUN}/deliverables/funnel.csv",
    f"{RUN}/deliverables/stress.csv", f"{RUN}/deliverables/borrow_scenarios.csv",
    f"{RUN}/deliverables/statistics.csv", f"{RUN}/deliverables/attribution.csv",
    f"{RUN}/checkpoint.json",
]


def run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"failed: {cmd}")
    return r.stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    # 1. remote run folder
    remote = f"{DATE_TAG}/{RUN_TAG}"
    run(f"rclone mkdir gdrive:{remote} --drive-root-folder-id {RESULT_ROOT} 2>/dev/null")
    # 2. upload whole research dir except heavy raw data
    excludes = ["--exclude data/panel.npz", "--exclude *.npz", "--exclude __pycache__/**",
                "--exclude runs/*/data/*.npz", "--exclude .pytest_cache/**"]
    cmd = (f"rclone copy . gdrive:{remote} --drive-root-folder-id {RESULT_ROOT} "
           + " ".join(excludes) + " --transfers 8 2>&1 | grep -v NOTICE | tail -3")
    print(run(cmd, check=False))
    # 3. study-dir docs
    for f in ["data_readiness.md", "spec_frozen.md", "00_summary_zh.md", "test_evidence.md",
              "statistics.md", "attribution.md", "reproduction.md", "README.md"]:
        if os.path.exists(f):
            run(f"rclone copyto {f} gdrive:{f} --drive-root-folder-id {STUDY_FOLDER} 2>/dev/null")
    # 4. read back key files and compare
    rb = "/tmp/qm_readback"
    os.makedirs(rb, exist_ok=True)
    rows = []
    for f in KEY:
        if not os.path.exists(f):
            rows.append((f, "MISSING_LOCAL", "", ""))
            continue
        local = sha(f)
        out = os.path.join(rb, os.path.basename(f))
        r = subprocess.run(f"rclone copyto gdrive:{remote}/{f} {out} --drive-root-folder-id {RESULT_ROOT} 2>/dev/null",
                           shell=True, capture_output=True, text=True)
        ok = os.path.exists(out) and sha(out) == local
        rows.append((f, "OK" if ok else "MISMATCH", local[:16], sha(out)[:16] if os.path.exists(out) else ""))
    with open(f"{RUN}/artifact_readback.txt", "w") as fh:
        for f, st, a, b in rows:
            fh.write(f"{st:14s} {a} {b} {f}\n")
    print("\n".join(f"{s:14s} {a} {f}" for f, s, a, b in rows))
    print(f"\nremote: https://drive.google.com/drive/folders/{RESULT_ROOT}")
    bad = [r for r in rows if r[1] != "OK"]
    print("readback_bad", len(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
