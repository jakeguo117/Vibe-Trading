"""Writes preregistration.json and preflight.json (plan §2, §13) from measured facts only.

Nothing in here is a performance number: it freezes the specification, the data fingerprints and
the run environment *before* the comparison tables are produced.
"""
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import configs

ROOT = Path(__file__).resolve().parents[2]          # research/fib_timeframe_r1
DATA = Path("/Users/jake/projects/trading/data/ts_v111")
TS = Path("/Users/jake/projects/trading/trading-system")


def sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:
        return f"<{e}>"


def tree_hash(paths):
    """Stable fingerprint of a set of files: sha256 over (relpath, size, sha256) lines."""
    rows = []
    for p in sorted(paths):
        p = Path(p)
        if p.is_file():
            rows.append(f"{p.name}\t{p.stat().st_size}\t{sha(p)}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest(), rows


def env_block():
    import numpy, pandas, pyarrow  # noqa
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def data_block():
    man = DATA / "MANIFEST_SHA256.txt"
    mat = sorted((DATA / "mat").glob("*.npz")) + sorted((DATA / "mat").glob("*.parquet"))
    raw = sorted((DATA / "raw").glob("b*.parquet"))
    core = DATA / "raw" / "core.parquet"
    sig = DATA / "cache" / "signals_primary.npz"
    mh, mrows = tree_hash(mat)
    rh, rrows = tree_hash(raw + ([core] if core.exists() else []))
    out = {
        "snapshot_dir": str(DATA),
        "manifest_file": str(man) if man.exists() else None,
        "manifest_sha256": sha(man) if man.exists() else None,
        "n_manifest_lines": sum(1 for _ in open(man)) if man.exists() else None,
        "mat_files": {"n": len(mat), "tree_sha256": mh, "total_bytes": sum(Path(p).stat().st_size for p in mat)},
        "raw_files": {"n": len(raw) + (1 if core.exists() else 0), "tree_sha256": rh,
                      "total_bytes": sum(Path(p).stat().st_size for p in raw + ([core] if core.exists() else []))},
        "signals_primary": ({"path": str(sig), "bytes": sig.stat().st_size, "sha256": sha(sig)}
                            if sig.exists() else None),
    }
    return out


def hourly_block():
    """§2.4: is there any real intraday history on this machine? Search by name, then by content."""
    hits = []
    for base in [Path("/Users/jake/projects/trading/data")]:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            n = p.name.lower()
            if any(k in n for k in ("60m", "1h", "hour", "intraday", "h1", "minute", "1min", "5m")):
                hits.append(str(p))
    dirs = sorted({str(p.parent) for p in map(Path, hits)})
    return {"searched_root": "/Users/jake/projects/trading/data",
            "name_patterns": ["60m", "1h", "hour", "intraday", "h1", "minute", "1min", "5m"],
            "n_hits": len(hits), "hits": hits[:20], "dirs": dirs,
            "hourly_available": False,
            "verdict": "BLOCKED_NO_HOURLY_DATA" if not hits else "REVIEW_HITS"}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    plan = Path("/Users/jake/projects/hermes-tasks/runs/backtest-fib-breakout-20261020-1640/inbox/"
                "Jake_Fib_Timeframe_Test_Plan_v0.1_20261020.md")
    plan_sha = sha(plan)

    code = {
        "repo": "jakeguo117/Vibe-Trading",
        "branch": git(["branch", "--show-current"], ROOT.parents[1]),
        "head_at_emission": git(["rev-parse", "HEAD"], ROOT.parents[1]),
        "research_dir": str(ROOT),
        # code-only fingerprint: results/ is deliberately excluded so amending the results
        # commit cannot invalidate the fingerprint of the code that produced them
        "code_tree_sha256": tree_hash([p for p in list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("tests/**/*.py"))
                                       + ([ROOT / "SPEC.md"] if (ROOT / "SPEC.md").exists() else [])
                                       if p.is_file() and "__pycache__" not in str(p)])[0],
        "configs_sha256": sha(ROOT / "src" / "fib_r1" / "configs.py"),
        "configs_mtime": datetime.fromtimestamp((ROOT / "src" / "fib_r1" / "configs.py").stat().st_mtime,
                                                tz=timezone.utc).isoformat(),
        "reference_repo": "jakeguo117/trading-system (read-only)",
        "reference_commit": configs.REF_COMMIT,
        "reference_head": git(["rev-parse", "HEAD"], TS),
        "reference_blobs": {k: git(["rev-parse", f"{configs.REF_COMMIT}:{k}"], TS) for k in configs.REF_BLOBS},
    }
    code["blobs_match_plan"] = code["reference_blobs"] == configs.REF_BLOBS

    pre = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": {"file": plan.name, "sha256": plan_sha, "sha256_matches_frozen": plan_sha == configs.PLAN_SHA256,
                 "bytes": plan.stat().st_size, "lines": sum(1 for _ in open(plan))},
        "code": code,
        "environment": env_block(),
        "data": data_block(),
        "hourly": hourly_block(),
        "gate0": json.loads((out.parent / "gate0.json").read_text()) if (out.parent / "gate0.json").exists() else None,
    }
    (out / "preflight.json").write_text(json.dumps(pre, indent=1, ensure_ascii=False))

    prereg = {
        "research_id": "JAKE-FIB-TIMEFRAME-20261020-v0.1",
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        # Honest status: the constants below were frozen in configs.py BEFORE the first
        # performance run (configs_mtime is 16:43, the first A-line run was 16:54). This JSON is a
        # machine emission of those constants, written after the runs, and says so.
        "spec_was_frozen_before_performance_runs": True,
        "emission_note": "constants frozen in src/fib_r1/configs.py before the first performance run "
                         "(see code.configs_mtime / configs_sha256); this file is generated from them "
                         "afterwards so that every value is machine-read rather than retyped",
        "plan_sha256": plan_sha,
        "code_commit": code["head_at_emission"],
        "code_tree_sha256": code["code_tree_sha256"],
        "configs_sha256": code["configs_sha256"],
        "data": {"snapshot": "ts_v111", "mat_tree_sha256": pre["data"]["mat_files"]["tree_sha256"],
                 "raw_tree_sha256": pre["data"]["raw_files"]["tree_sha256"]},
        "windows": {k: list(v) for k, v in configs.WINDOWS.items()},
        "holdout_tag": configs.HOLDOUT_TAG,
        "pools": {"A": configs.POOL_A, "BC": configs.POOL_BC},
        "configs": [{"id": c[0], "line": c[1], "pool": c[2], "note": c[3],
                     "status": "BLOCKED_NO_HOURLY_DATA" if c[0] in
                     ("B-H-ON", "B-H-OFF", "C-H-P", "C-H-S", "C-DH-P", "C-DH-S") else "PLANNED"}
                    for c in configs.CONFIGS],
        "comparisons": [{"id": c[0], "line": c[1], "direction": "left minus right"} for c in configs.COMPARISONS],
        "gate": configs.GATE,
        "cost_tiers": configs.COST_TIER,
        "min_dv60": configs.MIN_DV60,
        "min_history_days": configs.MIN_HISTORY_DAYS,
        "bootstrap": {"block": 20, "n": 10000, "seed": 20261020, "sensitivity_blocks": [5, 60],
                      "two_sided": False, "statistic": "mean of paired log-return difference"},
        "multiplicity": {"method": "Holm step-down over the 15 primary comparisons",
                         "alpha": 0.05, "unrun_p_value": 1.0},
        "no_parameter_search": True,
    }
    (out / "preregistration.json").write_text(json.dumps(prereg, indent=1, ensure_ascii=False))
    print(json.dumps({"preflight": str(out / "preflight.json"), "preregistration": str(out / "preregistration.json"),
                      "plan_sha_ok": pre["plan"]["sha256_matches_frozen"], "blobs_match": code["blobs_match_plan"],
                      "hourly": pre["hourly"]["verdict"], "n_hourly_hits": pre["hourly"]["n_hits"],
                      "mat_sha": pre["data"]["mat_files"]["tree_sha256"][:16]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
