"""Resume checkpoint from files already on disk. Does not call Databento and does not read API keys."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

G = Path("/workspace/gpt-candidates")
OUT = G / "out"
MIN_DIR = G / "data" / "minutes"
MANIFEST = OUT / "pb_av_vs_pb_manifest.json"
PULL = OUT / "pb_minute_ticker_days_2023plus.csv"


def write_checkpoint() -> dict:
    pull = pd.read_csv(PULL, dtype=str)
    pull["date"] = pull["date"].str.slice(0, 10)
    pull["symbol"] = pull["symbol"].astype(str)
    on_disk = {}
    for path in sorted(MIN_DIR.glob("*.parquet")):
        if path.name.startswith("."):
            continue
        df = pd.read_parquet(path, columns=["symbol"])
        on_disk[path.stem] = set(df["symbol"].astype(str).unique()) if len(df) else set()
    pulled_dates = set(on_disk)
    pull["downloaded"] = pull["date"].isin(pulled_dates)
    pull["has_rth_bars"] = [
        int(sym in on_disk.get(date, ())) for sym, date in zip(pull.symbol, pull.date)
    ]
    got = pull[pull.downloaded][["symbol", "date", "has_rth_bars"]]
    left = pull[~pull.downloaded][["symbol", "date"]]
    got_path = OUT / "pb_av_downloaded_ticker_days.csv"
    left_path = OUT / "pb_av_remaining_ticker_days.csv"
    got.to_csv(got_path, index=False)
    left.to_csv(left_path, index=False)
    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    spent = manifest.get("cum_downloaded_usd")
    ck = {
        "manifest_path": str(MANIFEST),
        "downloaded_list": str(got_path),
        "remaining_list": str(left_path),
        "n_pull_list": int(len(pull)),
        "n_downloaded_ticker_days": int(len(got)),
        "n_downloaded_with_rth_bars": int(got.has_rth_bars.sum()) if len(got) else 0,
        "n_remaining_ticker_days": int(len(left)),
        "n_dates_on_disk": len(on_disk),
        "credit_spent_usd": spent,
        "credit_quoted_avwap_usd": manifest.get("cum_quoted_avwap_usd"),
        "stop_reason": manifest.get("stop_reason"),
        "next_pull": "pb_av_remaining_ticker_days.csv is the only minute list still unpaid. Do not re-request dates already on disk.",
    }
    (OUT / "pb_av_resume_checkpoint.json").write_text(json.dumps(ck, indent=1))
    if manifest:
        manifest["resume_checkpoint"] = ck
        tmp = MANIFEST.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(MANIFEST)
    print(
        f"checkpoint downloaded={ck['n_downloaded_ticker_days']} remaining={ck['n_remaining_ticker_days']} spent={spent}"
    )
    return ck


if __name__ == "__main__":
    write_checkpoint()
