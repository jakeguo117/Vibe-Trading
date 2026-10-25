"""Step-3 combinations. At most 8. Uses the same scanner as the sweep."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_run import FM, DATA, OUT, ensure_ma, one, pack_row
import pandas as pd

BASE = {"K": 3, "reserve": 0.30, "gross_cap": 0.70, "vt": 0.16, "ma": 30, "cut": 0.50, "span": 32}
COMBOS = [
    ("combo_cap80_span64", {**BASE, "reserve": 0.20, "gross_cap": 0.80, "span": 64}),
    ("combo_k2_span64", {**BASE, "K": 2, "span": 64}),
    ("combo_k2_cap80", {**BASE, "K": 2, "reserve": 0.20, "gross_cap": 0.80}),
    ("combo_ma40_cap80", {**BASE, "ma": 40, "reserve": 0.20, "gross_cap": 0.80}),
    ("combo_k2_ma40", {**BASE, "K": 2, "ma": 40}),
    ("combo_span64_vt18", {**BASE, "span": 64, "vt": 0.18}),
    ("combo_k2_cut25", {**BASE, "K": 2, "cut": 0.25}),
    ("combo_k2_cap80_span64", {**BASE, "K": 2, "reserve": 0.20, "gross_cap": 0.80, "span": 64}),
]

def main():
    D = FM(str(DATA), processes=2)
    long_a = str(pd.Timestamp(D.cal[1]).date())
    ensure_ma(D, 40, processes=2)
    rows = []
    vol_cache = {}
    for name, spec in COMBOS:
        print("RUN", name, flush=True)
        df2019, dflong = one(D, spec, vol_cache, long_a)
        rows.append(pack_row(name, "combo", spec, df2019, dflong, long_a))
        df2019.to_csv(OUT / f"equity_{name}.csv", float_format="%.10g")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "combos.csv", index=False)
    b = pd.read_csv(OUT / "sweep.csv")
    base = b[b.name == "v1.1.1"].iloc[0]
    print(json.dumps({
        "n": len(out),
        "oos_cagr_min": float(base["2023_plus_cagr"]),
        "oos_dd_min": float(base["2023_plus_maxdd"]),
    }), flush=True)
    for _, r in out.iterrows():
        oos_ok = r["2023_plus_cagr"] >= base["2023_plus_cagr"] - 1e-12 and r["2023_plus_maxdd"] >= base["2023_plus_maxdd"] - 1e-12
        long_ok = r["long_maxdd"] >= -0.30
        print(json.dumps({
            "name": r["name"],
            "is_cagr": r["2019_2022_cagr"],
            "oos_cagr": r["2023_plus_cagr"],
            "oos_dd": r["2023_plus_maxdd"],
            "long_cagr": r["long_cagr"],
            "long_dd": r["long_maxdd"],
            "oos_ok": bool(oos_ok),
            "long_ok": bool(long_ok),
            "both": bool(oos_ok and long_ok),
        }), flush=True)

if __name__ == "__main__":
    main()
