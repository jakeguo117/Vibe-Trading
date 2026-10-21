import json, sys, urllib.request
import numpy as np, pandas as pd

sys.path.insert(0, 'src')
from qm.core import load_kit

UA = {"User-Agent": "Mozilla/5.0"}
k = load_kit('runs/qm-001-20261022-0041/data/panel.npz')
out = []
for sym in ["AAPL", "MSFT", "AEIS"]:
    j = int(np.where(k.syms == sym)[0][0])
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5y&interval=1d&events=div%2Csplit"
    req = urllib.request.Request(url, headers=UA)
    d = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
    ts = pd.to_datetime(d["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date
    y = pd.DataFrame({"date": ts, "close_y": d["indicators"]["quote"][0]["close"],
                      "vol_y": d["indicators"]["quote"][0]["volume"]}).dropna()
    p = pd.DataFrame({"date": pd.DatetimeIndex(k.dates).date, "close_p": k.cr[:, j], "vol_p": k.v[:, j]})
    m = y.merge(p, on="date", how="inner")
    m = m[m.date >= pd.Timestamp("2021-10-01").date()]
    rd = (m["close_y"] - m["close_p"]).abs()
    rv = (m["vol_y"] - m["vol_p"]).abs() / m["vol_p"].replace(0, np.nan)
    rec = {"sym": sym, "overlap_days": int(len(m)), "max_abs_close_diff": float(rd.max()),
           "mean_abs_close_diff": float(rd.mean()), "max_rel_vol_diff": float(rv.max()),
           "first": str(m.date.min()), "last": str(m.date.max())}
    out.append(rec); print(rec)
json.dump(out, open('runs/qm-001-20261022-0041/data/sample_verification.json', 'w'), indent=1)
