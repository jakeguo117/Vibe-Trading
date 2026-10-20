import json, sys
from pathlib import Path
import pandas as pd
pd.set_option("display.width", 250)
R = Path(sys.argv[1])
res = json.loads((R / "result.json").read_text())
rows = []
for k, v in res["configs"].items():
    if v.get("status") != "DONE":
        rows.append({"id": k, "status": v["status"]})
        continue
    t = v.get("trades") or {}
    e = v.get("entry") or {}
    rows.append({"id": k, "cagr": round(v["cagr"] * 100, 2), "maxdd": round(v["maxdd"] * 100, 2),
                 "sharpe": round(v["sharpe"], 3), "sortino": round(v.get("sortino") or 0, 3),
                 "vol": round(v["ann_vol"] * 100, 2), "expo": round(v["avg_exposure"] * 100, 1),
                 "trades": t.get("n_trades"), "win": round((t.get("win_rate") or 0) * 100, 1),
                 "pf": round(t.get("profit_factor") or 0, 2), "exp_net": round(t.get("expectancy") or 0, 4),
                 "mae_p90": round(e.get("mae_p90_loss") or 0, 4), "adv1d": round(e.get("adverse_rate_1d") or 0, 3),
                 "n_adv": e.get("n_mae_1d"), "fail_bo": round(e.get("failed_breakout_rate") or 0, 3),
                 "status": "DONE"})
print(pd.DataFrame(rows).to_string(index=False))
print()
print("cancellation:", json.dumps(res["cancellation"], indent=1)[:900])
print()
print("identity:", res["identity"], "divergence:", json.dumps(res["divergence"], indent=1)[:600])
