"""Quick A00 identity probe: research copy (official switches) vs untouched engine."""
import sys, time
sys.path.insert(0, "/Users/jake/projects/trading/trading-system/backtest")
sys.path.insert(0, "/Users/jake/projects/trading/Vibe-Trading/research/fib_timeframe_r1/src")
import numpy as np
from fib_r1 import aport
from upleg_bt.engine.fm import FM
from upleg_bt.engine import portfolio as official
from upleg_bt.paths import data_paths

D = FM(data_paths("/Users/jake/projects/trading/data/ts_v111"), processes=6)
print("FM loaded", D.O.shape, D.cal[0], D.cal[-1], flush=True)
t0 = time.time(); a = aport.run_portfolio(D, g_mode="official", rank="hist", trace=True)
print("aport A00 %.1fs" % (time.time() - t0), flush=True)
t0 = time.time(); b = official.run_portfolio(D)
print("official %.1fs" % (time.time() - t0), flush=True)
d = np.abs(a["eq"]["equity"].values - b["eq"]["equity"].values)
print("index_match", bool((a["eq"].index == b["eq"].index).all()))
print("maxabs_equity", float(d.max()), "n_days", len(d))
print("final_eq aport %.12f official %.12f" % (a["eq"]["equity"].iloc[-1], b["eq"]["equity"].iloc[-1]))
print("events_equal", a["events"].equals(b["events"]))
print("weights_equal", a["weights"].equals(b["weights"]) if "weights" in b else "n/a")
print("PASS", bool(d.max() < 1e-12))
