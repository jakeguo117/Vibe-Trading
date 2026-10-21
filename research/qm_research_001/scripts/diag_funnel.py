import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from qm.core import load_kit, bo_conditions

k = load_kit('runs/qm-001-20261022-0041/data/panel.npz')
d0 = k.didx('2016-01-04'); d1 = k.didx('2026-10-20')
idx = pd.DatetimeIndex(k.dates); cols = k.syms
print("base_ok/day:", float(np.mean(k.base_ok[d0:d1+1].sum(axis=1))), float(np.min(k.base_ok[d0:d1+1].sum(axis=1))))
cdf = pd.DataFrame(k.c, index=idx, columns=cols)
s20 = pd.DataFrame(k.sma20, index=idx, columns=cols)
s50 = pd.DataFrame(k.sma50, index=idx, columns=cols)
hdf = pd.DataFrame(k.h, index=idx, columns=cols)
ldf = pd.DataFrame(k.l, index=idx, columns=cols)
leader = (pd.DataFrame(k.p21, index=idx, columns=cols) >= 98) | (pd.DataFrame(k.p63, index=idx, columns=cols) >= 98) | (pd.DataFrame(k.p126, index=idx, columns=cols) >= 98)
prior = (cdf.shift(15) / cdf.shift(63) - 1) >= 0.30
maxh = hdf.rolling(15, min_periods=15).max()
minl = ldf.rolling(15, min_periods=15).min()
tight = (maxh / minl - 1) <= 0.18
r5 = hdf.rolling(5, min_periods=5).max() - ldf.rolling(5, min_periods=5).min()
r10p = hdf.shift(5).rolling(10, min_periods=10).max() - ldf.shift(5).rolling(10, min_periods=10).min()
ratio = (r5 / r10p) <= 0.60
hl = ldf.rolling(5, min_periods=5).min() > ldf.shift(5).rolling(10, min_periods=10).min()
trend = (cdf >= s20) & (s20 >= s50) & (s20 > s20.shift(5))
base = pd.DataFrame(k.base_ok, index=idx, columns=cols)
steps = [("base", base), ("leader", leader & base), ("leader+prior", leader & prior & base),
         ("+tight", leader & prior & tight & base), ("+ratio", leader & prior & tight & ratio & base),
         ("+hl", leader & prior & tight & ratio & hl & base), ("+trend", leader & prior & tight & ratio & hl & trend & base)]
for nm, m in steps:
    print(f"{nm:14s}", int(m.to_numpy()[d0:d1 + 1].sum()))

trig = bo_conditions(k).shift(1, fill_value=False).to_numpy() & (k.c > maxh.shift(1).to_numpy())
dd, jj = np.nonzero(trig[d0:d1 + 1]); dd = dd + d0
print("trig", len(dd))
print("nan r63", int((~np.isfinite(k.r63[dd - 1, jj])).sum()), "nan dv", int((~np.isfinite(k.dv20[dd - 1, jj])).sum()))
print("years", pd.Series(pd.DatetimeIndex(k.dates[dd]).year).value_counts().sort_index().to_dict())
print("uniq syms", len(set(jj)))
