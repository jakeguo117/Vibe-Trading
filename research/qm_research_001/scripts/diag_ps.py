import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from qm.core import load_kit, ps_conditions, ps_signals

k = load_kit('runs/qm-001-20261022-0041/data/panel.npz')
d0, d1 = k.didx('2016-01-04'), k.didx('2026-10-20')
idx = pd.DatetimeIndex(k.dates); cols = k.syms
cdf = pd.DataFrame(k.c, index=idx, columns=cols)
s10 = pd.DataFrame(k.sma10, index=idx, columns=cols)
r10 = cdf / cdf.shift(10) - 1
up = (cdf > cdf.shift(1)) & (cdf.shift(1) > cdf.shift(2)) & (cdf.shift(2) > cdf.shift(3))
ext = (cdf / s10 - 1) >= 0.25
base = pd.DataFrame(k.base_ok, index=idx, columns=cols)
for nm, m in [('r10>=.5', (r10 >= .5) & base), ('+3up', (r10 >= .5) & up & base), ('+ext', (r10 >= .5) & up & ext & base)]:
    print(nm, int(m.to_numpy()[d0:d1+1].sum()))
mask = ps_conditions(k)
print('mask', int(mask.to_numpy()[d0:d1+1].sum()))
weak = (k.c < k.o) & (k.c < (k.h + k.l + k.c) / 3.0) & (k.c < np.vstack([np.full((1, k.n), np.nan), k.c[:-1]]))
print('weak', int(weak[d0:d1+1].sum()))
trig = ps_signals(k, mask)
print('trig', int(trig[d0:d1+1].sum()))
dd, jj = np.nonzero(trig[d0:d1+1]); dd += d0
if len(dd):
    print('stop nan', int((~np.isfinite(k.l[dd, jj])).sum()), 'atr nan', int((~np.isfinite(k.atr14[dd-1, jj])).sum()))
    print(list(zip(k.syms[jj[:10]], k.dates[dd[:10]])))
