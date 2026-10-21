import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from qm.core import load_kit

k = load_kit('runs/qm-001-20261022-0041/data/panel.npz')
d0 = k.didx('2016-01-04'); d1 = k.didx('2026-10-20')
prev_c = np.vstack([np.full((1, k.c.shape[1]), np.nan, np.float32), k.c[:-1]])
gap = k.o / prev_c - 1
with np.errstate(invalid='ignore'):
    ev = (gap >= 0.10) & k.base_ok
sel = ev[d0:d1 + 1].any(axis=0)
syms = sorted(k.syms[sel])
print("gap-event symbols:", len(syms), "of", len(k.syms), "events:", int(ev[d0:d1 + 1].sum()))
open('runs/qm-001-20261022-0041/data/ep_syms.txt', 'w').write("\n".join(syms) + "\n")
cik = json.load(open('/tmp/cik.json'))
m = {v['ticker']: v['cik_str'] for v in cik.values()}
json.dump(m, open('runs/qm-001-20261022-0041/data/cik_map.json', 'w'))
print("mapped:", sum(1 for s in syms if s in m))
