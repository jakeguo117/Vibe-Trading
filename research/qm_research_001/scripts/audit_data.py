import json, os, subprocess, sys
import numpy as np, pandas as pd

sys.path.insert(0, 'src')
from qm.core import load_kit

RUN = 'runs/qm-001-20261022-0041'
k = load_kit(f'{RUN}/data/panel.npz')
T, N = k.c.shape
years = pd.DatetimeIndex(k.dates).year
prev_c = np.vstack([np.full((1, N), np.nan, np.float32), k.c[:-1]])
gap = k.o / prev_c - 1
trig_bo = None

rows = []
for y in sorted(set(years)):
    m = years == y
    rows.append({
        'year': int(y),
        'trading_days': int(m.sum()),
        'symbols_with_any_data': int(np.isfinite(k.c[m]).any(axis=0).sum()),
        'avg_symbols_base_eligible': round(float(k.base_ok[m].sum(axis=1).mean()), 1),
        'split_events': int((np.isfinite(k.div[m]) | True).sum() * 0),  # filled below
        'gap_ge_10pct_events': int(((gap[m] >= 0.10) & k.base_ok[m]).sum()),
    })
cov = pd.DataFrame(rows)
# split / dividend events need the raw panel columns
z = np.load(f'{RUN}/data/panel.npz')
sp = z['split']; dv = z['dividend']
for i, y in enumerate(cov['year']):
    m = years == y
    cov.loc[i, 'split_events'] = int((sp[m] > 0).sum())
    cov.loc[i, 'dividend_events'] = int((dv[m] > 0).sum())
    cov.loc[i, 'minute_bars_available'] = 0
cov.to_csv(f'{RUN}/data/coverage.csv', index=False)

audit = {
    'daily_source': '/Users/jake/projects/trading/data/ts_live/raw/*.parquet (local, read-only)',
    'daily_rows_2009_2026': int(z['close'].shape[0] * z['close'].shape[1]),
    'symbols_total': int(z['syms'].shape[0]),
    'symbols_in_kit': int(N),
    'date_range': [str(k.dates[0]), str(k.dates[-1])],
    'symbols_alive_at_end': int(np.isfinite(k.c[-1]).sum()),
    'symbols_not_alive_at_end': int((~np.isfinite(k.c[-1])).sum()),
    'split_events_total': int((sp > 0).sum()),
    'dividend_events_total': int((dv > 0).sum()),
    'gap_ge_10_events_all': int(((gap >= 0.10) & k.base_ok).sum()),
    'avg_base_eligible_per_day': float(k.base_ok.sum(axis=1).mean()),
}
json.dump(audit, open(f'{RUN}/data/audit_summary.json', 'w'), indent=1)
print(json.dumps(audit, indent=1))
print(cov.to_string(index=False))
