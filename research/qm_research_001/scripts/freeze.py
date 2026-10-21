import hashlib, json, os, sys, time

RUN = 'runs/qm-001-20261022-0041'
SRC = 'src'
sys.path.insert(0, 'scripts')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


ch = hashlib.sha256()
for root, _, files in sorted(os.walk(SRC)):
    for fn in sorted(files):
        if fn.endswith('.py'):
            ch.update(open(os.path.join(root, fn), 'rb').read())
code_hash = ch.hexdigest()[:16]
data_hash = sha(f'{RUN}/data/panel.npz')[:16]
ev_hash = sha(f'{RUN}/data/edgar_8k202.csv')[:16] if os.path.exists(f'{RUN}/data/edgar_8k202.csv') else None

SEGS = {"DEV": ["2010-01-04", "2015-12-31"], "CONT": ["2016-01-04", "2026-10-20"],
        "S1": ["2016-01-04", "2019-12-31"], "S2": ["2020-01-02", "2022-12-30"],
        "S3": ["2023-01-03", "2026-10-20"]}

BO = [("BO-MAIN", {}, {}), ("BO-C1", {"consolidation": False}, {}),
      ("BO-C2", {"leader": False, "prior_up": False}, {}),
      ("BO-C3", {}, {"no_partial": True, "no_breakeven": True, "no_sma_exit": True, "max_hold": 20}),
      ("BO-P01", {"prior_up_min": 0.24}, {}), ("BO-P02", {"prior_up_min": 0.36}, {}),
      ("BO-P03", {"ratio_max": 0.48}, {}), ("BO-P04", {"ratio_max": 0.72}, {}),
      ("BO-P05", {}, {"sma_len": 8}), ("BO-P06", {}, {"sma_len": 12}),
      ("BO-P07", {}, {"risk": 0.004}), ("BO-P08", {}, {"risk": 0.006})]

PS = [("PS-MAIN", {}, {}), ("PS-C1", {"wait": False}, {}),
      ("PS-C2", {"r10_min": 0.5, "drop_extra": True}, {}),
      ("PS-C3", {}, {"no_sma_exit": True}),
      ("PS-P01", {"r10_min": 0.40}, {}), ("PS-P02", {"r10_min": 0.60}, {}),
      ("PS-P03", {"ext_min": 0.20}, {}), ("PS-P04", {"ext_min": 0.30}, {}),
      ("PS-P05", {}, {"ps_hold": 4}), ("PS-P06", {}, {"ps_hold": 6}),
      ("PS-P07", {}, {"risk": 0.004}), ("PS-P08", {}, {"risk": 0.006})]

EP = [("EP-MAIN", {}, {}), ("EP-C1", {"vol_mult": 0.0}, {}),
      ("EP-C3", {}, {"no_sma_exit": True, "max_hold": 20}),
      ("EP-P01", {"gap_min": 0.08}, {}), ("EP-P02", {"gap_min": 0.12}, {}),
      ("EP-P03", {"vol_mult": 0.8}, {}), ("EP-P04", {"vol_mult": 1.2}, {}),
      ("EP-P05", {}, {"sma_len": 16}), ("EP-P06", {}, {"sma_len": 24}),
      ("EP-P07", {}, {"risk": 0.004}), ("EP-P08", {}, {"risk": 0.006})]

configs = {"QM-BO-001": BO, "QM-EP-001": EP, "QM-PS-001": PS}
evals = []
for strat, lst in configs.items():
    for name, mp, ep in lst:
        for seg in SEGS:
            evals.append({"eval_id": f"{strat}|{name}|{seg}|slip10", "strategy": strat, "config": name,
                          "segment": seg, "scenario": "base_10bps"})
for strat in ("QM-BO-001", "QM-EP-001", "QM-PS-001"):
    for bps in (25, 50):
        for seg in ["CONT", "S1", "S2", "S3"]:
            evals.append({"eval_id": f"{strat}|MAIN|{seg}|slip{bps}", "strategy": strat, "config": "MAIN",
                          "segment": seg, "scenario": f"stress_{bps}bps"})
for rate in (10, 30, 100):
    for seg in ["CONT", "S1", "S2", "S3"]:
        evals.append({"eval_id": f"QM-PS-001|MAIN|{seg}|borrow{rate}", "strategy": "QM-PS-001",
                      "config": "MAIN", "segment": seg, "scenario": f"borrow_{rate}pct"})

cfg = {
    "run_id": "qm-research-001-20261022",
    "spec": "Qullamaggie_Backtest_Spec_v1.0_20261022.md",
    "spec_sha256": sha('spec/Qullamaggie_Backtest_Spec_v1.0_20261022.md'),
    "code_hash": code_hash,
    "data_hash_panel": data_hash,
    "data_hash_edgar": ev_hash,
    "account": {"initial_equity": 16000, "risk_per_trade": 0.005, "max_notional_weight": 0.30,
                "max_positions": 1, "commission_per_share": 0.005, "commission_min": 1.0,
                "slippage_bps_per_side": 10, "cash_yield": 0.0},
    "segments": SEGS,
    "data_cutoff": "2026-10-20",
    "configs": {k: [{"name": n, "screen": m, "engine": e} for n, m, e in v] for k, v in configs.items()},
    "missing_configs": [{"strategy": "QM-EP-001", "config": "EP-C2",
                         "reason": "NOT_RUN_DATA_GAP: no point-in-time revenue/EPS/guidance panel"}],
    "evaluation_count": len(evals),
    "evaluations": evals,
    "proxy_variants": {
        "QM-BO-001": "QM-BO-EOD-DIAG",
        "QM-EP-001": "QM-EP-EOD-DIAG",
        "QM-PS-001": "QM-PS-EOD-DIAG / PS-PRICE",
    },
    "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
os.makedirs(f'{RUN}/config', exist_ok=True)
json.dump(cfg, open(f'{RUN}/config/config.json', 'w'), indent=1, ensure_ascii=False)
json.dump({"evaluations": evals, "count": len(evals), "code_hash": code_hash,
           "data_hash": data_hash, "frozen_at": cfg["frozen_at"]},
          open(f'{RUN}/config/manifest.json', 'w'), indent=1)
print("code", code_hash, "data", data_hash, "edgar", ev_hash, "evals", len(evals))
