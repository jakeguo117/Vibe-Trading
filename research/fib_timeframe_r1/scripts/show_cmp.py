import sys
import pandas as pd
pd.set_option("display.width", 200)
df = pd.read_csv(sys.argv[1])
cols = [c for c in ("comparison", "cagr_diff_pp", "p_raw", "p_holm_threshold", "holm_reject",
                    "trades_left", "trades_right", "segments_positive", "cagr_diff_pp_cost_x2",
                    "maxdd_left", "maxdd_right", "sharpe_left", "sharpe_right", "status") if c in df]
print(df[cols].to_string(index=False))
