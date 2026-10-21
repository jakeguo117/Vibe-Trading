#!/usr/bin/env python3
"""Build aligned daily panels (2014-01-01 .. 2026-10-20) from the local ts_live parquet store.

Output: runs/<run_id>/data/panel.npz with date-major matrices.
Raw store: /Users/jake/projects/trading/data/ts_live/raw/*.parquet  (read-only use)
"""
import os, sys, time, json
import duckdb
import numpy as np
import pandas as pd

SRC = "/Users/jake/projects/trading/data/ts_live/raw/*.parquet"
UNIV = "/Users/jake/projects/trading/data/ts_live/symbols/universe.csv"
START = "2009-01-01"
END = "2026-10-20"


def main(outdir):
    t0 = time.time()
    con = duckdb.connect()
    df = con.execute(f"""
        select sym, date, Open as o, High as h, Low as l, Close as c,
               "Adj Close" as ac, Volume as v, Dividends as div, "Stock Splits" as split
        from '{SRC}'
        where date >= date '{START}' and date <= date '{END}'
        order by date, sym
    """).df()
    print("rows", len(df), "load_s", round(time.time() - t0, 1))

    dates = np.sort(df["date"].unique())
    syms = np.sort(df["sym"].unique())
    didx = pd.Series(np.arange(len(dates)), index=dates)
    sidx = pd.Series(np.arange(len(syms)), index=syms)
    di = didx.reindex(df["date"]).to_numpy()
    si = sidx.reindex(df["sym"]).to_numpy()

    D, N = len(dates), len(syms)
    shape = (D, N)
    o = np.full(shape, np.nan, dtype=np.float32)
    h = np.full(shape, np.nan, dtype=np.float32)
    lo = np.full(shape, np.nan, dtype=np.float32)
    c = np.full(shape, np.nan, dtype=np.float32)
    ac = np.full(shape, np.nan, dtype=np.float32)
    v = np.full(shape, np.nan, dtype=np.float32)
    dv = np.full(shape, np.nan, dtype=np.float64)
    split = np.zeros(shape, dtype=np.float32)
    div = np.zeros(shape, dtype=np.float32)

    o[di, si] = df["o"].to_numpy(np.float32)
    h[di, si] = df["h"].to_numpy(np.float32)
    lo[di, si] = df["l"].to_numpy(np.float32)
    c[di, si] = df["c"].to_numpy(np.float32)
    ac[di, si] = df["ac"].to_numpy(np.float32)
    v[di, si] = df["v"].to_numpy(np.float32)
    dv[di, si] = (df["c"].to_numpy(np.float64) * df["v"].to_numpy(np.float64))
    split[di, si] = df["split"].to_numpy(np.float32)
    div[di, si] = df["div"].to_numpy(np.float32)

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        os.path.join(outdir, "panel.npz"),
        dates=dates.astype("datetime64[D]").astype(str),
        syms=syms.astype(str),
        open=o, high=h, low=lo, close=c, adjclose=ac, volume=v, dollarvol=dv,
        split=split, dividend=div,
    )
    meta = {
        "source": SRC, "universe_csv": UNIV, "start": START, "end": END,
        "n_dates": int(D), "n_syms": int(N), "rows": int(len(df)),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows_with_nan_close": int(np.isnan(c).sum()),
    }
    json.dump(meta, open(os.path.join(outdir, "panel_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print("elapsed_s", round(time.time() - t0, 1))


if __name__ == "__main__":
    main(sys.argv[1])
