"""Pure tests for the PB-AV filter and same-bar review. No network."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_av_lib as L


def test_forbid_all_symbols():
    for bad in ("ALL_SYMBOLS", "all_symbols", "*"):
        try:
            L.map_symbol_yahoo_to_databento(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    try:
        L.assert_one_calendar_day("2024-06-03", "2024-06-05", ["AAPL"])
    except ValueError:
        pass
    else:
        raise AssertionError("multi-day request was accepted")
    try:
        L.assert_one_calendar_day("2024-06-03", "2024-06-04", ["ALL_SYMBOLS"])
    except ValueError:
        pass
    else:
        raise AssertionError("ALL_SYMBOLS was accepted")


def test_class_share_map():
    assert L.map_symbol_yahoo_to_databento("BF-B") == "BF.B"
    assert L.map_symbol_yahoo_to_databento("AAPL") == "AAPL"
    assert L.assert_one_calendar_day("2024-06-03", "2024-06-04", ["BF.B"]) == ["BF.B"]


def test_rth_and_avwap():
    idx = pd.DatetimeIndex(
        [
            "2024-06-03 13:29:00+00:00",  # 09:29 ET EDT, excluded
            "2024-06-03 13:30:00+00:00",  # 09:30
            "2024-06-03 19:59:00+00:00",  # 15:59
            "2024-06-03 20:00:00+00:00",  # 16:00, excluded
        ]
    )
    mask = L.rth_mask(idx)
    assert mask.tolist() == [False, True, True, False]
    # typical 11 and 12, equal volume -> 11.5
    av, sv, n = L.avwap_from_ohlcv([12, 13], [10, 11], [11, 12], [100, 100])
    assert n == 2 and sv == 200 and abs(av - 11.5) < 1e-12
    av0, sv0, _ = L.avwap_from_ohlcv([10], [9], [9.5], [0])
    assert sv0 == 0 and np.isnan(av0)


def test_same_bar_paths():
    def bars(rows):
        return pd.DataFrame(rows)

    later = bars(
        [
            dict(ts="2024-06-03T13:30:00+00:00", open=9.0, high=9.5, low=8.9, close=9.2),
            dict(ts="2024-06-03T13:31:00+00:00", open=9.8, high=10.2, low=9.7, close=10.1),
            dict(ts="2024-06-03T13:32:00+00:00", open=10.3, high=10.4, low=8.5, close=8.6),
        ]
    )
    r = L.review_same_bar(later, 10.0, 9.0)
    assert r["resolution"] == "entry_then_stop" and r["ambiguous_1m"] == 0

    same = bars([dict(ts="2024-06-03T13:31:00+00:00", open=9.5, high=10.5, low=8.5, close=9.0)])
    r = L.review_same_bar(same, 10.0, 9.0)
    assert r["resolution"] == "same_minute_conservative" and r["ambiguous_1m"] == 1

    before = bars(
        [
            dict(ts="2024-06-03T13:30:00+00:00", open=9.4, high=9.6, low=8.4, close=9.2),
            dict(ts="2024-06-03T13:31:00+00:00", open=9.8, high=10.4, low=9.6, close=10.2),
        ]
    )
    r = L.review_same_bar(before, 10.0, 9.0)
    assert r["resolution"] == "stop_before_entry" and r["ambiguous_1m"] == 0

    # Enter at the open of the bar; the same bar still cannot be ordered.
    gap = bars([dict(ts="2024-06-03T13:30:00+00:00", open=10.2, high=10.4, low=8.8, close=9.1)])
    r = L.review_same_bar(gap, 10.0, 9.0)
    assert r["entry_at_open"] == 1 and r["resolution"] == "same_minute_conservative"


def test_rank_and_split_factor():
    df = pd.DataFrame(
        [
            dict(t="2024-01-02", symbol="A", R63=0.5, ADV20=10, j=2, B=10, S=9),
            dict(t="2024-01-02", symbol="B", R63=0.5, ADV20=10, j=1, B=11, S=10),
            dict(t="2024-01-02", symbol="C", R63=0.4, ADV20=99, j=0, B=12, S=11),
            dict(t="2024-01-03", symbol="D", R63=0.2, ADV20=5, j=3, B=8, S=7),
        ]
    )
    top = L.pick_top_orders(df)
    assert top["symbol"].tolist() == ["B", "D"]
    idx = pd.bdate_range("2024-01-02", periods=4)
    splits = pd.Series([0.0, 2.0, 0.0, 0.0], index=idx)
    f = L.future_split_factor(splits)
    assert list(f.round(6)) == [2.0, 1.0, 1.0, 1.0]


if __name__ == "__main__":
    test_forbid_all_symbols()
    test_class_share_map()
    test_rth_and_avwap()
    test_same_bar_paths()
    test_rank_and_split_factor()
    print("test_pb_av OK")
