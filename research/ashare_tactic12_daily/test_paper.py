import numpy as np
import pandas as pd

from ashare_tactic12_daily import paper
from ashare_tactics.signals import _Book


def _view(close, start="2026-01-01", name="测试"):
    n = len(close); c = np.asarray(close, float)
    d = np.datetime64(start) + np.arange(n)
    b = _Book("sh600000", d, c, c, c, c, np.full(n, 1e4), np.full(n, 1e6), np.ones(n), False)
    from ashare_tactics.indicators import sma
    return paper.SymView("sh600000", name, False, b, sma(b.q_close, 5), sma(b.q_close, 20))


def test_exit_below_ma20_and_max_hold():
    v = _view([10.0] * 60 + [9.0])
    last = str(pd.Timestamp(v.book.dates[-1]).date())
    assert paper.exit_reason(v, last, last) == "收盘跌破20日均线"
    v2 = _view([10.0] * 61)
    first = str(pd.Timestamp(v2.book.dates[10]).date())
    assert "持有满" in paper.exit_reason(v2, first, str(pd.Timestamp(v2.book.dates[-1]).date()))
    assert paper.exit_reason(v2, str(pd.Timestamp(v2.book.dates[-1]).date()), str(pd.Timestamp(v2.book.dates[-1]).date())) is None


def test_suspended_no_signal():
    v = _view([10.0] * 90)
    assert paper.entry_signal(v, "2030-01-01") is False


def test_fills_limit_up_skip_and_cash(tmp_path):
    close = [10.0] * 90 + [11.0]  # 最后一天涨停开盘
    v = _view(close)
    day = str(pd.Timestamp(v.book.dates[-1]).date())
    s = paper.new_state()
    s["pending"] = {"date": day, "sells": [], "buys": [{"symbol": "sh600000", "name": "测试", "shares": 500, "ref": 10.0, "amount": 5000}]}
    notes = paper.apply_fills(s, {"sh600000": v}, day)
    assert s["positions"] == {} and "涨停" in notes[0]
    v2 = _view([10.0] * 91)
    s["pending"] = {"date": day, "sells": [], "buys": [{"symbol": "sh600000", "name": "测试", "shares": 500, "ref": 10.0, "amount": 5000}]}
    paper.apply_fills(s, {"sh600000": v2}, day)
    p = s["positions"]["sh600000"]
    assert p["shares"] == 500 and abs(s["cash"] + p["cost"] - 100000) < 1e-6
    paper.save_state(tmp_path, s); assert paper.load_state(tmp_path)["positions"]["sh600000"]["shares"] == 500


def test_mailer_without_secret_does_not_send(monkeypatch):
    from ashare_tactic12_daily import mailer
    monkeypatch.setattr(mailer, "_secret", lambda: None)
    assert mailer.send_mail("x", "y").startswith("未发送")
