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


def test_env_config_paths_and_defaults(tmp_path):
    c = paper.config_from_env({"STATE_DIR": str(tmp_path / "s"), "CACHE_DIR": str(tmp_path / "c"), "DRY_RUN": "1",
                               "MIN_COVERAGE": "0.5", "GITHUB_ACTIONS": "true"})
    assert c["state_dir"] == tmp_path / "s" and c["cache_dir"] == tmp_path / "c"
    assert c["dry_run"] and c["min_coverage"] == 0.5 and c["quiet"]
    d = paper.config_from_env({})
    assert str(d["state_dir"]) == paper.MAC_STATE_DIR and not d["dry_run"] and not d["quiet"]


def test_mail_config_and_secret_from_env(monkeypatch):
    from ashare_tactic12_daily import mailer
    c = mailer.mail_config({"GMAIL_USER": "a@x.com", "MAIL_TO": "b@y.com, c@z.com", "STATE_DIR": "/tmp/s"})
    assert c["sender"] == "a@x.com" and c["to"] == ["b@y.com", "c@z.com"]
    assert str(c["secret_file"]) == "/tmp/s/.secrets/gmail_app_password"
    assert mailer.mail_config({})["to"] == ["394592848@qq.com"]
    assert mailer._secret({"GMAIL_APP_PASSWORD": "pw"}) == "pw"
    called = []
    monkeypatch.setattr(mailer.subprocess, "run", lambda *a, **k: called.append(a))
    assert mailer._secret({"STATE_DIR": "/nonexistent"}, platform="linux") is None and called == []  # 非 mac 不碰钥匙串


def _patch_run(monkeypatch, cal):
    monkeypatch.setattr(paper, "trade_dates", lambda: cal)
    monkeypatch.setattr(paper, "load_universe", lambda: (_ for _ in ()).throw(AssertionError("should not fetch")))


def test_idempotent_skip_when_marker_exists(tmp_path, monkeypatch):
    _patch_run(monkeypatch, ["2026-10-19", "2026-10-20"])
    paper.mark_sent(tmp_path, "2026-10-20")
    cfg = paper.config_from_env({"STATE_DIR": str(tmp_path)})
    assert paper.run("2026-10-20", tmp_path, False, True, False, False, cfg) == ""


def test_non_trading_day_skips(tmp_path, monkeypatch):
    _patch_run(monkeypatch, ["2026-10-16", "2026-10-19"])
    cfg = paper.config_from_env({"STATE_DIR": str(tmp_path)})
    assert paper.run("2026-10-17", tmp_path, False, False, False, False, cfg) == ""


def test_rerun_resends_existing_sheet_without_rebooking(tmp_path, monkeypatch):
    _patch_run(monkeypatch, ["2026-10-19", "2026-10-20"])
    s = paper.new_state(); s["start_date"] = "2026-10-01"; s["pending"] = {"date": "2026-10-20", "sells": [], "buys": []}
    paper.save_state(tmp_path, s)
    (tmp_path / "sheets").mkdir(); (tmp_path / "sheets" / "order_sheet_2026-10-20.txt").write_text("SHEET", "utf-8")
    import mailer as m  # paper 运行时按脚本目录导入
    sent = []
    monkeypatch.setattr(m, "send_mail", lambda subj, body: sent.append(body) or "已发送")
    cfg = paper.config_from_env({"STATE_DIR": str(tmp_path)})
    assert paper.run("2026-10-20", tmp_path, False, True, False, False, cfg) == "SHEET"
    assert sent == ["SHEET"] and paper.already_sent(tmp_path, "2026-10-20")
    assert paper.load_state(tmp_path)["history"] == []  # 未重复记账
    # DRY_RUN 下即使带 --send 也不发送、不写标记
    d = tmp_path / "dry"; paper.save_state(d, s)
    (d / "sheets").mkdir(); (d / "sheets" / "order_sheet_2026-10-20.txt").write_text("SHEET", "utf-8")
    sent.clear()
    paper.run("2026-10-20", d, False, True, False, False, paper.config_from_env({"DRY_RUN": "1"}))
    assert sent == [] and not paper.already_sent(d, "2026-10-20")


def test_quality_gate():
    import pytest
    v = _view([10.0] * 90)
    last = str(pd.Timestamp(v.book.dates[-1]).date())
    assert "覆盖率" in paper.quality_check(1, {"sh600000": v}, last, {}, 0.85, 1)
    with pytest.raises(paper.QualityError):
        paper.quality_check(10, {"sh600000": v}, last, {}, 0.85, 1)
    with pytest.raises(paper.QualityError):
        paper.quality_check(1, {"sh600000": v}, last, {"sz000001": {}}, 0.85, 1)


def test_snapshot_append_and_exrights(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "CACHE", tmp_path)
    (tmp_path / "bars").mkdir()
    f = pd.DataFrame({"date": pd.to_datetime(["2026-10-15", "2026-10-16"]), "open": [10.0, 10.0], "close": [10.0, 10.5],
                      "high": [10.0, 10.6], "low": [10.0, 9.9], "volume": [100.0, 100.0], "amount": [1e5, 1e5], "factor": [1.2, 1.2]})
    f.to_parquet(tmp_path / "bars" / "sh600000.parquet", index=False)
    row = {"open": 10.6, "high": 11.0, "low": 10.4, "close": 10.8, "prev_close": 10.5, "volume": 50.0, "amount": 5e4}
    assert paper.apply_snapshot("sh600000", {**row, "prev_close": 9.0}, "2026-10-19", "2026-10-16") == "need_fetch"  # 除权
    assert paper.apply_snapshot("sh600000", {**row, "volume": 0}, "2026-10-19", "2026-10-16") == "suspended"
    assert paper.apply_snapshot("sh600000", row, "2026-10-19", "2026-10-16") == "snap"
    g = pd.read_parquet(tmp_path / "bars" / "sh600000.parquet")
    assert len(g) == 3 and g["factor"].iloc[-1] == 1.2 and g["close"].iloc[-1] == 10.8
    assert paper.apply_snapshot("sh600000", row, "2026-10-19", "2026-10-16") == "fresh"
    assert paper.apply_snapshot("sh600001", row, "2026-10-19", "2026-10-16") == "need_fetch"


def test_snapshot_only_before_auction():
    tz = "Asia/Shanghai"
    assert paper.snapshot_usable("2026-10-19", pd.Timestamp("2026-10-19 08:30", tz=tz))
    assert not paper.snapshot_usable("2026-10-19", pd.Timestamp("2026-10-19 10:00", tz=tz))


def test_symbol_and_universe_frame_exclude_st_and_b_shares():
    assert paper.symbol_from_code("600000") == "sh600000"
    assert paper.symbol_from_code("000001") == "sz000001"
    assert paper.symbol_from_code("688001") == "sh688001"
    assert paper.symbol_from_code("830799") == "bj830799"
    assert paper.symbol_from_code("bj920196") == "bj920196"
    assert paper.symbol_from_code("900901") is None
    assert paper.symbol_from_code("200001") is None
    frame = paper.universe_from_code_name(
        ["600000", "000001", "688001", "830799", "900901", "200468", "sh601398"],
        ["浦发银行", "平安银行", "*ST海马", "艾融软件", "仪电B股", "宁通信B", "工商银行"],
        "eastmoney_spot",
    )
    assert set(frame["symbol"]) == {"sh600000", "sz000001", "bj830799", "sh601398"}
    assert not frame["name"].str.upper().str.contains("ST").any()
    assert set(frame["source"]) == {"eastmoney_spot"}


def _reset_breakers():
    paper._BREAKER_FAILS.update({k: 0 for k in paper._BREAKER_FAILS})
    paper._BREAKER_UNTIL.update({k: 0.0 for k in paper._BREAKER_UNTIL})


def test_load_universe_falls_back_to_eastmoney_spot(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "STATE_DIR", tmp_path)
    monkeypatch.setattr(paper, "_UNIVERSE_FALLBACK_MIN", 1)
    import ashare_tactics.download as download
    monkeypatch.setattr(download, "fetch_universe", lambda: (_ for _ in ()).throw(RuntimeError("sina list down")))
    spot = paper.universe_from_code_name(["600000", "000001"], ["浦发银行", "ST岩石"], "eastmoney_spot")
    monkeypatch.setattr(paper, "_universe_from_em_spot", lambda: spot)
    monkeypatch.setattr(paper, "_universe_from_sina_spot", lambda: (_ for _ in ()).throw(AssertionError("sina spot")))
    universe = paper.load_universe()
    assert list(universe["symbol"]) == ["sh600000"]
    assert (tmp_path / "universe_live.parquet").exists()


def test_load_universe_cache_before_network_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "STATE_DIR", tmp_path)
    cached = paper.universe_from_code_name(["601318"], ["中国平安"], "cache")
    paper._write_parquet(cached, tmp_path / "universe_live.parquet")
    import ashare_tactics.download as download
    monkeypatch.setattr(download, "fetch_universe", lambda: (_ for _ in ()).throw(RuntimeError("sina list down")))
    monkeypatch.setattr(paper, "_universe_from_em_spot", lambda: (_ for _ in ()).throw(AssertionError("network")))
    universe = paper.load_universe()
    assert list(universe["symbol"]) == ["sh601318"]


def test_load_universe_raises_when_every_source_fails(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(paper, "STATE_DIR", tmp_path)
    import ashare_tactics.download as download
    monkeypatch.setattr(download, "fetch_universe", lambda: (_ for _ in ()).throw(RuntimeError("sina list down")))
    monkeypatch.setattr(paper, "_universe_from_em_spot", lambda: (_ for _ in ()).throw(RuntimeError("em")))
    monkeypatch.setattr(paper, "_universe_from_sina_spot", lambda: (_ for _ in ()).throw(RuntimeError("sina spot")))
    monkeypatch.setattr(paper, "_universe_from_exchange_lists", lambda: (_ for _ in ()).throw(RuntimeError("exch")))
    with pytest.raises(RuntimeError, match="股票池获取失败"):
        paper.load_universe()


def test_fetch_uses_tencent_after_em_and_sina_fail(monkeypatch):
    _reset_breakers()
    monkeypatch.setattr(paper, "_fetch_em", lambda *a: (_ for _ in ()).throw(RuntimeError("em")))
    monkeypatch.setattr(paper, "_fetch_sina", lambda *a: (_ for _ in ()).throw(RuntimeError("sina")))
    bars = pd.DataFrame({"date": [pd.Timestamp("2026-10-15")], "open": [1.0], "close": [1.0],
                         "high": [1.0], "low": [1.0], "volume": [10.0], "amount": [100.0], "factor": [1.0]})
    monkeypatch.setattr(paper, "_fetch_tencent", lambda *a: bars)
    out = paper._fetch("600000", "20150101", "20261015")
    assert len(out) == 1 and float(out["volume"].iloc[0]) == 10.0
    _reset_breakers()


def test_refresh_stops_at_budget_without_dropping_finished_files(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "CACHE", tmp_path)
    monkeypatch.setattr(paper, "_past_deadline", lambda deadline: True)
    monkeypatch.setattr(paper, "refresh_symbol", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should defer")))
    counts = paper.refresh_all(["sh600000", "sz000001"], "2026-10-15", workers=1, budget_sec=1)
    assert counts == {"deferred": 2}
