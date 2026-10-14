"""Metrics, short-rally capture, charts, and the Chinese write-up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gso.config import (
    CASH_FLOOR,
    CONFIRM_WEEKS,
    COST_BPS,
    DAILY_BARS_PER_WEEK,
    EWMA_SPAN,
    FIB_BANDS,
    FIB_CONFIRM_WEEKS,
    IS_END,
    MAIN_START,
    MAX_HOLDINGS,
    OOS_START,
    RALLY_MAX_WEEKS,
    RALLY_MIN_GAIN,
    RALLY_START,
    SLOPE_LAG_WEEKS,
    SLIPPAGE_BPS,
    STAGE_WEIGHT,
    SUB_2022_START,
    TARGET_VOL,
    TRADING_DAYS_PER_YEAR,
)
from gso.indicators import to_weekly

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def slice_nav(nav: pd.Series, start, end) -> pd.Series:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    out = nav.loc[(nav.index >= start) & (nav.index <= end)]
    if out.empty:
        raise RuntimeError(f"empty nav slice {start.date()} .. {end.date()}")
    return out


def annualized_return(nav: pd.Series) -> float:
    rets = nav.pct_change().dropna()
    if len(rets) < 2 or nav.iloc[0] <= 0:
        return float("nan")
    growth = float(nav.iloc[-1] / nav.iloc[0])
    if growth <= 0:
        return float("nan")
    return growth ** (TRADING_DAYS_PER_YEAR / len(rets)) - 1.0


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min())


def sharpe_sortino(nav: pd.Series) -> tuple[float, float]:
    rets = nav.pct_change().dropna()
    if len(rets) < 2:
        return float("nan"), float("nan")
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1))
    sharpe = mu / sd * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(rets.to_numpy(dtype=float), 0.0) ** 2)))
    sortino = mu / downside * np.sqrt(TRADING_DAYS_PER_YEAR) if downside > 0 else float("nan")
    return sharpe, sortino


def calendar_year_returns(nav: pd.Series, start, end) -> dict[int, float]:
    window = slice_nav(nav, start, end)
    out = {}
    for year in range(int(window.index[0].year), int(window.index[-1].year) + 1):
        this = window[window.index.year == year]
        if this.empty:
            continue
        prev = nav[nav.index < pd.Timestamp(year=year, month=1, day=1)]
        base = float(prev.iloc[-1]) if len(prev) else float(this.iloc[0])
        out[year] = float(this.iloc[-1]) / base - 1.0
    return out


def worst_year(nav: pd.Series, start, end) -> tuple[int, float]:
    years = calendar_year_returns(nav, start, end)
    year = min(years, key=years.get)
    return year, years[year]


def trade_stats(episodes: list[dict], start, end) -> tuple[float, float]:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    chosen = [ep for ep in episodes if start <= pd.Timestamp(ep["entry"]) <= end]
    span_days = max((end - start).days, 1)
    years = span_days / 365.25
    per_year = len(chosen) / years
    if not chosen:
        return per_year, float("nan")
    weeks = []
    for ep in chosen:
        weeks.append((pd.Timestamp(ep["exit"]) - pd.Timestamp(ep["entry"])).days / 7.0)
    return per_year, float(np.mean(weeks))


def metric_row(version: str, freq: str, ma_weeks: int, period: str, nav: pd.Series, episodes: list[dict], start, end) -> dict:
    window = slice_nav(nav, start, end)
    sharpe, sortino = sharpe_sortino(window)
    year, yret = worst_year(nav, window.index[0], window.index[-1])
    trades, hold = trade_stats(episodes, window.index[0], window.index[-1])
    return {
        "版本": version,
        "频率": "周频" if freq == "W" else "日频",
        "均线周数": ma_weeks,
        "区间": period,
        "起始": window.index[0].strftime("%Y-%m-%d"),
        "结束": window.index[-1].strftime("%Y-%m-%d"),
        "年化收益": annualized_return(window),
        "最大回撤": max_drawdown(window),
        "Sortino": sortino,
        "Sharpe": sharpe,
        "最差年份": year,
        "最差年收益": yret,
        "年均交易次数": trades,
        "平均持有周数": hold,
    }


def detect_rallies(
    weekly: pd.DataFrame,
    symbol: str,
    start: str = RALLY_START,
    min_gain: float = RALLY_MIN_GAIN,
    max_weeks: int = RALLY_MAX_WEEKS,
) -> list[dict]:
    """Non-overlapping trough-to-peak rallies of at least ``min_gain`` inside ``max_weeks``."""
    frame = weekly.loc[weekly.index >= pd.Timestamp(start)].copy()
    if frame.empty:
        return []
    fridays = list(frame.index)
    px = frame["Close"].to_numpy(dtype=float)
    sessions = list(pd.to_datetime(frame["session"]))
    n = len(px)
    candidates = []
    for i in range(n):
        j_end = min(n, i + max_weeks + 1)
        if i + 1 >= j_end:
            continue
        j = i + 1 + int(np.argmax(px[i + 1 : j_end]))
        if px[i] <= 0:
            continue
        gain = px[j] / px[i] - 1.0
        if gain + 1e-12 < min_gain:
            continue
        if px[i] > np.min(px[i : j + 1]) + 1e-9:
            continue
        candidates.append((i, j, float(gain)))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    chosen: list[tuple[int, int, float]] = []
    for i, j, gain in candidates:
        if any(_intervals_overlap(i, j, i2, j2) for i2, j2, _ in chosen):
            continue
        chosen.append((i, j, gain))
    chosen.sort(key=lambda item: item[0])
    rows = []
    for i, j, gain in chosen:
        rows.append(_rally_row(symbol, fridays, sessions, px, i, j, gain, ""))
    return rows


def best_jul_aug(weekly: pd.DataFrame, symbol: str, year: int, max_weeks: int = RALLY_MAX_WEEKS) -> dict | None:
    """Largest trough-to-peak move whose peak Friday falls in July or August, lookback ≤ max_weeks."""
    if weekly.empty or int(weekly.index.max().year) < year:
        return None
    if not ((weekly.index.year == year) & (weekly.index.month.isin([7, 8]))).any():
        return None
    fridays = list(weekly.index)
    px = weekly["Close"].to_numpy(dtype=float)
    sessions = list(pd.to_datetime(weekly["session"]))
    best = None
    for j, friday in enumerate(fridays):
        if friday.year != year or friday.month not in (7, 8):
            continue
        i0 = max(0, j - max_weeks)
        i = i0 + int(np.argmin(px[i0 : j + 1]))
        if i >= j or px[i] <= 0:
            continue
        gain = float(px[j] / px[i] - 1.0)
        if best is None or gain > best[0]:
            best = (gain, i, j)
    if best is None:
        return None
    gain, i, j = best
    note = f"{year}年7–8月"
    if gain + 1e-12 < RALLY_MIN_GAIN:
        note += "（涨幅未达10%，仍单独列出）"
    return _rally_row(symbol, fridays, sessions, px, i, j, gain, note)


def _intervals_overlap(i: int, j: int, i2: int, j2: int) -> bool:
    if j < i2 or j2 < i:
        return False
    if j == i2 or j2 == i:
        return False
    return True


def _rally_row(symbol, fridays, sessions, px, i, j, gain, note) -> dict:
    return {
        "标的": symbol,
        "起涨日": pd.Timestamp(fridays[i]).strftime("%Y-%m-%d"),
        "见顶日": pd.Timestamp(fridays[j]).strftime("%Y-%m-%d"),
        "trough": pd.Timestamp(fridays[i]),
        "peak": pd.Timestamp(fridays[j]),
        "trough_session": pd.Timestamp(sessions[i]),
        "peak_session": pd.Timestamp(sessions[j]),
        "周数": (pd.Timestamp(fridays[j]) - pd.Timestamp(fridays[i])).days / 7.0,
        "涨幅": float(gain),
        "备注": note,
    }


def attach_gold_windows(rallies: list[dict], weekly_by_symbol: dict[str, pd.DataFrame]) -> list[dict]:
    rows = list(rallies)
    gld = weekly_by_symbol["GLD"]
    for year in (2025, 2026):
        season = best_jul_aug(gld, "GLD", year)
        if season is None:
            rows.append(
                {
                    "标的": "GLD",
                    "起涨日": "",
                    "见顶日": "",
                    "trough": pd.NaT,
                    "peak": pd.NaT,
                    "trough_session": pd.NaT,
                    "peak_session": pd.NaT,
                    "周数": float("nan"),
                    "涨幅": float("nan"),
                    "备注": f"{year}年7–8月无行情数据",
                }
            )
            continue
        same_peak = [
            row
            for row in rows
            if row["标的"] == "GLD" and row.get("peak") == season["peak"] and row.get("trough") == season["trough"]
        ]
        if same_peak:
            label = f"{year}年7–8月"
            if season["涨幅"] + 1e-12 < RALLY_MIN_GAIN:
                label += "（涨幅未达10%，仍单独列出）"
            same_peak[0]["备注"] = label
        else:
            if not season["备注"]:
                season["备注"] = f"{year}年7–8月指定窗口"
            rows.append(season)
    return rows


def _on_index(index: pd.DatetimeIndex, day) -> pd.Timestamp | None:
    day = pd.Timestamp(day)
    if pd.isna(day):
        return None
    if day in index:
        return day
    prev = index[index <= day]
    if len(prev) == 0:
        return None
    return pd.Timestamp(prev[-1])


def capture_one(result: dict, rally: dict) -> dict:
    """Share of the trough-to-peak log move earned while the up-leg was on (half credit if halved)."""
    if pd.isna(rally.get("trough_session")) or pd.isna(rally.get("peak_session")):
        return {"捕获比例": float("nan"), "入场滞后周": float("nan"), "出场滞后周": float("nan")}
    symbol = rally["标的"]
    closes = result["closes"][symbol]
    opens = result["opens"][symbol]
    part = result["participation"][symbol]
    trough_s = _on_index(closes.index, rally["trough_session"])
    peak_s = _on_index(closes.index, rally["peak_session"])
    if trough_s is None or peak_s is None or trough_s >= peak_s:
        return {"捕获比例": float("nan"), "入场滞后周": float("nan"), "出场滞后周": float("nan")}
    days = closes.loc[trough_s:peak_s].index
    total = float(np.log(closes.loc[days[-1]] / closes.loc[days[0]]))
    captured = 0.0
    for left, right in zip(days[:-1], days[1:]):
        left_part = float(part.loc[left])
        right_part = float(part.loc[right])
        if left_part == right_part:
            captured += left_part * float(np.log(closes.loc[right] / closes.loc[left]))
        else:
            captured += left_part * float(np.log(opens.loc[right] / closes.loc[left]))
            captured += right_part * float(np.log(closes.loc[right] / opens.loc[right]))
    ratio = captured / total if abs(total) > 1e-12 else float("nan")
    entry_lag, exit_lag = _lags(part, rally["trough"], rally["peak"], trough_s, peak_s)
    return {"捕获比例": ratio, "入场滞后周": entry_lag, "出场滞后周": exit_lag}


def _lags(part: pd.Series, trough_f, peak_f, trough_s, peak_s) -> tuple[float, float]:
    episodes = []
    active = False
    entry = None
    for day, value in part.items():
        if value > 0 and not active:
            active = True
            entry = pd.Timestamp(day)
        elif value == 0 and active:
            episodes.append((entry, pd.Timestamp(day)))
            active = False
            entry = None
    if active and entry is not None:
        episodes.append((entry, None))
    overlapping = []
    for en, ex in episodes:
        ex_cmp = ex if ex is not None else part.index[-1] + pd.Timedelta(days=1)
        if en <= peak_s and ex_cmp > trough_s:
            overlapping.append((en, ex))
    if not overlapping:
        return float("nan"), float("nan")
    entry_lag = (overlapping[0][0] - pd.Timestamp(trough_f)).days / 7.0
    exit_day = overlapping[-1][1]
    if exit_day is None:
        return entry_lag, float("nan")
    return entry_lag, (exit_day - pd.Timestamp(peak_f)).days / 7.0


def build_capture_table(rallies: list[dict], runs: dict[str, dict]) -> pd.DataFrame:
    records = []
    for rally in rallies:
        base = {
            "标的": rally["标的"],
            "起涨日": rally["起涨日"],
            "见顶日": rally["见顶日"],
            "周数": rally["周数"],
            "涨幅": rally["涨幅"],
            "备注": rally["备注"],
        }
        for name, result in runs.items():
            stats = capture_one(result, rally)
            base[f"{name}捕获"] = stats["捕获比例"]
            base[f"{name}入场滞后周"] = stats["入场滞后周"]
            base[f"{name}出场滞后周"] = stats["出场滞后周"]
        records.append(base)
    return pd.DataFrame(records)


def _fmt_pct(value, digits=2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value, digits=2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_lag(value) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.1f}"


def metrics_markdown(frame: pd.DataFrame) -> str:
    headers = ["版本", "区间", "年化", "最大回撤", "Sortino", "Sharpe", "最差年份", "交易/年", "平均持有周数"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        worst = f"{int(row['最差年份'])} {_fmt_pct(row['最差年收益'])}"
        cells = [
            row["版本"],
            f"{row['区间']} {row['起始']}~{row['结束']}",
            _fmt_pct(row["年化收益"]),
            _fmt_pct(row["最大回撤"]),
            _fmt_num(row["Sortino"]),
            _fmt_num(row["Sharpe"]),
            worst,
            _fmt_num(row["年均交易次数"], 2),
            _fmt_num(row["平均持有周数"], 1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def capture_markdown(frame: pd.DataFrame, versions: list[str]) -> str:
    if frame.empty:
        return "（没有识别到短涨）"
    headers = ["标的", "起涨", "见顶", "周数", "涨幅", "备注"]
    for name in versions:
        headers.extend([f"{name}捕获", f"{name}入场滞后", f"{name}出场滞后"])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        cells = [
            str(row["标的"]),
            str(row["起涨日"]) or "—",
            str(row["见顶日"]) or "—",
            _fmt_num(row["周数"], 1),
            _fmt_pct(row["涨幅"]),
            str(row["备注"] or ""),
        ]
        for name in versions:
            cells.append(_fmt_pct(row[f"{name}捕获"]))
            cells.append(_fmt_lag(row[f"{name}入场滞后周"]))
            cells.append(_fmt_lag(row[f"{name}出场滞后周"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_capture(frame: pd.DataFrame, version: str, min_gain: float = RALLY_MIN_GAIN) -> float:
    if frame.empty or f"{version}捕获" not in frame.columns:
        return float("nan")
    mask = frame["涨幅"].ge(min_gain) & frame["起涨日"].astype(str).str.len().gt(0)
    values = frame.loc[mask, f"{version}捕获"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def configure_font() -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    path = Path(FONT_PATH)
    if not path.exists():
        raise RuntimeError(f"CJK font missing: {path}")
    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False
    return family


def _text_overlaps(fig, ax) -> list[tuple[str, str]]:
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_ids = {id(label) for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels())}
    items = []
    for text in fig.findobj(match=lambda obj: hasattr(obj, "get_text") and hasattr(obj, "get_window_extent")):
        if not text.get_visible():
            continue
        if type(text).__name__ == "TextArea":
            continue
        raw = text.get_text()
        if raw is None or not str(raw).strip():
            continue
        box = text.get_window_extent(renderer=renderer)
        if box.width < 4 or box.height < 4:
            continue
        shrunk = Bbox.from_extents(box.x0 + 2, box.y0 + 2, box.x1 - 2, box.y1 - 2)
        if shrunk.width <= 0 or shrunk.height <= 0:
            continue
        items.append((str(raw).replace("\n", " ")[:48], shrunk, id(text) in tick_ids))
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a[2] and b[2]:
                continue
            if a[1].overlaps(b[1]):
                pairs.append((a[0], b[0]))
    return pairs


def plot_three(
    navs: dict[int, pd.Series],
    last: pd.Timestamp,
    out_equity: Path,
    out_dd: Path,
) -> None:
    """Exactly three lines: 30-week baseline, then 20 and 10."""
    import matplotlib.pyplot as plt

    configure_font()
    end = pd.Timestamp(last).strftime("%Y-%m-%d")
    styles = {
        30: {"color": "black", "linewidth": 2.8, "linestyle": "-", "zorder": 3},
        20: {"color": "#D55E00", "linewidth": 1.9, "linestyle": (0, (6, 2)), "zorder": 2},
        10: {"color": "#0072B2", "linewidth": 2.0, "linestyle": (0, (1, 1)), "zorder": 1},
    }
    labels = {
        30: f"30周均线（基准）\n主样本 2019-01-02 至 {end}\n周频 · 现金底仓30% · 目标波动16%",
        20: f"20周均线（对比）\n主样本 2019-01-02 至 {end}\n周频 · 现金底仓30% · 目标波动16%",
        10: f"10周均线（对比）\n主样本 2019-01-02 至 {end}\n周频 · 现金底仓30% · 目标波动16%",
    }
    subtitle = (
        "周线周五收盘；连续2周收盘高于均线且均线高于4周前才进入\n"
        "收盘低于均线连续2周减半；均线同时低于4周前连续2周则清空\n"
        "最多3个持仓；现金底仓30%；单名预算=70%÷3×斐波那契阶段权重\n"
        "目标波动16%且不加杠杆；单边成本10bp；周五决策，次周第2个交易日开盘成交"
    )
    panels = (
        (
            out_equity,
            f"升腿均线比较：10周、20周、30周净值（主样本 2019-01-02 至 {end}）",
            "净值（起点=1）",
            {n: navs[n] for n in (10, 20, 30)},
            False,
        ),
        (
            out_dd,
            f"升腿均线比较：10周、20周、30周回撤（主样本 2019-01-02 至 {end}）",
            "回撤",
            {n: navs[n] / navs[n].cummax() - 1.0 for n in (10, 20, 30)},
            True,
        ),
    )
    for path, title, ylabel, series, is_dd in panels:
        fig, ax = plt.subplots(figsize=(14.8, 8.2), layout="constrained")
        for n in (10, 20, 30):
            ax.plot(series[n].index, series[n].to_numpy(), label=labels[n], **styles[n])
        heading = ax.set_title(title + "\n" + subtitle, fontsize=11, loc="left", pad=12)
        heading.set_linespacing(1.45)
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
        ax.set_xlabel("日期", fontsize=11, labelpad=8)
        ax.tick_params(labelsize=9)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8.5,
            labelspacing=0.7,
            borderpad=0.6,
        )
        overlaps = _text_overlaps(fig, ax)
        if overlaps:
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(1.01, 0.0),
                borderaxespad=0.0,
                frameon=True,
                fontsize=8.5,
                labelspacing=0.7,
            )
            overlaps = _text_overlaps(fig, ax)
        handles, _ = ax.get_legend_handles_labels()
        if len(handles) != 3:
            raise RuntimeError(f"chart must have exactly 3 lines, found {len(handles)}")
        fig.savefig(path, dpi=140, facecolor="white", bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)
        if overlaps:
            raise RuntimeError(f"overlapping chart text in {path.name}: {overlaps[:8]}")


def render_summary(
    metrics: pd.DataFrame,
    capture: pd.DataFrame,
    versions: list[str],
    meta: dict,
) -> str:
    weekly_names = [name for name in versions if not name.endswith("日频")]
    main = metrics[metrics["区间"] == "主样本"]
    main_w = main[main["版本"].isin(weekly_names)].copy()
    best_sortino = main_w.loc[main_w["Sortino"].idxmax()]
    best_ann = main_w.loc[main_w["年化收益"].idxmax()]
    shallow_dd = main_w.loc[main_w["最大回撤"].idxmax()]  # least negative
    captures = {name: mean_capture(capture, name) for name in versions}
    best_cap = max(weekly_names, key=lambda name: captures[name] if np.isfinite(captures[name]) else -1e9)

    def row(version, period):
        hit = metrics[(metrics["版本"] == version) & (metrics["区间"] == period)]
        return hit.iloc[0]

    lines = []
    lines.append("# GLD / SLV / USO 升腿均线比较（10周、20周、30周）")
    lines.append("")
    lines.append("## 研究在回答什么")
    lines.append("")
    lines.append(
        "想抓住大约一到两个月的上涨，参照黄金 2025 年 7–8 月那种波段。"
        "比较对象只有三条均线长度：10 周、20 周、30 周。三版规则相同，只改均线长度 n。"
        "品种只看 GLD、SLV、USO。另外各做一版日频检查，只放进表里，不放进图。"
    )
    lines.append("")
    lines.append("## 规则")
    lines.append("")
    lines.append(
        f"周线用周五收盘（周五休市则用当周最后一个交易日的收盘）。"
        f"进入升腿：收盘高于 MA(n)，并且 MA(n) 高于 4 周前的 MA(n)，连续 {CONFIRM_WEEKS} 周成立。"
        f"离开升腿：收盘低于 MA(n) 连续 {CONFIRM_WEEKS} 周，该持仓减半；"
        f"若同时 MA(n) 低于 4 周前的 MA(n)，也连续 {CONFIRM_WEEKS} 周，则清空。"
        "减半看的是当下这一串条件，收盘重新站上均线后恢复该升腿的正常预算，不会一直锁在半仓。"
        "清空之后要重新走出进场条件才再买。比较都用严格大于、严格小于。"
    )
    lines.append("")
    lines.append(
        f"只持有处在升腿里的品种，最多 {MAX_HOLDINGS} 个。"
        f"现金底仓固定 {CASH_FLOOR:.0%}，所以最多投入 {1-CASH_FLOOR:.0%}。"
        "每个品种的预算是 70%÷3，再乘斐波那契阶段权重："
        + "、".join(f"{k} {v:.1f}" for k, v in STAGE_WEIGHT.items())
        + "。减半时再乘 0.5。"
    )
    lines.append("")
    lines.append(
        f"斐波那契阶段用周线拐点，左右各 {3} 周；摆动至少要有 ATR({14}) 的 {1.5} 倍。"
        f"从最近一段下跌的低点往上量，带宽 {FIB_BANDS[0]} / {FIB_BANDS[1]} / {FIB_BANDS[2]}。"
        "收盘在 0.618 倍以下是磨底，0.618 到 1 倍是震荡，1 倍到 1.618 倍是主升，1.618 倍及以上是鱼尾。"
        "刚好压在边界上算更高的一档。阶段要连续 "
        f"{FIB_CONFIRM_WEEKS} 周相同才切换，开始时先按震荡（权重 0.5）。"
    )
    lines.append("")
    lines.append(
        f"组合波动用周收益的 EWMA（span {EWMA_SPAN}，adjust=False）年化，相关系数固定 0.35。"
        f"目标波动 {TARGET_VOL:.0%}。组合波动高于目标就按比例缩小仓位；低于目标不放大，不加杠杆。"
        f"持仓名里只要还有一个波动率没满 {EWMA_SPAN} 个收益，这一周就不做波动缩放。"
        "份额可以是小数。周五收盘做决定，下一周的第 2 个交易日开盘成交"
        "（该周只有一个交易日就用那一天；整周没有交易日就往后顺延）。"
        f"单边成本 {COST_BPS:.0f} bp 佣金加 {SLIPPAGE_BPS:.0f} bp 滑点，只打在成交金额上。"
    )
    lines.append("")
    lines.append("## 日频变体")
    lines.append("")
    lines.append(
        f"每个 n 另跑一版日频。信号改成日线收盘对 MA(n×{DAILY_BARS_PER_WEEK})："
        f"10 周对应 {10*DAILY_BARS_PER_WEEK} 日，20 周对应 {20*DAILY_BARS_PER_WEEK} 日，"
        f"30 周对应 {30*DAILY_BARS_PER_WEEK} 日。"
        f"「4 周」改成 {SLOPE_LAG_WEEKS*DAILY_BARS_PER_WEEK} 个交易日，"
        f"「连续 2 周」改成 {CONFIRM_WEEKS*DAILY_BARS_PER_WEEK} 个交易日。"
        "斐波那契阶段仍用周线，当天用最近一个已经走完的周线阶段。"
        f"波动改用日收益 EWMA，span 仍是 {EWMA_SPAN} 个交易日，年化按 {TRADING_DAYS_PER_YEAR}。"
        "信号在当日收盘确认，下一个交易日开盘成交。这样做是为了看更快的检查能不能多吃到一到两个月的波段。"
        "日频结果只出现在表里。"
    )
    lines.append("")
    lines.append("## 数据与区间")
    lines.append("")
    lines.append(
        f"价格是 Yahoo 日线复权价，缓存在本地，不进版本库。快照截止 {meta['last']}。"
        f"GLD 首日 {meta['first']['GLD']}，SLV 首日 {meta['first']['SLV']}，USO 首日 {meta['first']['USO']}。"
        f"长历史净值从三只都有行情的 {meta['long_start']} 起算，更早的数据只用来预热均线、ATR 和拐点。"
        f"主样本是 {MAIN_START} 至 {meta['last']}，沿用同一条已经在跑的仓位，不是在 2019 年把仓位清零重开。"
        f"2022 年以来从 {SUB_2022_START} 起切。样本内 {MAIN_START} 至 {IS_END}，样本外 {OOS_START} 至 {meta['last']}。"
        "三条长度都是事先指定的，没有在样本内挑选。"
    )
    lines.append("")
    lines.append(
        "年化按 252 个交易日。Sharpe、Sortino 的无风险利率按 0，Sortino 的下行偏差用全部日收益"
        "（亏损平方的均值再开方）。最差年份是区间内自然年简单收益里最低的一年，最后一年如果还没过完就用截至数据日的收益。"
        "交易次数只数从空仓到持有的开仓，减半和加减仓不算新交易。平均持有周数含样本结束时还没平的仓，算到样本最后一天。"
    )
    lines.append("")
    lines.append("## 关键数字")
    lines.append("")
    lines.append(metrics_markdown(metrics))
    lines.append("")
    lines.append("## 短涨捕获")
    lines.append("")
    lines.append(
        f"从 {RALLY_START} 起，在周线收盘上找涨幅至少 {RALLY_MIN_GAIN:.0%}、"
        f"时长不超过 {RALLY_MAX_WEEKS} 周的上涨。重叠的只留涨幅更大的一段，端点相接不算重叠。"
        "黄金 2025 年与 2026 年的 7–8 月另外强制列出：取见顶落在 7 月或 8 月、向前最多 12 周的最大涨幅；"
        "数据里没有这段就注明。涨幅不到 10% 也照样列出。"
    )
    lines.append("")
    lines.append(
        "捕获比例是这段对数涨幅里，升腿参与度盖住的部分。满仓参与度是 1，减半是 0.5，空仓是 0。"
        "它量的是有没有骑在这段行情上，用的是升腿参与度，不是组合里那百分之十几的资金权重。"
        "中间的下跌日如果空仓，捕获比例可以高于 100%。"
        "入场滞后是开仓日相对起涨周五的周数，负数表示起涨前已经在场内。"
        "出场滞后是该段最后一次平仓日相对见顶周五的周数，负数表示见顶前已经出来；仍持有则记为「—」。"
    )
    lines.append("")
    lines.append(capture_markdown(capture, versions))
    lines.append("")
    qual = capture[capture["涨幅"].ge(RALLY_MIN_GAIN) & capture["起涨日"].astype(str).str.len().gt(0)]
    lines.append(
        f"达到 10% 条件的短涨有 {len(qual)} 段。平均捕获比例："
        + "，".join(f"{name} {_fmt_pct(captures[name])}" for name in versions)
        + "。"
    )
    lines.append("")
    lines.append("## 怎么读这些数")
    lines.append("")
    lines.append(
        f"主样本上周频年化最高的是{best_ann['版本']}（{_fmt_pct(best_ann['年化收益'])}），"
        f"Sortino 最高的是{best_sortino['版本']}（{_fmt_num(best_sortino['Sortino'])}），"
        f"回撤最浅的是{shallow_dd['版本']}（{_fmt_pct(shallow_dd['最大回撤'])}）。"
        f"短涨平均捕获最高的周频版本是{best_cap}（{_fmt_pct(captures[best_cap])}）。"
    )
    for n in (10, 20, 30):
        wname, dname = f"{n}周", f"{n}周日频"
        w_main, d_main = row(wname, "主样本"), row(dname, "主样本")
        lines.append(
            f"{n}周改成日频之后，主样本年化从 {_fmt_pct(w_main['年化收益'])} 变为 {_fmt_pct(d_main['年化收益'])}，"
            f"最大回撤从 {_fmt_pct(w_main['最大回撤'])} 变为 {_fmt_pct(d_main['最大回撤'])}，"
            f"短涨平均捕获从 {_fmt_pct(captures[wname])} 变为 {_fmt_pct(captures[dname])}。"
        )
    for n in (10, 20, 30):
        inside = row(f"{n}周", "样本内")
        outside = row(f"{n}周", "样本外")
        lines.append(
            f"{n}周样本内年化 {_fmt_pct(inside['年化收益'])}、回撤 {_fmt_pct(inside['最大回撤'])}；"
            f"样本外年化 {_fmt_pct(outside['年化收益'])}、回撤 {_fmt_pct(outside['最大回撤'])}。"
        )
    def _misses(version: str) -> tuple[int, int]:
        mask = capture["涨幅"].ge(RALLY_MIN_GAIN) & capture["起涨日"].astype(str).str.len().gt(0)
        col = capture.loc[mask, f"{version}捕获"]
        missed = (~np.isfinite(col.to_numpy(dtype=float))) | (col.fillna(0).to_numpy(dtype=float) <= 1e-12)
        return int(np.sum(missed)), int(mask.sum())

    miss_bits = []
    for name in weekly_names:
        missed, total = _misses(name)
        miss_bits.append(f"{name}空仓错过 {missed}/{total}")
    lines.append(
        "平均捕获高，主要来自慢均线在起涨前已经持有，入场滞后因此经常是负数。空仓错过的段数："
        + "；".join(miss_bits)
        + "。"
    )
    gold = capture[capture["备注"].astype(str).str.contains("7–8月", na=False)]
    if gold.empty:
        lines.append("黄金 2025 与 2026 的 7–8 月窗口没有写进捕获表。")
    else:
        lines.append("黄金夏天窗口：")
        for _, item in gold.iterrows():
            bits = [
                f"{item['备注']} {item['起涨日']} → {item['见顶日']}，涨幅 {_fmt_pct(item['涨幅'])}"
            ]
            for name in weekly_names:
                bits.append(
                    f"{name}捕获 {_fmt_pct(item[f'{name}捕获'])}"
                    f"（入场 {_fmt_lag(item[f'{name}入场滞后周'])} 周，出场 {_fmt_lag(item[f'{name}出场滞后周'])} 周）"
                )
            lines.append("；".join(bits) + "。")
    october = capture[(capture["标的"] == "GLD") & (capture["起涨日"] == "2025-07-25")]
    if len(october):
        item = october.iloc[0]
        lines.append(
            f"2025 年黄金更大的一段从 {item['起涨日']} 到 {item['见顶日']}，涨幅 {_fmt_pct(item['涨幅'])}，"
            f"高点在 10 月，7–8 月本身在周线上只是横盘。"
            f"10周捕获 {_fmt_pct(item['10周捕获'])}，20周捕获 {_fmt_pct(item['20周捕获'])}，"
            f"30周捕获 {_fmt_pct(item['30周捕获'])}。"
        )
    lines.append("")
    lines.append(
        "图只画主样本上的三条周频净值，以及同一条净值上的回撤。"
        "30 周是黑色粗实线，20 周是橙色虚线，10 周是蓝色点线。日频不进图。"
    )
    lines.append("")
    lines.append(
        "这是研究草稿，用来比较规则对一到两个月上涨的覆盖，不是交易建议，也没有扣税和资金容量。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"
