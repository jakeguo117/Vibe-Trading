"""Equity charts. At most three lines. CSI 300 is the thick black line.

Prefer Hiragino Sans GB on macOS (Hermes); fall back to WenQuanYi on Linux.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Hiragino Sans GB.ttc",
    "/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
)
_FONT_READY = False
_FONT_NAME = "sans-serif"


def _use_cjk() -> None:
    global _FONT_READY, _FONT_NAME
    if _FONT_READY:
        return
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            _FONT_NAME = prop.get_name()
            plt.rcParams["font.family"] = _FONT_NAME
            break
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def plot_equity(
    path: Path,
    title: str,
    strategy_name: str,
    strategy: pd.Series,
    csi300: pd.Series,
    csi_all: pd.Series | None = None,
) -> None:
    """Save one chart. If ``csi_all`` is None, only strategy + CSI 300 (2 lines)."""
    _use_cjk()
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    series = [
        (strategy_name, strategy, {"color": "#c0392b", "linewidth": 1.6}),
        ("沪深300", csi300, {"color": "black", "linewidth": 2.6}),
    ]
    if csi_all is not None:
        series.append(("中证全指", csi_all, {"color": "#2471a3", "linewidth": 1.4, "linestyle": "--"}))
    for name, values, style in series:
        if values is None or len(values) == 0:
            continue
        base = float(values.iloc[0])
        if base == 0:
            continue
        ax.plot(values.index, values / base, label=name, **style)
    ax.set_ylabel("净值（起点 = 1）", fontsize=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.grid(True, color="#d5d8dc", linewidth=0.6)
    fig.suptitle(title, fontsize=15, y=0.98)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(3, len(series)),
        frameon=False,
        fontsize=12,
        columnspacing=1.8,
        handlelength=2.6,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_top2_vs_csi300(
    path: Path,
    title: str,
    named_series: list[tuple[str, pd.Series]],
    csi300: pd.Series,
) -> None:
    """Exactly: top 2 tactics + CSI 300 (thick black). Max 3 lines."""
    _use_cjk()
    colors = ("#c0392b", "#1e8449")
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    ax.plot(
        csi300.index,
        csi300 / float(csi300.iloc[0]),
        label="沪深300",
        color="black",
        linewidth=2.8,
    )
    for (name, series), color in zip(named_series[:2], colors):
        ax.plot(
            series.index,
            series / float(series.iloc[0]),
            label=name,
            color=color,
            linewidth=1.7,
        )
    ax.set_ylabel("净值（起点 = 1）", fontsize=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.grid(True, color="#d5d8dc", linewidth=0.6)
    fig.suptitle(title, fontsize=15, y=0.98)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=12,
        columnspacing=1.8,
        handlelength=2.6,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
