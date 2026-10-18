"""Equity charts. At most three lines. CSI 300 is the thick black line.

The legend sits above the axes so it does not cover the curves or the title.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

_FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
_FONT_READY = False


def _use_cjk() -> None:
    global _FONT_READY
    if _FONT_READY:
        return
    if Path(_FONT).exists():
        font_manager.fontManager.addfont(_FONT)
        plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True


def plot_equity(
    path: Path,
    title: str,
    strategy_name: str,
    strategy: pd.Series,
    csi300: pd.Series,
    csi_all: pd.Series,
) -> None:
    """Save one chart: the tactic, CSI 300, and CSI All Share, each rebased to 1."""
    _use_cjk()
    fig, ax = plt.subplots(figsize=(11.2, 6.6))
    series = (
        (strategy_name, strategy, {"color": "#c0392b", "linewidth": 1.6}),
        ("沪深300", csi300, {"color": "black", "linewidth": 2.6}),
        ("中证全指", csi_all, {"color": "#2471a3", "linewidth": 1.4, "linestyle": "--"}),
    )
    for name, values, style in series:
        base = float(values.iloc[0])
        if base == 0:
            continue
        ax.plot(values.index, values / base, label=name, **style)
    ax.set_ylabel("净值（起点 = 1）", fontsize=12)
    ax.set_xlabel("日期", fontsize=12)
    ax.grid(True, color="#d5d8dc", linewidth=0.6)
    # Title and legend sit in separate bands above the axes so they cannot collide.
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
