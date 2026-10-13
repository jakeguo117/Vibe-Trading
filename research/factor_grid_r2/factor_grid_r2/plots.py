"""Line charts: at most three series, baseline drawn thick and black."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

_STYLES = [
    {"color": "#C45C26", "linestyle": "--", "linewidth": 1.6},
    {"color": "#1F4E79", "linestyle": "-.", "linewidth": 1.6},
]


def line_chart(path: Path, title: str, ylabel: str, series: list[tuple[str, object]], baseline: int = 0):
    """``series`` is (legend, pandas Series). Length must be 1..3.

    The axes sit in a fixed box so the title and the legend each have their
    own band and do not cover the lines or each other.
    """
    if not 1 <= len(series) <= 3:
        raise ValueError(f"chart must have 1 to 3 lines, got {len(series)}")
    fig = plt.figure(figsize=(11.4, 8.0))
    # Title band above 0.78, legend band below 0.30, plot in between.
    ax = fig.add_axes([0.09, 0.34, 0.86, 0.42])
    other = 0
    for i, (label, curve) in enumerate(series):
        c = curve.dropna()
        if c.empty:
            continue
        if i == baseline:
            ax.plot(c.index, c.to_numpy(), color="black", linestyle="-", linewidth=2.6, label=label, zorder=3)
        else:
            style = _STYLES[other % len(_STYLES)]
            other += 1
            ax.plot(c.index, c.to_numpy(), label=label, zorder=2, **style)
    ax.set_title(title, fontsize=10, loc="left", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("日期")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=1,
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
