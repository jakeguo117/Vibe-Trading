"""Fib stage via the locked weekly function. No second stage table.

The frame passed in stops at the signal session, so the last week is only
the days already known. ``fib_stages_weekly`` is still the production function.
"""

from __future__ import annotations

import pandas as pd

from bakeoff.fixed20.models import BookView
from bakeoff.fixed20.upleg_link import require

STAGE_FUNCTION = "upleg_bt.engine.fib_locked.fib_stages_weekly"
# 主升 can be opened. 鱼尾 can be kept, not opened fresh. 磨底 / 震荡 cannot.
HOLDABLE_STAGES = frozenset({"trend", "fish"})
NEW_STAGES = frozenset({"trend"})


def stage_snapshot(view: BookView, symbol: str) -> tuple[str, int]:
    """Latest stage name and the route code ``FM.score`` expects.

    Route matches ``name_signals``: the latest stage code that is not 主升
    (code 2). ``FM.score`` folds code 3 into 2 itself.
    """
    require()
    from upleg_bt.engine.fib_locked import fib_stages_weekly, to_weekly
    from upleg_bt.engine.fm import SCODE

    dates = view.dates.get(symbol, ())
    if len(dates) < 2:
        return "grind", 0
    frame = pd.DataFrame(
        {
            "open": view.opens[symbol],
            "high": view.highs[symbol],
            "low": view.lows[symbol],
            "close": view.closes[symbol],
        },
        index=pd.DatetimeIndex(pd.to_datetime(list(dates))),
    )
    staged = fib_stages_weekly(to_weekly(frame))
    labels = [str(label) for label in staged["stage"].tolist()]
    name = labels[-1] if labels else "grind"
    if name not in SCODE:
        name = "grind"
    route = 0
    for label in labels:
        code = int(SCODE.get(label, 0))
        if code != 2:
            route = code
    return name, route
