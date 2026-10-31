"""Historical win-rate rank. A10: call ``FM.score``, do not reimplement it.

Synthetic runs pass an empty segment list. With no finished 主升 sample,
``FM.score`` returns the side-cost penalty (overall mean is 0). That is the
function's own fallback, not a bakeoff formula. A real window must pass the
production segment rows into this same function before CA publishes numbers.
"""

from __future__ import annotations

import numpy as np

from bakeoff.fixed20.upleg_link import require

HIST_FUNCTION = "upleg_bt.engine.fm.FM.score"
HIST_CELLS_FUNCTION = "upleg_bt.engine.fm.FM.cells"


class ProductionLinkError(RuntimeError):
    """The production hist function did not run. Do not substitute a score."""


def load_hist_view():
    """Attribute bag ``FM.score`` / ``FM.cells`` already know how to read."""
    require()
    from upleg_bt.engine.fm import FM

    class _HistView:
        cells = FM.cells

    return _HistView


def score_symbols(
    symbols: list[str],
    *,
    route: list[int],
    kind: list[int],
    side: list[float],
    market: int = 0,
    segments: tuple[tuple[int, float, float, float, int], ...] | None = None,
    cost_mult: float = 1.0,
) -> dict[str, float]:
    """Return ``FM.score(..., rank='hist')`` keyed by symbol.

    ``segments`` rows are ``(cell_id, gross_return, side_cost, n_halfs, exit_index)``.
    ``None`` means there is no finished sample. ``market`` is the production
    SPY 主升/鱼尾 flag; Fixed20 has no SPY, so the shell passes 0.
    """
    require()
    from upleg_bt.engine.fm import FM

    count = len(symbols)
    if not (len(route) == len(kind) == len(side) == count):
        raise ProductionLinkError("hist 输入长度不一致")
    if count == 0:
        return {}
    view_cls = load_hist_view()
    view = view_cls()
    view.T = 0
    view._cells = {}
    if segments:
        cid, ret, seg_side, halves, exits = zip(*segments)
        view._sg = (
            np.asarray(cid, dtype=np.int64),
            np.asarray(ret, dtype=np.float64),
            np.asarray(seg_side, dtype=np.float64),
            np.asarray(halves, dtype=np.float64),
            np.asarray(exits, dtype=np.int64),
        )
    else:
        view._sg = (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )
    view.ROUTE = np.asarray(route, dtype=np.int64).reshape(1, count)
    view.kind = np.asarray(kind, dtype=np.int64)
    view.MKT = np.asarray([market], dtype=np.int64)
    view.SIDE = np.asarray(side, dtype=np.float64).reshape(1, count)
    js = np.arange(count, dtype=np.int64)
    try:
        raw = FM.score(view, "hist", 0, js, cost_mult)
    except Exception as exc:
        raise ProductionLinkError("FM.score 未能计算 hist") from exc
    return {symbol: float(raw[index]) for index, symbol in enumerate(symbols)}
