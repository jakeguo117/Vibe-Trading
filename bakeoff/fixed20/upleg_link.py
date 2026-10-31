"""Read-only link to trading-system ``upleg_bt``. No local Fib or hist.

The F leg must call ``fib_stages_weekly`` and ``FM.score`` from the pinned
commit. If that package is not installed, callers fail. They do not switch
to another formula.
"""

from __future__ import annotations

PIN = "70038a53e7992b49f10e5906b18573a2beddb965"
REPO = "https://github.com/jakeguo117/trading-system"
INSTALL = f'pip install --no-deps "git+{REPO}@{PIN}"'

_FETCH_NOTE = (
    "本环境拉取失败：GitHub 返回 Repository not found。"
    "当前凭证读不到私有仓库 jakeguo117/trading-system。"
    "需要能读该仓库的 token，或把上述提交加为只读 submodule，再执行安装命令。"
)


class UplegUnavailable(RuntimeError):
    """Pinned upleg_bt could not be imported. Do not substitute a score."""


def probe() -> str | None:
    """Return a skip/error reason, or None when the pinned modules import."""
    try:
        import upleg_bt.costs  # noqa: F401
        import upleg_bt.engine.fib_locked  # noqa: F401
        import upleg_bt.engine.fm  # noqa: F401
    except ImportError as exc:
        return (
            "F 腿必须调用 upleg_bt.engine.fib_locked.fib_stages_weekly、"
            "upleg_bt.engine.fm.FM.score 和 FM.cells，费用必须调用 "
            "upleg_bt.costs.trade_fee。这些算法不在本仓库里重写。"
            f"钉死提交 {PIN}。安装：{INSTALL}。"
            f"{_FETCH_NOTE}"
            f"导入错误：{exc.__class__.__name__}: {exc}"
        )
    return None


def require() -> None:
    reason = probe()
    if reason:
        raise UplegUnavailable(reason)
