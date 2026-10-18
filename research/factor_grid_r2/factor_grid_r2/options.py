"""Black-Scholes calls with a fixed IV premium. Not a historical print."""

from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def call_price(spot: float, strike: float, tau: float, iv: float, rate: float, div: float) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    if tau <= 1.0 / 365.0 or iv <= 1e-6:
        return max(spot * math.exp(-div * max(tau, 0.0)) - strike * math.exp(-rate * max(tau, 0.0)), 0.0)
    vol = iv * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * iv * iv) * tau) / vol
    d2 = d1 - vol
    return spot * math.exp(-div * tau) * _ncdf(d1) - strike * math.exp(-rate * tau) * _ncdf(d2)


def call_delta(spot: float, strike: float, tau: float, iv: float, rate: float, div: float) -> float:
    if spot <= 0 or strike <= 0 or tau <= 1.0 / 365.0 or iv <= 1e-6:
        return 1.0 if spot > strike else 0.0
    vol = iv * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * iv * iv) * tau) / vol
    return math.exp(-div * tau) * _ncdf(d1)


def strike_for_delta(spot: float, tau: float, iv: float, delta: float, rate: float, div: float) -> float:
    lo, hi = spot * 0.05, spot * 5.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if call_delta(spot, mid, tau, iv, rate, div) > delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clip_iv(realized: float, mult: float, floor: float, cap: float) -> float:
    base = realized if np.isfinite(realized) and realized > 0 else floor
    return float(min(cap, max(floor, base * mult)))
