"""Write a seeded synthetic factor journal whose answers are known in advance.

    python examples/make_synthetic.py            # -> examples/synthetic.csv

30 names, ~1 year of business days. Each name's daily return is
    common market move + a * s + b * s2 (first half only) + noise
where s and s2 are slow AR(1) signals. The journal logs, per date and name:

  momentum_like   s seen through noise: really predictive, all year
  decays          s2 seen through noise: predictive in H1, worthless in H2
  noise_01..10    slow AR(1) series with no link to returns
  fwd_1/5/20      forward returns in percent; blank where not yet matured

Nothing here is market data. The point is that you know which factors are
real before running the tools, so you can see whether the tools agree.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SEED = 42
N_NAMES = 30
N_DAYS = 260
START = date(2025, 1, 6)
PHI = 0.97              # persistence of every latent series
SIGNAL_A = 0.13         # % daily return per unit of s
SIGNAL_B = 0.13         # same for s2, first half only
NOISE_SD = 2.0          # idiosyncratic daily return sd, %
MARKET_SD = 1.0         # common daily move, %
OBS_SD = 0.5            # measurement noise on the logged factors
N_NOISE = 10
HORIZONS = (1, 5, 20)


def business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def ar1(rng: np.random.Generator, n_days: int, n_names: int) -> np.ndarray:
    x = np.empty((n_days, n_names))
    x[0] = rng.normal(size=n_names)
    shock_sd = np.sqrt(1 - PHI ** 2)
    for t in range(1, n_days):
        x[t] = PHI * x[t - 1] + shock_sd * rng.normal(size=n_names)
    return x


def simulate(seed: int = SEED) -> tuple[list[date], dict[str, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    days = business_days(START, N_DAYS)
    s, s2 = ar1(rng, N_DAYS, N_NAMES), ar1(rng, N_DAYS, N_NAMES)
    first_half = (np.arange(N_DAYS) < N_DAYS // 2)[:, None]
    daily = np.zeros((N_DAYS, N_NAMES))
    daily[1:] = (SIGNAL_A * s[:-1] + SIGNAL_B * s2[:-1] * first_half[:-1]
                 + MARKET_SD * rng.normal(size=(N_DAYS - 1, 1))
                 + NOISE_SD * rng.normal(size=(N_DAYS - 1, N_NAMES)))
    factors = {"momentum_like": s + OBS_SD * rng.normal(size=s.shape),
               "decays": s2 + OBS_SD * rng.normal(size=s2.shape)}
    for k in range(1, N_NOISE + 1):
        factors[f"noise_{k:02d}"] = ar1(rng, N_DAYS, N_NAMES)
    growth = np.cumprod(1 + daily / 100, axis=0)
    fwd = {}
    for h in HORIZONS:
        f = np.full((N_DAYS, N_NAMES), np.nan)
        f[:-h] = (growth[h:] / growth[:-h] - 1) * 100
        fwd[h] = f
    return days, factors, fwd


def write(path: Path, seed: int = SEED) -> int:
    days, factors, fwd = simulate(seed)
    names = [f"SYM{i:02d}" for i in range(N_NAMES)]
    header = ["date", "symbol", *factors, *(f"fwd_{h}" for h in HORIZONS)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for t, d in enumerate(days):
            for i, name in enumerate(names):
                vals = [f"{factors[k][t, i]:.4f}" for k in factors]
                rets = ["" if np.isnan(fwd[h][t, i]) else f"{fwd[h][t, i]:.4f}" for h in HORIZONS]
                w.writerow([d.isoformat(), name, *vals, *rets])
    return N_DAYS * N_NAMES


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "synthetic.csv"
    n = write(out)
    print(f"wrote {n} rows to {out}")
