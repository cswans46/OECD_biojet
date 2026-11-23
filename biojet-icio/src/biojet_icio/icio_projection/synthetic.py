"""Synthetic ICIO time series (current prices and PYP) for tests and demos.

Not data. Volumes follow log random walks with drift and AR(1) growth,
coefficients drift slowly, prices follow log random walks; tables are
written in the OECD wide csv layout (one file per year).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .table import Table, to_wide

SECTORS = ["A01", "B06", "C19", "C20", "D35", "F", "G", "H51", "J62_63", "K"]
FD = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"]


def make_history(countries=("CAN", "USA", "IND", "ROW"), years=range(1995, 2023), seed: int = 0,
                 sectors=SECTORS) -> dict[int, tuple[Table, Table]]:
    rng = np.random.default_rng(seed)
    keys = [f"{c}_{s}" for c in countries for s in sectors]
    fdc = [f"{c}_{k}" for c in countries for k in FD]
    n, nc, ns = len(keys), len(countries), len(sectors)
    ctry = np.repeat(np.arange(nc), ns)
    A0 = rng.uniform(0, 1, (n, n)) * (rng.uniform(0, 1, (n, n)) < 0.5)
    A0 *= np.where(ctry[:, None] == ctry[None, :], 1.0, 0.08)
    A0 = A0 / A0.sum(axis=0, keepdims=True) * rng.uniform(0.35, 0.6, n)[None, :]
    f0 = rng.uniform(50, 150, n) * np.repeat(rng.uniform(1, 5, nc), ns)
    drift = rng.normal(0.025, 0.01, n)
    phi = 0.4
    pdrift = rng.normal(0.015, 0.006, n)
    w_fd = np.array([0.6, 0.02, 0.15, 0.2, 0.01, 0.02])
    dest = np.zeros((n, nc))
    for i in range(n):
        d = rng.dirichlet(np.ones(nc))
        dest[i] = 0.75 * np.eye(nc)[ctry[i]] + 0.25 * d
    out = {}
    lf, g_prev, lp = np.log(f0), drift.copy(), np.zeros(n)
    A = A0.copy()
    p_prev = np.ones(n)
    years = list(years)
    for t, y in enumerate(years):
        if t:
            g = drift + phi * (g_prev - drift) + rng.normal(0, 0.02, n)
            if y in (2009, 2020):
                g -= 0.06
            lf, g_prev = lf + g, g
            lp = lp + pdrift + rng.normal(0, 0.01, n)
            A = np.clip(A * np.exp(rng.normal(0, 0.01, A.shape)), 0, None)
            A = A / A.sum(axis=0, keepdims=True) * np.clip(A.sum(axis=0), 0.3, 0.65)
        f = np.exp(lf)
        x = np.linalg.solve(np.eye(n) - A, f)
        p = np.exp(lp)
        Y = np.zeros((n, len(fdc)))
        for i in range(n):
            Y[i] = np.kron(dest[i], w_fd) * f[i]
            hc = ctry[i] * len(FD)
            inv = rng.normal(0, 0.01) * f[i]
            Y[i, hc + 4] = inv
            Y[i, hc + 3] += w_fd[4] * f[i] - inv
        tabs = []
        for prices in (p, p_prev):
            Zv = (prices[:, None] * A) * x[None, :]
            Yv = prices[:, None] * Y
            xv = prices * x
            tls = 0.03 * xv
            va = xv - Zv.sum(axis=0) - tls
            W = pd.DataFrame(np.vstack([np.r_[tls, np.zeros(len(fdc))], np.r_[va, np.zeros(len(fdc))]]),
                             index=["TLS", "VA"], columns=keys + fdc)
            tabs.append(Table(pd.DataFrame(Zv, keys, keys), pd.DataFrame(Yv, keys, fdc), W))
        out[y] = (tabs[0], tabs[1])  # (current prices, previous year's prices)
        p_prev = p
    return out


def write_history(hist: dict, cp_dir: Path, pyp_dir: Path) -> None:
    cp_dir, pyp_dir = Path(cp_dir), Path(pyp_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)
    pyp_dir.mkdir(parents=True, exist_ok=True)
    for y, (cp, pyp) in hist.items():
        to_wide(cp).to_csv(cp_dir / f"{y}.csv")
        to_wide(pyp).to_csv(pyp_dir / f"{y}_PYP.csv")


def synthetic_external(countries=("CAN", "USA", "IND", "ROW"), years=range(2015, 2031), seed: int = 1) -> pd.DataFrame:
    """A tidy external-projections table with WEO-, IEO- and AEO-like items."""
    rng = np.random.default_rng(seed)
    rows = []
    weo_names = {"ROW": "WEOWORLD"}
    for c in countries:
        lvl = 1.0
        for y in years:
            g = rng.normal(2.5, 0.5)
            rows.append({"source": "IMF WEO", "region": weo_names.get(c, c), "item": "GDP_R_GROWTH", "year": y, "value": g})
            rows.append({"source": "IMF WEO", "region": weo_names.get(c, c), "item": "IMP_R_GROWTH", "year": y, "value": g + 1})
            lvl *= 1 + (g + 2.0) / 100
            rows.append({"source": "IMF WEO", "region": weo_names.get(c, c), "item": "GDP_USD", "year": y, "value": 1000 * lvl})
    regions = {"CAN": "Canada", "USA": "United States", "IND": "India"}
    for c, reg in regions.items():
        for item, g in (("GO:mining", 1.0), ("GO:refining", -0.5), ("GO:basic_chemicals", 1.5),
                        ("GO:construction", 2.0), ("GO:agriculture", 1.2), ("GO:services", 2.4),
                        ("GO:industrial_total", 1.5), ("GDP_R", 2.2), ("PROD:electricity", 2.5),
                        ("PROD:liquids_gas", 0.8), ("PRICE:brent", -1.0), ("PRICE:electricity", 1.0),
                        ("ENERGY:industrial:electricity", 2.0), ("ENERGY:industrial:liquids", 0.0),
                        ("ENERGY:transportation:liquids", 0.5), ("ENERGY:buildings:electricity", 2.5)):
            for y in (2022, 2025, 2030):
                rows.append({"source": "EIA IEO", "region": reg, "item": item, "year": y,
                             "value": 100 * (1 + g / 100) ** (y - 2022)})
    for item, g in (("SHIP:bulk_chemicals", 1.8), ("SHIP:refining", -0.8)):
        for y in range(2022, 2031):
            rows.append({"source": "EIA AEO", "region": "United States", "item": item, "year": y,
                         "value": 100 * (1 + g / 100) ** (y - 2022)})
    return pd.DataFrame(rows)
