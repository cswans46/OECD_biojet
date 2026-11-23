"""Projection of the (disaggregated) ICIO from 2022 to 2030: interface only.

The proposal combines the OECD projection approach (OECD, Trends in Global
Value Chains, ICIO projections in current and previous year's prices) with
ECCC sources (E3MC for Canada, plus IEO/AEO/EPA). E3MC and the related ECCC
projection data are proprietary, so this module provides only:

* ``load_growth``: reads a user-supplied template
  ``data/projections/output_index.csv`` with columns
  ``year, country, sector, output_index`` (2022 = 1) and optionally
  ``fd_index.csv`` (``year, country, fd, index``) and
  ``coefficient_changes.csv`` (``year, country, input, sector, factor``) for
  technology change such as biojet blending mandates;
* ``project_year``: applies coefficient changes, scales outputs and final
  demand, and re-balances with RAS (gross-output row targets, cost column
  targets), as in the OECD projection approach.

Wire your E3MC/IEO/AEO/EPA routine in by writing those csv files.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .icio import ICIO, make_label
from .ras import ras_signed

log = logging.getLogger(__name__)


def load_growth(proj_dir: Path) -> dict[str, pd.DataFrame]:
    proj_dir = Path(proj_dir)
    out = {}
    for name in ("output_index", "fd_index", "coefficient_changes"):
        f = proj_dir / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f)
    if "output_index" not in out:
        raise FileNotFoundError(f"{proj_dir}/output_index.csv not found: fill the template first")
    return out


def project_year(base: ICIO, year: int, growth: dict[str, pd.DataFrame], ras_kw: dict | None = None) -> ICIO:
    g = growth["output_index"]
    g = g[g.year == year]
    idx = pd.Series(1.0, index=base.Z.index)
    for _, r in g.iterrows():
        lab = make_label(r.country, r.sector)
        if lab in idx.index:
            idx[lab] = r.output_index
    x0 = base.output_rows()
    x1 = x0 * idx
    A = base.Z.div(x0.where(x0 > 0), axis=1).fillna(0.0)
    if "coefficient_changes" in growth:
        cc = growth["coefficient_changes"]
        for _, r in cc[cc.year == year].iterrows():
            rows = [l for l in A.index if l.endswith("_" + r.input)]
            col = make_label(r.country, r.sector)
            if col in A.columns:
                A.loc[rows, col] *= r.factor
    Z1 = A * x1.values[None, :]
    W1 = base.W.copy()
    w_ratio = base.W[base.Z.columns].div(x0.where(x0 > 0), axis=1).fillna(0.0)
    W1[base.Z.columns] = w_ratio * x1.values[None, :]
    Y1 = base.Y.copy()
    if "fd_index" in growth:
        fdg = growth["fd_index"]
        for _, r in fdg[fdg.year == year].iterrows():
            c = make_label(r.country, r.fd)
            if c in Y1.columns:
                Y1[c] *= r["index"]
                W1[c] *= r["index"]
    else:
        Y1 = Y1 * (x1.sum() / x0.sum())
    M = np.hstack([Z1.values, Y1.values])
    nZ = len(x1)
    col_t = np.concatenate([x1.values - W1[base.Z.columns].sum().values, Y1.values.sum(axis=0)])
    kw = dict(max_iter=10000, tol=1e-6)
    kw.update(ras_kw or {})
    Mb, info = ras_signed(M, x1.values, col_t, label=f"projection {year}", **kw)
    return ICIO(Z=pd.DataFrame(Mb[:, :nZ], base.Z.index, base.Z.columns),
                Y=pd.DataFrame(Mb[:, nZ:], base.Y.index, base.Y.columns), W=W1)


def write_templates(proj_dir: Path, icio: ICIO, years=range(2023, 2031)) -> None:
    proj_dir = Path(proj_dir)
    proj_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"year": y, "country": l[:3], "sector": l[4:], "output_index": 1.0}
            for y in years for l in icio.Z.index]
    pd.DataFrame(rows).to_csv(proj_dir / "output_index_TEMPLATE.csv", index=False)
    pd.DataFrame([{"year": 2030, "country": "CAN", "input": "BJT", "sector": "H51", "factor": 1.0}]).to_csv(
        proj_dir / "coefficient_changes_TEMPLATE.csv", index=False)
