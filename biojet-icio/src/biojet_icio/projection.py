"""Project the (disaggregated) ICIO from 2022 to 2023-2030.

Follows the OECD ICIO projection logic: scale gross output and final demand
with external drivers, keep technical coefficients (optionally changed by
``coefficient_changes``), then restore consistency with RAS. Each target year
is projected directly from the 2022 base, not chained, so errors don't
accumulate.

Inputs in ``data/projections`` (all optional except output_index):

* ``output_index.csv``: year, country, sector, output_index[, output_level_musd]
  built from public sources by :mod:`projection_inputs`, Canada excluded;
* ``output_index_CAN.csv``: same layout, your E3MC-based Canada drivers;
* ``fd_index.csv``: year, country, fd, index;
* ``coefficient_changes.csv``: year, country, input, sector, factor.

Countries with no rows (Canada until you add its file) are held at their
2022 values, and the log says so. A sector whose 2022 output is zero but
which gets an ``output_level_musd`` (e.g. BJT in Singapore by 2030) is
seeded with the output-weighted average cost structure of that sector
elsewhere, and its sales go to the country's own H51 (for BJT) or its C19
users (other fuels). RAS then fits it into the table.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .disaggregate import origin_mix
from .icio import ICIO, make_label, split_label
from .ras import ras_signed

log = logging.getLogger(__name__)
SEED_BUYER = {"BJT": "H51"}


def load_growth(proj_dir: Path) -> dict[str, pd.DataFrame]:
    proj_dir = Path(proj_dir)
    out = {}
    for name in ("output_index", "output_index_CAN", "fd_index", "coefficient_changes"):
        f = proj_dir / f"{name}.csv"
        if f.exists():
            out[name] = pd.read_csv(f)
    if "output_index" not in out:
        raise FileNotFoundError(f"{proj_dir}/output_index.csv not found: run scripts/build_projection_inputs.py")
    if "output_index_CAN" in out:
        out["output_index"] = pd.concat([out["output_index"][out["output_index"].country != "CAN"],
                                         out["output_index_CAN"]], ignore_index=True)
    return out


def _seed_column(base: ICIO, country: str, sector: str) -> tuple[np.ndarray, float, float]:
    """Average input coefficients (by product) of ``sector`` across producers, re-sourced
    with ``country``'s origin mix. Returns (coefficient column, VA ratio, TLS ratio)."""
    x = base.output_rows()
    prods = [l for l in base.Z.columns if split_label(l)[1] == sector and x[l] > 0]
    if not prods:
        raise ValueError(f"cannot seed {sector}: no producer in base table")
    w = x[prods] / x[prods].sum()
    A = base.Z[prods].div(x[prods], axis=1)
    by_prod = (A * w.values[None, :]).sum(axis=1).groupby(lambda l: split_label(l)[1]).sum()
    col = pd.Series(0.0, index=base.Z.index)
    for prod, a in by_prod.items():
        if a <= 0:
            continue
        mix = origin_mix(base, country, prod)
        col[mix.index] += a * mix.values
    va = float((base.W.loc["VA", prods] / x[prods] * w).sum())
    tls = float((base.W.loc["TLS", prods] / x[prods] * w).sum())
    return col.values, va, tls


def project_year(base: ICIO, year: int, growth: dict[str, pd.DataFrame], ras_kw: dict | None = None) -> ICIO:
    g = growth["output_index"]
    g = g[g.year == year]
    labels = list(base.Z.index)
    x0 = base.output_rows()
    idx = pd.Series(1.0, index=labels)
    levels = {}
    covered = set(g.country)
    for c in base.countries:
        if c not in covered:
            log.warning("%d: no projection drivers for %s; held at 2022 (add output_index_%s.csv)", year, c, c)
    for r in g.itertuples(index=False):
        lab = make_label(r.country, r.sector)
        if lab not in idx.index:
            continue
        lvl = getattr(r, "output_level_musd", np.nan)
        if x0[lab] <= 0 and pd.notna(lvl) and lvl > 0:
            levels[lab] = float(lvl)
        elif pd.notna(r.output_index):
            idx[lab] = float(r.output_index)
    x1 = x0 * idx
    A = base.Z.div(x0.where(x0 > 0), axis=1).fillna(0.0)
    wr = base.W[base.Z.columns].div(x0.where(x0 > 0), axis=1).fillna(0.0)

    # seed zero-base sectors that get a level
    Y = base.Y.copy()
    row_seed = {}
    for lab, lvl in levels.items():
        c, sec = split_label(lab)
        col, va, tls = _seed_column(base, c, sec)
        A[lab] = col
        wr.loc["VA", lab], wr.loc["TLS", lab] = va, tls
        x1[lab] = lvl
        buyer = make_label(c, SEED_BUYER.get(sec, "C19"))
        row_seed[lab] = buyer
        log.info("%d: seeded %s at %.1f USD m (sales to %s)", year, lab, lvl, buyer)

    if "coefficient_changes" in growth:
        cc = growth["coefficient_changes"]
        for r in cc[cc.year == year].itertuples(index=False):
            rows = [l for l in A.index if split_label(l)[1] == r.input]
            col = make_label(r.country, r.sector)
            if col in A.columns:
                A.loc[rows, col] *= r.factor

    Z1 = A * x1.values[None, :]
    for lab, buyer in row_seed.items():
        Z1.loc[lab, buyer] += 0.9 * x1[lab]      # seed; RAS adjusts
    W1 = base.W.copy()
    W1[base.Z.columns] = wr * x1.values[None, :]

    # final demand
    if "fd_index" in growth:
        fdg = growth["fd_index"]
        fdg = fdg[fdg.year == year]
        for r in fdg.itertuples(index=False):
            col = make_label(r.country, r.fd)
            if col in Y.columns:
                Y[col] *= r.index
                W1[col] *= r.index
    # rows whose base is zero but now have output need some FD support to reach targets
    for lab in levels:
        if Y.loc[lab].abs().sum() == 0:
            Y.loc[lab, make_label(lab[:3], "HFCE")] = 0.1 * x1[lab]

    M = np.hstack([Z1.values, Y.values])
    nZ = len(labels)
    col_t = np.concatenate([x1.values - W1[base.Z.columns].sum().values, Y.values.sum(axis=0)])
    gap = x1.sum() - col_t.sum()
    # The gap between gross output and (inputs + final demand) after independent scaling
    # is closed by scaling final demand (GDP-side identity), as in the OECD approach.
    fd_tot = Y.values.sum()
    if fd_tot != 0:
        col_t[nZ:] *= 1 + gap / fd_tot
    kw = dict(max_iter=10000, tol=1e-6)
    kw.update(ras_kw or {})
    Mb, info = ras_signed(M, x1.values, col_t, label=f"projection {year}", **kw)
    log.info("%d: RAS %d it, err %.2e; FD rescaled by %+.2f%%", year, info["iterations"], info["error"],
             100 * gap / fd_tot if fd_tot else 0.0)
    return ICIO(Z=pd.DataFrame(Mb[:, :nZ], base.Z.index, base.Z.columns),
                Y=pd.DataFrame(Mb[:, nZ:], base.Y.index, base.Y.columns), W=W1)


def project_all(base: ICIO, years, growth: dict[str, pd.DataFrame], ras_kw: dict | None = None) -> dict[int, ICIO]:
    return {y: project_year(base, y, growth, ras_kw) for y in years}


def write_templates(proj_dir: Path, icio: ICIO, years=range(2023, 2031)) -> None:
    proj_dir = Path(proj_dir)
    proj_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"year": y, "country": "CAN", "sector": l[4:], "output_index": "", "output_level_musd": "",
             "source": "E3MC"} for y in years for l in icio.Z.index if l[:3] == "CAN"]
    pd.DataFrame(rows).to_csv(proj_dir / "output_index_CAN_TEMPLATE.csv", index=False)
    pd.DataFrame([{"year": 2030, "country": "CAN", "input": "BJT", "sector": "H51", "factor": 1.0}]).to_csv(
        proj_dir / "coefficient_changes_TEMPLATE.csv", index=False)
