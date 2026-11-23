"""Generic multi-regional sector disaggregation for the OECD ICIO.

A *new sector* is carved out of a *parent* industry in every region:

* its gross output ``x_new`` is subtracted from the parent's output;
* its **sales row** (to every industry and final-demand column of every
  region) is given in the original column space and subtracted from the
  parent's row;
* its **cost column** (purchases from every origin-industry, plus TLS and VA)
  is given in the original row space and subtracted from the parent's column.

Cells pushed below zero by the subtraction (e.g. a C19 parent that buys no
maize) are clipped and logged. The table is then re-balanced with RAS on the
Z+Y block: row targets are gross outputs, column targets are output less VA and
TLS for industries and original totals for final-demand columns. Original
negative cells (inventory changes) stay fixed, as in the baltic routine. This
matches the proposal: disaggregate with the external shares, then ensure
matrix balance using RAS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .icio import ICIO, make_label, split_label
from .ras import ras_signed

log = logging.getLogger(__name__)


@dataclass
class SectorSplit:
    """One new sector code, applied to every region in ``output``."""
    code: str
    insert_after: str
    parent: dict[str, str]                 # region -> parent industry code
    output: dict[str, float]               # region -> gross output (USD million)
    sales: dict[str, np.ndarray] = field(default_factory=dict)   # region -> vector over old Z cols + Y cols
    inputs: dict[str, np.ndarray] = field(default_factory=dict)  # region -> vector over old Z rows
    va: dict[str, float] = field(default_factory=dict)
    tls: dict[str, float] = field(default_factory=dict)


@dataclass
class SplitResult:
    icio: ICIO
    diagnostics: pd.DataFrame
    pre_ras: ICIO | None = None
    ras_info: dict | None = None


def new_labels(icio: ICIO, splits: list[SectorSplit]) -> list[str]:
    after: dict[str, list[str]] = {}
    for s in splits:
        after.setdefault(s.insert_after, []).append(s.code)
    out = []
    for c in icio.countries:
        for sec in icio.sectors_of(c):
            out.append(make_label(c, sec))
            for new in after.get(sec, []):
                out.append(make_label(c, new))
    return out


def apply_splits(icio: ICIO, splits: list[SectorSplit], ras: bool = True,
                 ras_kw: dict | None = None, keep_pre_ras: bool = False) -> SplitResult:
    old_rows = list(icio.Z.index)
    old_cols = list(icio.Z.columns) + list(icio.Y.columns)
    labels = new_labels(icio, splits)
    fd = list(icio.Y.columns)
    nZ, nAll = len(labels), len(labels) + len(fd)
    pos = {l: i for i, l in enumerate(labels)}
    col_pos = {**pos, **{l: nZ + k for k, l in enumerate(fd)}}

    # -- reindex into the new label space --------------------------------------
    M = np.zeros((nZ, nAll))
    ri = np.array([pos[l] for l in old_rows])
    ci = np.array([col_pos[l] for l in old_cols])
    M[np.ix_(ri, ci)] = np.hstack([icio.Z.values, icio.Y.values])
    orig_neg = M < 0
    W = np.zeros((2, nAll))
    W[:, ci] = icio.W[old_cols].values
    x_old = icio.output_rows()
    x = np.zeros(nZ)
    x[ri] = x_old.values

    old_ci = ci                      # old column j -> new column position
    old_ri = ri                      # old row i    -> new row position
    diag = []
    parents_touched: set[tuple[int, int]] = set()

    for s in splits:
        for r, xb in s.output.items():
            if xb <= 0:
                continue
            p = s.parent[r]
            pr = pos[make_label(r, p)]
            nr = pos[make_label(r, s.code)]
            # ---------------- output
            x[nr] += xb
            x[pr] -= xb
            # ---------------- sales row
            row = s.sales.get(r)
            if row is not None:
                M[nr, old_ci] += row
                M[pr, old_ci] -= row
            # ---------------- cost column
            col = s.inputs.get(r)
            if col is not None:
                M[old_ri, nr] += col
                M[old_ri, pr] -= col
            # ---------------- value added and taxes
            va = s.va.get(r, 0.0)
            tls = s.tls.get(r, 0.0)
            W[1, nr] += va
            W[1, pr] -= va
            W[0, nr] += tls
            W[0, pr] -= tls
            parents_touched.add((pr, nr))
            diag.append({"region": r, "sector": s.code, "parent": p, "output": xb,
                         "parent_output_before": float(x_old.get(make_label(r, p), np.nan))})

    # -- clip cells pushed negative ---------------------------------------------
    rows_touched = sorted({pr for pr, _ in parents_touched})
    cols_touched = rows_touched
    clipped_row = {}
    for pr in rows_touched:
        bad = (M[pr] < 0) & ~orig_neg[pr]
        clipped_row[pr] = -M[pr, bad].sum()
        M[pr, bad] = 0.0
    clipped_col = {}
    for pc in cols_touched:
        bad = (M[:, pc] < 0) & ~orig_neg[:, pc]
        clipped_col[pc] = -M[bad, pc].sum()
        M[bad, pc] = 0.0
    if (x < -1e-9).any():
        raise ValueError("Negative remainder output: lower disaggregation.max_share_of_parent")

    d = pd.DataFrame(diag)
    if not d.empty:
        d["share_of_parent"] = d.output / d.parent_output_before
        d["parent_row_clipped"] = [clipped_row[pos[make_label(r, p)]] for r, p in zip(d.region, d.parent)]
        d["parent_col_clipped"] = [clipped_col[pos[make_label(r, p)]] for r, p in zip(d.region, d.parent)]
        for c in ("parent_row_clipped", "parent_col_clipped"):
            d[c + "_pct_of_output"] = 100 * d[c] / d.parent_output_before
        big = d[(d.parent_row_clipped_pct_of_output > 1) | (d.parent_col_clipped_pct_of_output > 1)]
        if len(big):
            log.info("Clipping >1%% of parent output in %d region-sector cells (resolved by RAS); "
                     "see diagnostics", big.groupby(["region", "parent"]).ngroups)

    def build(Mx, Wx):
        Z = pd.DataFrame(Mx[:, :nZ], index=labels, columns=labels)
        Y = pd.DataFrame(Mx[:, nZ:], index=labels, columns=fd)
        Wd = pd.DataFrame(Wx, index=["TLS", "VA"], columns=labels + fd)
        return ICIO(Z=Z, Y=Y, W=Wd)

    pre = build(M.copy(), W.copy()) if keep_pre_ras else None
    info = None
    if ras:
        row_t = x
        col_t = np.concatenate([x - W[:, :nZ].sum(axis=0), M[:, nZ:].sum(axis=0)])
        # FD column totals: preserve original totals (Y + TLS on final products fixed)
        col_t[nZ:] = np.concatenate([icio.Y.values.sum(axis=0)])
        kw = dict(max_iter=10000, tol=1e-7)
        kw.update(ras_kw or {})
        gap = row_t.sum() - col_t.sum()
        if abs(gap) > 1e-6 * max(1.0, abs(row_t.sum())):
            log.warning("Row/column target totals differ by %.4f (rescaled inside RAS)", gap)
        Mb, info = ras_signed(M, row_t, col_t, label="ICIO disaggregation", **kw)
        M = Mb
    out = build(M, W)
    return SplitResult(icio=out, diagnostics=d, pre_ras=pre, ras_info=info)


# ----------------------------------------------------------------- helpers
def column_groups(icio: ICIO, region: str, spec: dict, all_sectors: list[str]) -> tuple[list[str], list[str]]:
    """Resolve a user-group spec to (industry labels, FD labels) of ``region``."""
    inds = spec.get("industries", "ALL")
    if isinstance(inds, str):
        if inds == "ALL":
            inds = all_sectors
        elif inds.startswith("ALL_EXCEPT:"):
            ex = set(inds.split(":", 1)[1].split(","))
            inds = [s for s in all_sectors if s not in ex]
        else:
            inds = [inds]
    ind_labels = [make_label(region, s) for s in inds if make_label(region, s) in icio.Z.columns]
    fd_labels = [make_label(region, f) for f in spec.get("final_demand", [])
                 if make_label(region, f) in icio.Y.columns]
    return ind_labels, fd_labels


def product_rows(icio: ICIO, product: str) -> list[str]:
    return [l for l in icio.Z.index if split_label(l)[1] == product]


def origin_mix(icio: ICIO, region: str, product: str) -> pd.Series:
    """Origin shares of ``region``'s intermediate purchases of ``product``."""
    rows = product_rows(icio, product)
    cols = [l for l in icio.Z.columns if l[:3] == region]
    v = icio.Z.loc[rows, cols].sum(axis=1).clip(lower=0)
    if v.sum() <= 0:
        v = pd.Series(0.0, index=rows)
        v[make_label(region, product)] = 1.0
    return v / v.sum()
