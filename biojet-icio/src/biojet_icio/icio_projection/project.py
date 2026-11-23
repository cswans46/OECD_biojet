"""Year-by-year projection and RAS balancing, in previous year's prices (PYP)
and current prices (CP), 2023-2030.

For each year t, starting from the balanced current-price table of t-1:

PYP step (volumes at t-1 prices)
    seed   Z = A_{t-1} * x_t^pyp           (t-1 input coefficients)
           Y = Y_{t-1} * fd volume growth  (per FD column)
           energy-coefficient factors (external method) applied to A
    pre-balance: each country's imported cells are scaled to its import
           volume target (the OECD's pre-balancing RAS on trade)
    RAS    rows  = x_t^pyp
           cols  = industry intermediate inputs x^pyp - VA^pyp - TLS^pyp,
                   final-demand totals (scaled to close the GDP identity)
CP step (reflate by row prices)
    seed   diag(p_t / p_{t-1}) [Z Y]^pyp
    RAS    rows  = x_t^pyp * p_t/p_{t-1};  industry cols = x^cp - VA^cp - TLS^cp
           (VA and TLS reflated with the industry's output price)

Negative cells (inventory changes) are held at their seed value by the
signed RAS (from cswans46/baltic). Both tables are written per year.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .methods import Projected
from .ras import ras_signed
from .table import Table, country_of, save_table, sector_of
from . import external as ex

log = logging.getLogger(__name__)


@dataclass
class ChainResult:
    method: str
    years: list[int]
    margins: pd.DataFrame                    # long: year, prices, var, key, value
    ras_log: pd.DataFrame
    tables: dict = field(default_factory=dict)  # (year, prices) -> Table or path


def _ras(Z: np.ndarray, Y: np.ndarray, rows: np.ndarray, cols_ind: np.ndarray, label: str, ras_kw: dict):
    M = np.hstack([Z, Y])
    col_t = np.concatenate([cols_ind, Y.sum(axis=0)])
    # close the identity on the FD columns (GDP = final demand)
    gap = rows.sum() - col_t.sum()
    fd_tot = col_t[Z.shape[1]:].sum()
    if fd_tot != 0:
        col_t[Z.shape[1]:] *= 1 + gap / fd_tot
    kw = {"max_iter": 10000, "tol": 1e-7, **ras_kw}
    Mb, info = ras_signed(M, rows, col_t, label=label, **kw)
    info["identity_gap_share"] = float(gap / max(abs(rows.sum()), 1e-9))
    return Mb[:, :Z.shape[1]], Mb[:, Z.shape[1]:], info


def _prebalance_imports(Z: np.ndarray, Y: np.ndarray, keys, fd_cols, target: dict[str, float]) -> None:
    rc = np.array([country_of(k) for k in keys])
    zc = np.array([country_of(k) for k in keys])
    yc = np.array([country_of(c) for c in fd_cols])
    for c, tgt in target.items():
        rows = rc != c
        zm, ym = zc == c, yc == c
        cur = Z[np.ix_(rows, zm)].sum() + np.clip(Y[np.ix_(rows, ym)], 0, None).sum()
        if cur > 0 and np.isfinite(tgt) and tgt > 0:
            f = tgt / cur
            Z[np.ix_(rows, zm)] *= f
            sub = Y[np.ix_(rows, ym)]
            Y[np.ix_(rows, ym)] = np.where(sub > 0, sub * f, sub)


def _apply_coef(A: np.ndarray, keys, cols_ind, fuel_rows: dict[str, np.ndarray], factor: pd.DataFrame | None) -> None:
    if factor is None:
        return
    for fuel, rmask in fuel_rows.items():
        if fuel not in factor.index:
            continue
        f = factor.loc[fuel, cols_ind].to_numpy(float)
        A[rmask, :] *= f[None, :]


def run_chain(base: Table, proj: Projected, out_dir: Path | None = None, ras_kw: dict | None = None,
              keep_tables: bool = False, write_rdata: bool = False) -> ChainResult:
    ras_kw = ras_kw or {}
    keys = base.keys
    fd_cols = list(base.Y.columns)
    n = len(keys)
    x_b = base.x().to_numpy(float)
    va_b = base.W.loc["VA", keys].to_numpy(float)
    tls_b = base.W.loc["TLS", keys].to_numpy(float) if "TLS" in base.W.index else np.zeros(n)
    countries = list(dict.fromkeys(country_of(k) for k in keys))
    rc = np.array([country_of(k) for k in keys])
    fuel_rows = {f: np.array([ex.first_match(sector_of(k), [(p, f)]) is not None for k in keys])
                 for p, f in ex.ENERGY_INPUT}

    def imports_of(t: Table) -> dict[str, float]:
        Z, Y = t.Z.to_numpy(), t.Y.to_numpy()
        yc = np.array([country_of(c) for c in fd_cols])
        return {c: Z[np.ix_(rc != c, rc == c)].sum() + np.clip(Y[np.ix_(rc != c, yc == c)], 0, None).sum()
                for c in countries}

    prev = base
    prev_year = proj.years[0] - 1
    g_prev = {"gx": np.ones(n), "gva": np.ones(n), "px": np.ones(n),
              "gfd": np.ones(len(fd_cols)), "pfd": np.ones(len(fd_cols)),
              "gimp": pd.Series(1.0, index=proj.gimp.index), "coef": None}
    imp_prev = imports_of(base)
    rows_out, ras_rows, tables = [], [], {}

    for t in proj.years:
        gx = proj.gx[t].to_numpy(float)
        gva = proj.gva[t].to_numpy(float)
        px = proj.px[t].to_numpy(float)
        gfd = proj.gfd[t].reindex(fd_cols).fillna(1.0).to_numpy(float)
        pfd = proj.pfd[t].reindex(fd_cols).fillna(1.0).to_numpy(float)
        coef = proj.coef.get(t)

        xp = prev.x().to_numpy(float)
        A = np.divide(prev.Z.to_numpy(), xp[None, :], out=np.zeros((n, n)), where=xp[None, :] > 0)
        if coef is not None:
            rel = coef.copy()
            if g_prev["coef"] is not None:
                rel = rel / g_prev["coef"].reindex_like(rel).replace(0, np.nan)
                rel = rel.fillna(1.0)
            _apply_coef(A, keys, keys, fuel_rows, rel)

        # ---- PYP targets
        vol_ratio = np.divide(gx, g_prev["gx"], out=np.ones(n), where=g_prev["gx"] > 0)
        x_pyp = xp * vol_ratio
        va_prev = prev.W.loc["VA", keys].to_numpy(float)
        tls_prev = prev.W.loc["TLS", keys].to_numpy(float) if "TLS" in prev.W.index else np.zeros(n)
        va_ratio = np.divide(gva, g_prev["gva"], out=np.ones(n), where=g_prev["gva"] > 0)
        va_pyp = va_prev * va_ratio
        tls_pyp = np.divide(tls_prev, xp, out=np.zeros(n), where=xp > 0) * x_pyp
        ii_pyp = x_pyp - va_pyp - tls_pyp
        neg = ii_pyp < 0
        if neg.any():
            va_pyp[neg] = np.maximum(x_pyp[neg] - tls_pyp[neg], 0) * 0.98
            ii_pyp = x_pyp - va_pyp - tls_pyp

        Z = A * x_pyp[None, :]
        fd_ratio = np.divide(gfd, g_prev["gfd"], out=np.ones_like(gfd), where=g_prev["gfd"] != 0)
        Y = prev.Y.to_numpy(float) * fd_ratio[None, :]
        if coef is not None:
            for fuel, rmask in fuel_rows.items():
                if fuel in rel.index:
                    Y[rmask, :] *= rel.loc[fuel, fd_cols].to_numpy(float)[None, :]
        imp_ratio = (proj.gimp[t] / g_prev["gimp"].reindex(proj.gimp.index).replace(0, np.nan)).fillna(1.0)
        imp_target = {c: imp_prev.get(c, 0.0) * float(imp_ratio.get(c, 1.0)) for c in countries}
        _prebalance_imports(Z, Y, keys, fd_cols, imp_target)
        Zp, Yp, info = _ras(Z, Y, x_pyp, ii_pyp, f"{proj.method} {t} PYP", ras_kw)
        ras_rows.append({"year": t, "prices": "pyp", **info})

        Wp = prev.W.copy()
        Wp.loc["VA", keys] = va_pyp
        if "TLS" in Wp.index:
            Wp.loc["TLS", keys] = tls_pyp
        Wp[fd_cols] = prev.W[fd_cols].to_numpy() * fd_ratio[None, :]
        pyp = Table(pd.DataFrame(Zp, keys, keys), pd.DataFrame(Yp, keys, fd_cols), Wp)

        # ---- CP step
        pr = np.divide(px, g_prev["px"], out=np.ones(n), where=g_prev["px"] > 0)
        Zc0, Yc0 = Zp * pr[:, None], Yp * pr[:, None]
        x_cp = x_pyp * pr
        va_cp, tls_cp = va_pyp * pr, tls_pyp * pr
        Zc, Yc, info = _ras(Zc0, Yc0, x_cp, x_cp - va_cp - tls_cp, f"{proj.method} {t} CP", ras_kw)
        ras_rows.append({"year": t, "prices": "cp", **info})
        Wc = Wp.copy()
        Wc.loc["VA", keys] = va_cp
        if "TLS" in Wc.index:
            Wc.loc["TLS", keys] = tls_cp
        pfd_ratio = np.divide(pfd, g_prev["pfd"], out=np.ones_like(pfd), where=g_prev["pfd"] > 0)
        Wc[fd_cols] = Wp[fd_cols].to_numpy() * pfd_ratio[None, :]
        cp = Table(pd.DataFrame(Zc, keys, keys), pd.DataFrame(Yc, keys, fd_cols), Wc)

        for prices, tab in (("pyp", pyp), ("cp", cp)):
            xs = tab.x()
            rows_out.append(pd.DataFrame({"year": t, "prices": prices, "var": "x", "key": keys, "value": xs.to_numpy()}))
            rows_out.append(pd.DataFrame({"year": t, "prices": prices, "var": "va", "key": keys,
                                          "value": tab.W.loc["VA", keys].to_numpy()}))
            rows_out.append(pd.DataFrame({"year": t, "prices": prices, "var": "fd", "key": fd_cols,
                                          "value": tab.Y.sum(axis=0).to_numpy()}))
            if out_dir is not None:
                save_table(tab, Path(out_dir) / proj.method / f"{t}_{prices}.npz")
                if write_rdata and prices == "cp":
                    _write_rdata(tab, Path(out_dir) / proj.method, t)
            if keep_tables:
                tables[(t, prices)] = tab

        log.info("%s %s: max imbalance CP %.3g", proj.method, t, cp.max_imbalance())
        imp_prev = imports_of(cp)
        g_prev = {"gx": gx, "gva": gva, "px": px, "gfd": gfd, "pfd": pfd,
                  "gimp": proj.gimp[t], "coef": coef}
        prev = cp

    return ChainResult(proj.method, proj.years, pd.concat(rows_out, ignore_index=True),
                       pd.DataFrame(ras_rows), tables)


def _write_rdata(tab: Table, out_dir: Path, year: int) -> None:  # pragma: no cover - needs pyreadr
    """A, X, FD in the Datacenters RData layout (X as one-row frame by year)."""
    try:
        import pyreadr
    except ImportError:
        log.warning("pyreadr missing: RData not written")
        return
    A, x, FD = tab.to_axfd()
    out_dir.mkdir(parents=True, exist_ok=True)
    pyreadr.write_rdata(str(out_dir / f"ICIOecon_A_{year}.RData"), A.reset_index(names="io_key"), df_name="A")
    pyreadr.write_rdata(str(out_dir / f"ICIOecon_X_{year}.RData"), pd.DataFrame([x.to_numpy()], columns=x.index, index=[str(year)]), df_name="X")
    pyreadr.write_rdata(str(out_dir / f"ICIOecon_FD_{year}.RData"), FD.reset_index(names="io_key"), df_name="FD")
