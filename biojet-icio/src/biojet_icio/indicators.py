"""Leontief-based TiVA, GHG-footprint and pass-through indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .icio import ICIO, split_label


def leontief(icio: ICIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = icio.output_rows().values
    xinv = np.divide(1.0, x, out=np.zeros_like(x), where=x > 1e-12)
    A = icio.Z.values * xinv[None, :]
    L = np.linalg.inv(np.eye(len(x)) - A)
    lab = icio.Z.index
    return pd.DataFrame(A, lab, lab), pd.DataFrame(L, lab, lab)


def fd_by_country(icio: ICIO) -> pd.DataFrame:
    """Final demand aggregated to destination country (rows: industries)."""
    Y = icio.Y.copy()
    Y.columns = [c[:3] for c in Y.columns]
    return Y.T.groupby(level=0, sort=False).sum().T


def va_coefficients(icio: ICIO) -> pd.Series:
    x = icio.output_rows()
    return (icio.W.loc["VA", icio.Z.columns] / x.where(x > 0)).fillna(0.0)


def tiva(icio: ICIO, L: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Core TiVA-style measures.

    * ``va_fd``   : VA of origin country-industry embodied in final demand of
                    each destination country (V-hat L Y);
    * ``FFD_DVA`` : domestic VA in foreign final demand, by origin country;
    * ``EXGR``, ``EXGR_DVA``, ``EXGR_FVA``: gross exports and their
      domestic/foreign VA content (v L e_r, global-inverse split).
    """
    if L is None:
        _, L = leontief(icio)
    v = va_coefficients(icio)
    Yc = fd_by_country(icio)
    VAFD = pd.DataFrame((v.values[:, None] * L.values) @ Yc.values, index=icio.Z.index, columns=Yc.columns)
    origin = [l[:3] for l in VAFD.index]
    by_c = VAFD.groupby(origin, sort=False).sum()
    ffd = pd.Series({c: by_c.loc[c].drop(c, errors="ignore").sum() for c in by_c.index}, name="FFD_DVA")
    # gross exports by exporting industry
    cty = np.array([l[:3] for l in icio.Z.index])
    zc = np.array([l[:3] for l in icio.Z.columns])
    yc = np.array([l[:3] for l in icio.Y.columns])
    ex = (icio.Z.values * (cty[:, None] != zc[None, :])).sum(axis=1) + \
         (icio.Y.values * (cty[:, None] != yc[None, :])).sum(axis=1)
    rows = []
    vL = v.values[:, None] * L.values
    for c in dict.fromkeys(cty):
        mask = cty == c
        e = np.where(mask, ex, 0.0)
        va_in = vL @ e
        dva = va_in[mask].sum()
        rows.append({"country": c, "EXGR": e.sum(), "EXGR_DVA": dva, "EXGR_FVA": va_in[~mask].sum()})
    return {"va_fd": VAFD, "FFD_DVA": ffd.to_frame(), "exports": pd.DataFrame(rows).set_index("country")}


def ghg_footprints(icio: ICIO, e: pd.Series, L: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Production- vs consumption-based emissions and product multipliers."""
    if L is None:
        _, L = leontief(icio)
    x = icio.output_rows()
    f = (e.reindex(x.index).fillna(0.0) / x.where(x > 0)).fillna(0.0)
    m = pd.Series(f.values @ L.values, index=x.index, name="total_intensity")   # kt per USD million
    Yc = fd_by_country(icio)
    emb = pd.DataFrame(f.values[:, None] * (L.values @ Yc.values), index=x.index, columns=Yc.columns)
    cba = emb.sum(axis=0)
    hh = e[[i for i in e.index if i.endswith("_HH")]]
    if len(hh):
        cba = cba.add(pd.Series(hh.values, index=[i[:3] for i in hh.index]), fill_value=0.0)
    origin = [l[:3] for l in emb.index]
    pba = e.reindex(x.index).fillna(0.0).groupby(origin, sort=False).sum()
    return {"multipliers": m.to_frame(), "embodied": emb,
            "summary": pd.DataFrame({"production_based": pba, "consumption_based": cba})}


def pass_through_frequency(L: pd.DataFrame, v: pd.Series, y: pd.Series,
                           origins: list[str], through: list[str]) -> float:
    """Average number of times value added generated in ``origins`` passes
    through the ``through`` industries before reaching final demand ``y``.

    Uses T_ikj = b_ik b_kj / b_ij (expected visits to k on the paths from i to
    j), weighted by the VA flows v_i b_ij y_j. This follows the frequency
    logic of the OECD pass-through-frequency (PTF) indicator, restricted to a
    chosen origin set (e.g. OPEC petroleum) and pass-through set (e.g. H51)."""
    B = L.values
    idx = {l: i for i, l in enumerate(L.index)}
    o = [idx[l] for l in origins if l in idx]
    k = [idx[l] for l in through if l in idx]
    vy = v.reindex(L.index).fillna(0.0).values
    yy = y.reindex(L.index).fillna(0.0).values
    num = 0.0
    for kk in k:
        num += (vy[o] @ B[o, kk]) * (B[kk, :] @ yy)
    den = vy[o] @ (B[o, :] @ yy)
    return float(num / den) if den > 0 else 0.0


def cost_push(A: pd.DataFrame, shocked: list[str], dp: float | dict[str, float]) -> pd.Series:
    """Leontief price model: exogenous proportional price change on ``shocked``
    industries; returns proportional price change of all other industries."""
    lab = list(A.index)
    S = [lab.index(s) for s in shocked if s in lab]
    N = [i for i in range(len(lab)) if i not in set(S)]
    dps = np.array([dp[lab[i]] if isinstance(dp, dict) else dp for i in S])
    Av = A.values
    ANN = Av[np.ix_(N, N)]
    ASN = Av[np.ix_(S, N)]
    dpn = (dps @ ASN) @ np.linalg.inv(np.eye(len(N)) - ANN)
    out = pd.Series(0.0, index=lab)
    out.iloc[S] = dps
    out.iloc[N] = dpn
    return out
