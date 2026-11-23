"""Bilateral biofuel trade matrices (tonnes and USD million) by ICIO region.

Totals come from the UNSD energy balances (production, imports, exports, in
kt). These are closer than customs data to the ICIO trade concept, which strips
re-exports. Destination shares come from UN Comtrade: the HS codes mapped to
the fuel, or a proxy HS pattern for fuels without an HS6 code (bio jet).

For each fuel:
  1. Row target E_r = min(UNSD exports_r, production_r), so re-exports are excluded.
  2. Column target M_s = UNSD imports_s, or the Comtrade mirror estimate where
     the importer does not report to UNSD.
  3. Seed: Comtrade destination shares x E_r for exporters with Comtrade data;
     importer-size shares otherwise. The diagonal is zero.
  4. RAS to the targets, with column totals rescaled to world exports.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .ras import ras_signed

log = logging.getLogger(__name__)


def destination_shares(ct_icio: pd.DataFrame, hs_codes: list[str], weights: dict[str, float],
                       regions: list[str]) -> pd.DataFrame:
    """Exporter x destination share matrix from Comtrade net weight (tonnes) of bio content."""
    s = ct_icio[(ct_icio.flowCode == "X") & ct_icio.cmdCode.isin(hs_codes)].copy()
    if s.empty:
        return pd.DataFrame(0.0, index=regions, columns=regions)
    s["t"] = s.netWgt / 1000.0 * s.cmdCode.map(weights).fillna(1.0)
    m = s.pivot_table(index="reporter", columns="partner", values="t", aggfunc="sum", fill_value=0.0)
    m = m.reindex(index=regions, columns=regions, fill_value=0.0).astype(float)
    arr = m.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    m = pd.DataFrame(arr, index=regions, columns=regions)
    tot = m.sum(axis=1)
    out = m.div(tot.where(tot > 0), axis=0).fillna(0.0)
    out.attrs["tonnes"] = tot
    return out


def build_trade_matrix(bal: pd.DataFrame, biofuel: str, shares: pd.DataFrame,
                       regions: list[str], ras_kw: dict | None = None,
                       aggregates: tuple[str, ...] = ("ROW",)) -> pd.DataFrame:
    """Return the (exporter x importer) tonnes matrix for one biofuel (kt).

    Regions in ``aggregates`` (ROW) bundle many reporting countries whose
    mutual trade is internal to the aggregate. Their diagonal cell may be filled
    during RAS and is then removed (it becomes domestic sales)."""
    b = bal[bal.biofuel == biofuel].set_index("icio").reindex(regions).fillna(0.0)
    E = np.minimum(b.exports.values, b.production.values)
    M = b.imports.values.astype(float).copy()

    seed = np.zeros((len(regions), len(regions)))
    ct_t = shares.attrs.get("tonnes", pd.Series(dtype=float)).reindex(regions).fillna(0.0).values / 1000.0
    # use Comtrade destinations only where they cover >= 10% of the exporter's UNSD volume
    has_ct = (shares.sum(axis=1).values > 0) & (ct_t >= 0.1 * E)
    weak = [r for r, h, e in zip(regions, shares.sum(axis=1).values > 0, has_ct) if h and not e]
    if weak:
        log.info("trade %s: Comtrade coverage <10%% of UNSD exports for %s; importer-size seed used",
                 biofuel, ", ".join(weak))
    imp_w = M / M.sum() if M.sum() > 0 else np.ones(len(regions)) / len(regions)
    agg = np.array([r in aggregates for r in regions])
    for i in range(len(regions)):
        if E[i] <= 0:
            continue
        if has_ct[i]:
            seed[i] = shares.values[i] * E[i]
        else:
            w = imp_w.copy()
            if not agg[i]:
                w[i] = 0.0
            seed[i] = (w / w.sum() if w.sum() > 0 else 0.0) * E[i]
        if agg[i]:
            seed[i, i] = max(seed[i, i], 0.5 * min(E[i], M[i]))

    # importers absent from UNSD: use the mirror estimate from the seed
    mirror = seed.sum(axis=0)
    M = np.where(M > 0, M, mirror)
    # a destination with a target but no seed support gets a small proportional seed
    if E.sum() <= 0 or M.sum() <= 0:
        return pd.DataFrame(0.0, index=regions, columns=regions)
    support = (seed.sum(axis=0) <= 0) & (M > 0)
    if support.any():
        expw = E / E.sum()
        for j in np.where(support)[0]:
            col = expw.copy()
            col[j] = 0.0
            if col.sum() > 0:
                seed[:, j] = col / col.sum() * M[j] * 1e-3
    kw = dict(max_iter=5000, tol=1e-9)
    kw.update(ras_kw or {})
    T, info = ras_signed(seed, E, M, label=f"trade {biofuel}", **kw)
    internal = np.diag(T).copy()
    np.fill_diagonal(T, 0.0)
    if internal.sum() > 0:
        log.info("trade %s: %.0f kt traded inside aggregate regions treated as domestic",
                 biofuel, internal.sum())
    log.info("trade %s: world exports %.0f kt, RAS %d it", biofuel, T.sum(), info["iterations"])
    return pd.DataFrame(T, index=regions, columns=regions)


def build_all(cfg, bal: pd.DataFrame, ct_icio: pd.DataFrame, regions: list[str]) -> dict[str, pd.DataFrame]:
    hs = cfg.table("hs")
    if not cfg.settings["trade"].get("include_271020", False):
        hs = hs[hs.hs6 != "271020"]
    proxies = cfg.settings["trade"].get("proxy_pattern_for", {})
    out = {}
    for bf in cfg.dis["biofuels"]:
        rows = hs[hs.biofuel == bf]
        codes = list(rows.hs6)
        weights = dict(zip(rows.hs6, rows.bio_value_share))
        if not codes and bf in proxies:
            codes, weights = [proxies[bf]], {proxies[bf]: 1.0}
            log.info("trade %s: no HS6 code, using HS %s destination pattern as proxy", bf, codes[0])
        sh = destination_shares(ct_icio, codes, weights, regions)
        out[bf] = build_trade_matrix(bal, bf, sh, regions, cfg.settings.get("ras"))
    return out


def reconcile_balances(bal: pd.DataFrame, trade: dict[str, pd.DataFrame], regions: list[str]) -> pd.DataFrame:
    """Recompute exports/imports/consumption from the final trade matrices."""
    rows = []
    for bf, T in trade.items():
        b = bal[bal.biofuel == bf].set_index("icio").reindex(regions).fillna(0.0)
        exp_, imp_ = T.sum(axis=1), T.sum(axis=0)
        cons = (b.production - exp_ + imp_)
        if (cons < -1e-6).any():
            log.warning("%s: negative consumption after trade in %s; clipped", bf,
                        list(cons[cons < -1e-6].index))
        for r in regions:
            rows.append({"icio": r, "biofuel": bf, "production": b.production[r],
                         "exports": exp_[r], "imports": imp_[r], "consumption": max(cons[r], 0.0),
                         "road": b.road[r], "aviation": b.aviation[r]})
    return pd.DataFrame(rows)
