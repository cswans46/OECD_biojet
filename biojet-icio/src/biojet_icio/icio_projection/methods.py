"""Margin projections 2023-2030 by three methods.

Each method returns ``Projected``: growth indices relative to the base year
(base = 1) for the margins of the yearly balancing problem.

    gx   output volume, per industry key
    gva  value-added volume, per industry key
    gfd  final-demand volume, per FD column
    gimp import volume, per country
    px   output price index, per industry key (also used for VA and TLS)
    pfd  final-demand price index, per FD column
    coef optional {year: DataFrame(rows=input key, cols=using key)} of input
         coefficient factors (energy intensity change, external method)

Methods
-------
``oecd``   OECD (2025, Trends in GVCs, Annex A): bridge regressions of each
           ICIO margin on its national-accounts proxy, with ARIMA errors,
           estimated on 1995-<last year>. Proxies: country real GDP and
           broad-industry value added (UN AMA when supplied, otherwise the
           ICIO's own aggregates, which are benchmarked to AMA). Future
           proxies: IMF WEO GDP and trade volumes. Gross output is bridged
           with a constraint on shifts in the VA/GO ratio. Final demand is
           extended with growth rates of its aggregate (AMA / WEO GDP).
           Output prices are bridged on the WEO USD GDP deflator.
``arima``  Univariate ARIMA(p,1,q) on the log of each margin's own 28-year
           history (volumes and prices separately), lags chosen by AICc.
``external`` Sector output growth from EIA AEO (US industries) and IEO
           (16 regions x sectors, energy production for energy sectors),
           scaled from region to country by relative WEO GDP; energy input
           coefficients follow IEO/AEO energy intensity by end use and fuel;
           energy prices from AEO/IEO, other prices from the WEO deflator.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import external as ex
from .panel import volumes_and_prices, wide
from .table import Table, country_of, sector_of
from .tsmodels import project_series

log = logging.getLogger(__name__)

AMA_GROUPS = [(r"^A", "A"), (r"^B", "B"), (r"^C", "C"), (r"^D|^E", "DE"), (r"^F", "F"),
              (r"^G|^I", "GI"), (r"^H|^J", "HJ"), (r".", "KU")]


@dataclass
class Projected:
    method: str
    years: list[int]
    gx: pd.DataFrame
    gva: pd.DataFrame
    gfd: pd.DataFrame
    gimp: pd.DataFrame
    px: pd.DataFrame
    pfd: pd.DataFrame
    coef: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _hist(panel: pd.DataFrame, var: str, last_year: int, base_year: int):
    vol, price, has_pyp = volumes_and_prices(panel, var, base_year=base_year)
    keep = [y for y in vol.columns if y <= last_year]
    return vol[keep], price[keep], has_pyp


def _idx(df_future: pd.DataFrame, hist_last: pd.Series) -> pd.DataFrame:
    return df_future.div(hist_last.where(hist_last.abs() > 1e-12), axis=0).fillna(1.0)


def _project_matrix(hist: pd.DataFrame, years: list[int], z_hist=None, z_future=None) -> pd.DataFrame:
    """Project every row of a key x year matrix; returns key x years levels."""
    h = len(years)
    out = np.empty((len(hist), h))
    zh = zf = None
    for r, (key, row) in enumerate(hist.iterrows()):
        y = row.to_numpy(float)
        if z_hist is not None:
            zh = z_hist(key)
            zf = z_future(key)
            if zh is None or zf is None or np.any(~np.isfinite(zh)) or np.any(zh <= 0):
                zh = zf = None
        if np.any(~np.isfinite(y)):
            y = pd.Series(y).ffill().bfill().fillna(0.0).to_numpy()
        out[r], _ = project_series(y, h, zh, zf)
    return pd.DataFrame(out, index=hist.index, columns=years)


def _group(code: str) -> str:
    return ex.first_match(code, AMA_GROUPS) or "KU"


def _constrain_ratio(gx: pd.DataFrame, gva: pd.DataFrame, x_hist: pd.DataFrame, va_hist: pd.DataFrame) -> pd.DataFrame:
    """Keep VA/GO within its historical range and limit year-on-year shifts
    to the largest historical shift (OECD: no implausible VA/GO changes)."""
    ratio_h = (va_hist / x_hist.where(x_hist.abs() > 1e-9)).replace([np.inf, -np.inf], np.nan)
    lo = ratio_h.min(axis=1).fillna(0.0)
    hi = ratio_h.max(axis=1).fillna(1.0)
    dmax = ratio_h.pct_change(axis=1).abs().max(axis=1).clip(upper=0.10).fillna(0.10)
    r_prev = ratio_h.iloc[:, -1].fillna(0.0)
    base_ratio = r_prev.copy()
    out = gva.copy()
    for y in gx.columns:
        r = base_ratio * gva[y] / gx[y].where(gx[y].abs() > 1e-12)
        r = r.fillna(r_prev)
        r = r.clip(lower=r_prev * (1 - dmax), upper=r_prev * (1 + dmax)).clip(lower=lo, upper=hi)
        out[y] = (r / base_ratio.where(base_ratio.abs() > 1e-12)).fillna(1.0) * gx[y]
        r_prev = r
    return out


def _fd_country_deflator(base: Table, pc: dict[str, pd.Series], years) -> pd.DataFrame:
    rows = {}
    for c in base.Y.columns:
        rows[c] = pc.get(country_of(c), pd.Series(1.0, index=years)).reindex(years).fillna(1.0)
    return pd.DataFrame(rows).T


def _invnt_like(col: str) -> bool:
    return col.endswith("INVNT") or col.endswith("INVNT_") or "INVNT" in col


def _fd_from_country(base: Table, gc: dict[str, pd.Series], years) -> pd.DataFrame:
    rows = {}
    for c in base.Y.columns:
        rows[c] = gc.get(country_of(c), pd.Series(1.0, index=years)).reindex(years).fillna(1.0)
    return pd.DataFrame(rows).T


def _map_to_base(g: pd.DataFrame, base_keys: list[str], parent_of: dict[str, str] | None) -> pd.DataFrame:
    """Expand panel-key growth to base-table keys (split children use parent)."""
    parent_of = parent_of or {}
    src = [parent_of.get(k, k) for k in base_keys]
    missing = sorted({s for s in src if s not in g.index})
    if missing:
        log.warning("%s base keys have no history (growth = 1): %s", len(missing), missing[:8])
    out = g.reindex(src).fillna(1.0)
    out.index = base_keys
    return out


# ----------------------------------------------------------------------------
# ARIMA
# ----------------------------------------------------------------------------

def project_arima(panel: pd.DataFrame, base: Table, years: list[int], last_year: int,
                  base_year: int = 2022, parent_of=None) -> Projected:
    xv, xp, has_pyp = _hist(panel, "x", last_year, base_year)
    vav, _, _ = _hist(panel, "va", last_year, base_year)
    fdv, _, _ = _hist(panel, "fd", last_year, base_year)
    impv, _, _ = _hist(panel, "imp", last_year, base_year)
    gdpv, gdpp, _ = _hist(panel, "gdp", last_year, base_year)
    anchor = last_year
    # horizon covers last_year+1 .. max(years)
    hy = list(range(last_year + 1, max(years) + 1))
    gx = _idx(_project_matrix(xv, hy), xv[anchor])
    gva = _idx(_project_matrix(vav, hy), vav[anchor])
    gva = _constrain_ratio(gx, gva, xv, vav)
    gfd = _idx(_project_matrix(fdv.loc[[c for c in fdv.index if not _invnt_like(c)]], hy), fdv[anchor])
    gimp = _idx(_project_matrix(impv, hy), impv[anchor])
    px = _idx(_project_matrix(xp, hy), xp[anchor]) if has_pyp else pd.DataFrame(1.0, index=xv.index, columns=hy)
    pc = _idx(_project_matrix(gdpp, hy), gdpp[anchor]) if has_pyp else pd.DataFrame(1.0, index=gdpv.index, columns=hy)
    gfd = _fill_invnt(base, gfd, hy)
    proj = _finish("arima", base, years, last_year, base_year, gx, gva, gfd, gimp, px,
                   {c: pc.loc[c] for c in pc.index}, parent_of)
    if not has_pyp:
        proj.notes.append("No PYP tables: projected nominal values, prices flat.")
    return proj


def _fill_invnt(base: Table, gfd: pd.DataFrame, hy) -> pd.DataFrame:
    """Inventory changes follow the country's GFCF (they change sign)."""
    rows = {}
    for c in base.Y.columns:
        if c in gfd.index:
            rows[c] = gfd.loc[c]
        else:
            g = f"{country_of(c)}_GFCF"
            rows[c] = gfd.loc[g] if g in gfd.index else pd.Series(1.0, index=hy)
    return pd.DataFrame(rows).T


def _finish(method, base, years, last_year, base_year, gx, gva, gfd, gimp, px, pc, parent_of, coef=None) -> Projected:
    """Re-anchor indices from last_year to base_year and to the base keys."""
    if last_year != base_year:
        raise ValueError("Projection anchor must be the base year (backtests pass their own base).")
    keys = base.keys
    return Projected(
        method=method, years=years,
        gx=_map_to_base(gx[years], keys, parent_of), gva=_map_to_base(gva[years], keys, parent_of),
        gfd=gfd.reindex(base.Y.columns).fillna(1.0)[years],
        gimp=gimp[years],
        px=_map_to_base(px[years], keys, parent_of),
        pfd=_fd_country_deflator(base, pc, years),
        coef=coef or {},
    )


# ----------------------------------------------------------------------------
# OECD bridge regressions with ARIMA errors
# ----------------------------------------------------------------------------

def project_oecd(panel: pd.DataFrame, base: Table, years: list[int], last_year: int,
                 ext: ex.External | None, base_year: int = 2022, parent_of=None,
                 future_actual: pd.DataFrame | None = None) -> Projected:
    """future_actual: optional panel with the realised proxies (backtests)."""
    xv, xp, has_pyp = _hist(panel, "x", last_year, base_year)
    vav, _, _ = _hist(panel, "va", last_year, base_year)
    fdv, _, _ = _hist(panel, "fd", last_year, base_year)
    impv, _, _ = _hist(panel, "imp", last_year, base_year)
    gdpv, gdpp, _ = _hist(panel, "gdp", last_year, base_year)
    hy = list(range(last_year + 1, max(years) + 1))
    countries = list(gdpv.index)

    # --- future proxies: GDP volume, USD deflator, import volume ---------
    if future_actual is not None:
        fv, fp, _ = volumes_and_prices(future_actual, "gdp", base_year=base_year)
        fi, _, _ = volumes_and_prices(future_actual, "imp", base_year=base_year)
        gdp_f = (fv[hy].div(fv[last_year], axis=0)).mul(gdpv[last_year], axis=0)
        defl_f = (fp[hy].div(fp[last_year], axis=0)).mul(gdpp[last_year], axis=0)
        imp_idx = fi[hy].div(fi[last_year], axis=0)
    else:
        if ext is None:
            raise ValueError("OECD method needs external proxies (IMF WEO) for future years.")
        gi = ex.world_fallback(ext, countries, [base_year] + hy, ext.gdp_real)
        di = ex.world_fallback(ext, countries, [base_year] + hy, ext.usd_deflator)
        ii = ex.world_fallback(ext, countries, [base_year] + hy, lambda c, y: ext.trade_volume(c, "IMP", y))
        gdp_f = pd.DataFrame({c: gi[c][hy] * gdpv.loc[c, last_year] for c in countries}).T
        defl_f = pd.DataFrame({c: di[c][hy] * gdpp.loc[c, last_year] for c in countries}).T
        imp_idx = pd.DataFrame({c: ii[c][hy] for c in countries}).T

    # --- broad-industry VA proxies (AMA groups) ---------------------------
    grp_of = {k: (country_of(k), _group(sector_of(k))) for k in vav.index}
    gkeys = pd.Series({k: f"{c}|{g}" for k, (c, g) in grp_of.items()})
    grp_hist = vav.groupby(gkeys).sum()
    grp_country = pd.Series({gk: gk.split("|")[0] for gk in grp_hist.index})

    def gdp_h(gk):
        c = grp_country[gk]
        return gdpv.loc[c].to_numpy() if c in gdpv.index else None

    def gdp_fut(gk):
        c = grp_country[gk]
        return gdp_f.loc[c].to_numpy() if c in gdp_f.index else None

    grp_future = _project_matrix(grp_hist, hy, gdp_h, gdp_fut)

    def g_h(k):
        return grp_hist.loc[gkeys[k]].to_numpy() if k in gkeys.index else None

    def g_f(k):
        return grp_future.loc[gkeys[k]].to_numpy() if k in gkeys.index else None

    gva = _idx(_project_matrix(vav, hy, g_h, g_f), vav[last_year])
    gx = _idx(_project_matrix(xv.reindex(vav.index), hy, g_h, g_f), xv[last_year])
    gva = _constrain_ratio(gx, gva, xv, vav)

    # --- prices bridged on the country USD deflator ----------------------
    if has_pyp:
        def p_h(k):
            c = country_of(k)
            return gdpp.loc[c].to_numpy() if c in gdpp.index else None

        def p_f(k):
            c = country_of(k)
            return defl_f.loc[c].to_numpy() if c in defl_f.index else None

        px = _idx(_project_matrix(xp, hy, p_h, p_f), xp[last_year])
        pc = {c: defl_f.loc[c] / gdpp.loc[c, last_year] for c in countries}
    else:
        px = pd.DataFrame(1.0, index=xv.index, columns=hy)
        pc = {c: pd.Series(1.0, index=hy) for c in countries}

    # --- final demand: growth of its aggregate (AMA / WEO GDP) -----------
    gdp_idx = {c: gdp_f.loc[c] / gdpv.loc[c, last_year] for c in countries}
    gfd = _fd_from_country(base, gdp_idx, hy)
    if ext is not None and future_actual is None:
        for col in gfd.index:
            s = ext.index(country_of(col), f"AMA:{sector_of(col)}", [base_year] + hy)
            if s is not None:
                gfd.loc[col] = s[hy].to_numpy()
    gimp = imp_idx
    proj = _finish("oecd", base, years, last_year, base_year, gx, gva, gfd, gimp, px, pc, parent_of)
    if not has_pyp:
        proj.notes.append("No PYP tables: projected nominal values, prices flat.")
    return proj


# ----------------------------------------------------------------------------
# External projections (AEO, IEO, WEO)
# ----------------------------------------------------------------------------

def project_external(base: Table, years: list[int], ext: ex.External, base_year: int = 2022,
                     parent_of=None) -> Projected:
    keys_panel = list(dict.fromkeys((parent_of or {}).get(k, k) for k in base.keys))
    yrs = [base_year] + list(years)
    countries = sorted({country_of(k) for k in keys_panel})
    gdp = ex.world_fallback(ext, countries, yrs, ext.gdp_real)
    defl = ex.world_fallback(ext, countries, yrs, ext.usd_deflator)
    imp = ex.world_fallback(ext, countries, yrs, lambda c, y: ext.trade_volume(c, "IMP", y))
    notes, used = [], {"AEO": 0, "IEO": 0, "IEO energy": 0, "WEO": 0}

    gx_rows, px_rows = {}, {}
    for k in keys_panel:
        c, s = country_of(k), sector_of(k)
        region = ex.IEO_REGION.get(c)
        series = None
        if c == "USA":
            ind = ex.first_match(s, ex.AEO_INDUSTRY)
            if ind:
                series = ext.index("United States", f"SHIP:{ind}", yrs)
                if series is not None:
                    used["AEO"] += 1
        if series is None and region:
            item = ex.first_match(s, ex.ENERGY_OUTPUT)
            if item:
                series = ext.index(region, item, yrs)
                if series is not None:
                    used["IEO energy"] += 1
            if series is None:
                series = ext.index(region, f"GO:{ex.first_match(s, ex.IEO_SECTOR)}", yrs)
                if series is not None:
                    rg = ext.index(region, "GDP_R", yrs)
                    if rg is not None:
                        series = series * gdp[c] / rg
                    used["IEO"] += 1
        if series is None:
            series = gdp[c]
            used["WEO"] += 1
        gx_rows[k] = series[years]
        # prices: energy from AEO/IEO price paths, else USD GDP deflator
        p = defl[c]
        pitem = ex.first_match(s, ex.ENERGY_PRICE)
        if pitem:
            for reg in ([region] if region else []) + ["United States", "World"]:
                ps = ext.index(reg, pitem, yrs)
                if ps is not None:
                    p = ps
                    break
        px_rows[k] = p[years]
    gx = pd.DataFrame(gx_rows).T
    px = pd.DataFrame(px_rows).T
    gva = gx.copy()  # VA/GO fixed at base year (external sources give no VA)
    gfd = _fd_from_country(base, {c: gdp[c][years] for c in countries}, years)
    gimp = pd.DataFrame({c: imp[c][years] for c in countries}).T
    notes.append(f"Output drivers used: {used}")

    coef = energy_coefficient_factors(base, ext, years, base_year, parent_of)
    proj = Projected("external", years,
                     _map_to_base(gx, base.keys, parent_of), _map_to_base(gva, base.keys, parent_of),
                     gfd.reindex(base.Y.columns).fillna(1.0), gimp, _map_to_base(px, base.keys, parent_of),
                     _fd_country_deflator(base, {c: defl[c][years] for c in countries}, years), coef, notes)
    return proj


def energy_coefficient_factors(base: Table, ext: ex.External, years, base_year: int = 2022,
                               parent_of=None) -> dict:
    """{year: DataFrame(fuel x using-column)} energy-intensity factors.

    factor = (energy use of the end-use sector / its activity) relative to
    the base year, by IEO region (AEO for the US). Activity: industrial
    gross output for 'industrial', real GDP otherwise.
    """
    yrs = [base_year] + list(years)
    cols = list(base.Z.columns) + list(base.Y.columns)
    out = {y: pd.DataFrame(1.0, index=[f for _, f in ex.ENERGY_INPUT], columns=cols) for y in years}
    cache = {}
    for col in cols:
        c = country_of(col)
        region = "United States" if c == "USA" else ex.IEO_REGION.get(c)
        if region is None:
            continue
        s = sector_of(col)
        if col in base.Y.columns:
            if not s.startswith("HFCE"):
                continue
            enduse = "buildings"
        else:
            enduse = ex.first_match(sector_of((parent_of or {}).get(col, col)), ex.END_USE)
        for fuel in [f for _, f in ex.ENERGY_INPUT]:
            key = (region, enduse, fuel)
            if key not in cache:
                e = ext.index(region, f"ENERGY:{enduse}:{fuel}", yrs)
                act = ext.index(region, "GO:industrial_total" if enduse == "industrial" else "GDP_R", yrs)
                if act is None:
                    act = ext.index(region, "GDP_R", yrs)
                cache[key] = None if e is None or act is None else (e / act)
            f = cache[key]
            if f is not None:
                for y in years:
                    out[y].loc[fuel, col] = float(f[y])
    return out


def project(method: str, panel, base: Table, years, ext=None, base_year: int = 2022, parent_of=None, **kw) -> Projected:
    if method == "arima":
        return project_arima(panel, base, years, base_year, base_year, parent_of)
    if method == "oecd":
        return project_oecd(panel, base, years, base_year, ext, base_year, parent_of, **kw)
    if method == "external":
        if ext is None:
            raise ValueError("external method needs the external projections file")
        return project_external(base, years, ext, base_year, parent_of)
    raise ValueError(method)
