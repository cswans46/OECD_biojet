"""Robustness evaluation and comparison of the projection methods.

Backtests (pseudo out-of-sample, the OECD's own validation design):
fit on 1995..origin, project origin+1..2022, compare with the published
tables.

    arima         univariate ARIMA, nothing from the future
    oecd_expost   bridge regressions fed with the realised proxies
                  (GDP, deflators, imports): the OECD nowcast setting,
                  an upper bound on accuracy for 2023-2024
    oecd_ar       bridge regressions fed with ARIMA-projected proxies:
                  what the OECD method can do without external forecasts
    naive         random walk with 5-year average drift (benchmark)
    external      only if a historical vintage of the external projections
                  (e.g. IEO2016/AEO2015) is supplied

Metrics by horizon: WAPE = sum|xhat - x| / sum x on gross output and VA in
current prices and on volumes; share of industries with APE > 20%.
Table backtests (optional) add the coefficient-weighted WAPE of A and the
mean absolute error of Leontief output multipliers.

Comparison of 2030 projections: per industry, the three methods' outputs,
their spread (max/min - 1) and coefficient of variation, CAGRs 2022-2030,
summaries by country and sector, and focus industries.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import external as ex
from .methods import Projected, project_arima, project_oecd, _hist, _idx
from .panel import volumes_and_prices, wide
from .project import run_chain
from .table import Table, country_of, sector_of

log = logging.getLogger(__name__)


def _keybase(panel: pd.DataFrame, year: int) -> SimpleNamespace:
    x = wide(panel, "x")
    fd = wide(panel, "fd")
    return SimpleNamespace(keys=list(x.index), Y=pd.DataFrame(columns=list(fd.index)))


def _wape(pred: pd.Series, act: pd.Series) -> float:
    a, p = act.align(pred, join="inner")
    den = a.abs().sum()
    return float((p - a).abs().sum() / den) if den > 0 else np.nan


def naive_projection(panel, base, years, origin) -> Projected:
    out = {}
    for var in ("x", "va", "fd", "imp"):
        vol, _, _ = _hist(panel, var, origin, origin)
        lg = np.log(vol.where(vol > 0))
        k = min(5, vol.shape[1] - 1)
        drift = ((lg[origin] - lg[origin - k]) / k).fillna(0.0).clip(-0.3, 0.3)
        out[var] = pd.DataFrame({y: np.exp(drift * (y - origin)) for y in years})
    xv, xp, has = _hist(panel, "x", origin, origin)
    if has:
        lp = np.log(xp.where(xp > 0))
        k = min(5, xp.shape[1] - 1)
        dp = ((lp[origin] - lp[origin - k]) / k).fillna(0.0)
        px = pd.DataFrame({y: np.exp(dp * (y - origin)) for y in years})
    else:
        px = pd.DataFrame(1.0, index=xv.index, columns=years)
    pfd = pd.DataFrame(1.0, index=base.Y.columns, columns=years)
    return Projected("naive", list(years), out["x"].reindex(base.keys).fillna(1.0), out["va"].reindex(base.keys).fillna(1.0),
                     out["fd"].reindex(base.Y.columns).fillna(1.0), out["imp"], px.reindex(base.keys).fillna(1.0), pfd)


def backtest_margins(panel: pd.DataFrame, origins=(2010, 2014, 2017), end: int = 2022,
                     methods=("arima", "oecd_expost", "oecd_ar", "naive"),
                     external_vintage: ex.External | None = None) -> pd.DataFrame:
    rows = []
    x_cp = wide(panel, "x")
    va_cp = wide(panel, "va")
    xvol_all, _, has_pyp = volumes_and_prices(panel, "x", base_year=end)
    for origin in origins:
        hist = panel[panel["year"] <= origin]
        base = _keybase(hist, origin)
        years = list(range(origin + 1, end + 1))
        for m in methods:
            try:
                if m == "arima":
                    pr = project_arima(hist, base, years, origin, origin)
                elif m == "oecd_expost":
                    pr = project_oecd(hist, base, years, origin, None, origin, future_actual=panel)
                elif m == "oecd_ar":
                    pr = _oecd_with_arima_proxies(hist, base, years, origin)
                elif m == "naive":
                    pr = naive_projection(hist, base, years, origin)
                elif m == "external" and external_vintage is not None:
                    from .methods import project_external
                    pr = project_external(_table_like(base), years, external_vintage, origin)
                else:
                    continue
            except Exception as exc:  # noqa: BLE001
                log.warning("backtest %s %s failed: %s", m, origin, exc)
                continue
            for y in years:
                xhat = x_cp[origin] * pr.gx[y] * pr.px[y]
                vahat = va_cp[origin] * pr.gva[y] * pr.px[y]
                act = x_cp[y]
                ape = ((xhat - act).abs() / act.where(act > 1)).dropna()
                rec = {"origin": origin, "method": m, "year": y, "horizon": y - origin,
                       "wape_x_cp": _wape(xhat, act), "wape_va_cp": _wape(vahat, va_cp[y]),
                       "share_ape_gt_20pct": float((ape > 0.2).mean()), "median_ape_x": float(ape.median())}
                if has_pyp:
                    vhat = xvol_all[origin] * pr.gx[y]
                    rec["wape_x_volume"] = _wape(vhat, xvol_all[y])
                rows.append(rec)
    return pd.DataFrame(rows)


def _table_like(base) -> Table:
    keys = base.keys
    return Table(pd.DataFrame(0.0, keys, keys), pd.DataFrame(0.0, keys, list(base.Y.columns)),
                 pd.DataFrame(0.0, ["TLS", "VA"], keys + list(base.Y.columns)))


def _oecd_with_arima_proxies(hist, base, years, origin) -> Projected:
    """OECD bridge with its proxies (GDP, deflator, imports) projected by ARIMA."""
    from .tsmodels import project_series
    gdpv, gdpp, has = _hist(hist, "gdp", origin, origin)
    impv, _, _ = _hist(hist, "imp", origin, origin)
    h = len(years)
    tidy = []
    for c in gdpv.index:
        gv, _ = project_series(gdpv.loc[c].to_numpy(), h)
        gp, _ = project_series(gdpp.loc[c].to_numpy(), h) if has else (np.ones(h), None)
        iv, _ = project_series(impv.loc[c].to_numpy(), h) if c in impv.index else (np.ones(h), None)
        lv, lp, li = gdpv.loc[c, origin], gdpp.loc[c, origin], impv.loc[c, origin] if c in impv.index else 1.0
        prev_v, prev_i = lv, li
        for k, y in enumerate(years):
            tidy.append({"region": c, "item": "GDP_R_GROWTH", "year": y, "value": 100 * (gv[k] / prev_v - 1)})
            tidy.append({"region": c, "item": "IMP_R_GROWTH", "year": y, "value": 100 * (iv[k] / prev_i - 1)})
            tidy.append({"region": c, "item": "GDP_USD", "year": y, "value": gv[k] * gp[k]})
            prev_v, prev_i = gv[k], iv[k]
        tidy.append({"region": c, "item": "GDP_USD", "year": origin, "value": lv * lp})
    ext = ex.External(pd.DataFrame(tidy), base_year=origin)
    pr = project_oecd(hist, base, years, origin, ext, origin)
    pr.method = "oecd_ar"
    return pr


def backtest_table(origin_table: Table, actual_end: Table, proj: Projected) -> dict:
    """Run the chain from the actual origin table and score the end-year table."""
    res = run_chain(origin_table, proj, keep_tables=True)
    end = proj.years[-1]
    pred = res.tables[(end, "cp")]
    keys = actual_end.keys
    Za, Zp = actual_end.Z.loc[keys, keys].to_numpy(), pred.Z.loc[keys, keys].to_numpy()
    xa, xp = actual_end.x().to_numpy(), pred.x().to_numpy()
    Aa = np.divide(Za, xa[None, :], out=np.zeros_like(Za), where=xa[None, :] > 0)
    Ap = np.divide(Zp, xp[None, :], out=np.zeros_like(Zp), where=xp[None, :] > 0)
    w = Za / max(Za.sum(), 1e-9)
    La = np.linalg.inv(np.eye(len(keys)) - Aa).sum(axis=0)
    Lp = np.linalg.inv(np.eye(len(keys)) - Ap).sum(axis=0)
    return {
        "method": proj.method, "origin": proj.years[0] - 1, "end": end,
        "wape_x_cp": float(np.abs(xp - xa).sum() / xa.sum()),
        "coef_wape_weighted": float((w * np.abs(Ap - Aa) / np.where(Aa > 0, Aa, np.inf)).sum()),
        "cell_wape_Z": float(np.abs(Zp - Za).sum() / Za.sum()),
        "output_multiplier_mae": float(np.abs(Lp - La).mean()),
        "output_multiplier_mape": float(np.mean(np.abs(Lp - La) / La)),
    }


def compare(chains: dict, base: Table, focus_keys: list[str] | None = None, year: int = 2030,
            base_year: int = 2022) -> dict[str, pd.DataFrame]:
    x0 = base.x()
    cols = {}
    for m, ch in chains.items():
        mg = ch.margins
        s = mg[(mg.year == year) & (mg.prices == "cp") & (mg["var"] == "x")].set_index("key")["value"]
        cols[m] = s.reindex(x0.index)
    X = pd.DataFrame(cols)
    by_key = X.copy()
    by_key.insert(0, f"x_{base_year}", x0)
    n = year - base_year
    for m in cols:
        by_key[f"cagr_{m}"] = (X[m] / x0.where(x0 > 0)) ** (1 / n) - 1
    by_key["spread_max_over_min"] = X.max(axis=1) / X.min(axis=1).where(X.min(axis=1) > 0) - 1
    by_key["cv_across_methods"] = X.std(axis=1) / X.mean(axis=1).where(X.mean(axis=1) > 0)
    by_key["country"] = [country_of(k) for k in by_key.index]
    by_key["sector"] = [sector_of(k) for k in by_key.index]

    def agg(level):
        g = by_key.groupby(level)[[f"x_{base_year}"] + list(cols)].sum()
        for m in cols:
            g[f"cagr_{m}"] = (g[m] / g[f"x_{base_year}"]) ** (1 / n) - 1
        g["spread_max_over_min"] = g[list(cols)].max(axis=1) / g[list(cols)].min(axis=1) - 1
        return g

    paths = []
    for m, ch in chains.items():
        mg = ch.margins
        tot = mg[(mg.prices == "cp") & (mg["var"] == "x")].groupby("year")["value"].sum()
        paths.append(tot.rename(m))
    world = pd.concat(paths, axis=1)
    world.loc[base_year] = x0.sum()
    world = world.sort_index()
    out = {"by_industry": by_key, "by_country": agg("country"), "by_sector": agg("sector"),
           "world_output_path": world}
    if focus_keys:
        fk = [k for k in focus_keys if k in by_key.index]
        rows = []
        for m, ch in chains.items():
            mg = ch.margins
            sel = mg[(mg.prices == "cp") & (mg["var"] == "x") & mg.key.isin(fk)]
            for _, r in sel.iterrows():
                rows.append({"method": m, "key": r.key, "year": r.year, "x_cp": r.value})
        for k in fk:
            for m in chains:
                rows.append({"method": m, "key": k, "year": base_year, "x_cp": float(x0[k])})
        out["focus_paths"] = pd.DataFrame(rows).sort_values(["key", "method", "year"])
        out["focus_2030"] = by_key.loc[fk]
    return out


def write(results: dict[str, pd.DataFrame], out_dir: Path, prefix: str = "") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, df in results.items():
        df.to_csv(out_dir / f"{prefix}{k}.csv")


def plot_focus(focus_paths: pd.DataFrame, path: Path, title: str) -> Path:  # pragma: no cover - plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(dict.fromkeys(focus_paths["key"]))
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        for m, g in focus_paths[focus_paths.key == k].groupby("method"):
            ax.plot(g.year, g.x_cp, marker="o", ms=3, label=m)
        ax.set_title(k, fontsize=9)
        ax.set_ylabel("USD million, current prices")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
