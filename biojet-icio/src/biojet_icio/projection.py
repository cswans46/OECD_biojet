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

Chained OECD / ARIMA / external projections (``run_projection``)
-----------------------------------------------------------------
``run_projection`` adds three year-by-year methods from the shared package
``biojet_icio.icio_projection`` (identical to cswans46/Datacenters): each
year 2023-2030 is projected from the previous year's balanced table and
re-balanced with RAS in previous year's prices, then in current prices.

* ``oecd``      bridge regressions with ARIMA errors on national-accounts
                proxies (OECD 2025, Trends in GVCs, Annex A), IMF WEO future;
* ``arima``     univariate ARIMA(p,1,q) on each margin's 1995-2022 history;
* ``external``  EIA AEO (US) / IEO (regions) sector output, energy intensity
                and energy prices, IMF WEO elsewhere.

Biofuel sectors follow ``data/projections/output_index.csv`` (built by
:mod:`projection_inputs`) when it exists, else their parent plus a biojet
share path (``[projection] biojet_theta_2030``). Countries in
``projections.toml [general] exclude_countries`` (Canada) are held at 2022,
or follow ``output_index_CAN.csv`` when supplied, in every method.
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


# ---------------------------------------------------------------------------
# OECD / ARIMA / external projections 2023-2030 (shared icio_projection)
# ---------------------------------------------------------------------------

YEARS = list(range(2023, 2031))


def parent_map(icio: ICIO, results: dict | None = None, crop_codes=(), biofuel_codes=("BGS", "BDS", "BJT", "OLB")) -> dict[str, str]:
    """Split key -> parent key, from the biofuel report or label rules."""
    out = {}
    rep = (results or {}).get("biofuel_report")
    if rep is not None and len(rep):
        for _, r in rep.iterrows():
            out[make_label(r.region, r.biofuel)] = make_label(r.region, r.parent)
    for lab in icio.Z.index:
        code = lab[4:]
        if lab not in out and code in biofuel_codes:
            out[lab] = make_label(lab[:3], "C19")
        if code in crop_codes:
            out[lab] = make_label(lab[:3], "A01")
    return out


def biojet_driver(base: ICIO, proj, theta_2030: float | None, price_ratio: float,
                  regions: list[str] | None = None, base_year: int = 2022):
    """Drive BJT output (volume) by a rising biojet share of H51 jet fuel.

    theta_t = theta_2022 + (theta_2030 - theta_2022) * (t - 2022) / 8, with
    theta = BJT purchases / (BJT + C19 purchases) of H51 at 2022 prices.
    Extra BJT (value) = (theta_t - theta_2022) * H51 jet spending_2022 *
    H51 growth * price_ratio, produced in the region; the fossil jet fuel it
    replaces (same energy, at fossil prices) is removed from the region's C19.
    """
    if theta_2030 is None:
        return proj
    x0 = base.output_rows()
    countries = regions or base.countries
    c19 = [make_label(c, "C19") for c in base.countries]
    bjt = [make_label(c, "BJT") for c in base.countries]
    for r in countries:
        h, kb, kc = make_label(r, "H51"), make_label(r, "BJT"), make_label(r, "C19")
        if h not in base.Z.columns or kb not in proj.gx.index or kc not in proj.gx.index:
            continue
        fossil = float(base.Z.loc[[k for k in c19 if k in base.Z.index], h].sum())
        bio = float(base.Z.loc[[k for k in bjt if k in base.Z.index], h].sum())
        spend = fossil + bio
        if spend <= 0:
            continue
        th0 = bio / spend
        for y in proj.years:
            th = th0 + (theta_2030 - th0) * (y - base_year) / (YEARS[-1] - base_year)
            d_fossil = (th - th0) * spend * float(proj.gx.loc[h, y])
            if x0[kb] > 0:
                new_b = x0[kb] * proj.gx.loc[kb, y] + d_fossil * price_ratio
                proj.gva.loc[kb, y] *= new_b / (x0[kb] * proj.gx.loc[kb, y])
                proj.gx.loc[kb, y] = new_b / x0[kb]
            if x0[kc] > 0:
                new_c = max(x0[kc] * proj.gx.loc[kc, y] - d_fossil, 0.05 * x0[kc])
                proj.gva.loc[kc, y] *= new_c / (x0[kc] * proj.gx.loc[kc, y])
                proj.gx.loc[kc, y] = new_c / x0[kc]
    proj.notes.append(f"BJT driven to theta_2030={theta_2030} (price ratio {price_ratio})")
    return proj


def inputs_driver(base: ICIO, proj, growth: dict[str, pd.DataFrame], sectors) -> object:
    """Biofuel sectors follow data/projections/output_index.csv (projection_inputs).

    Only rows with a positive 2022 base are indexed: a sector created from a
    zero base (``output_level_musd``) cannot be chained from a zero seed and
    stays at zero here; use ``project_year`` for those."""
    g = growth["output_index"]
    g = g[g.sector.isin(list(sectors)) & g.year.isin(proj.years)]
    x0 = base.output_rows()
    used, skipped = 0, []
    for r in g.itertuples(index=False):
        lab = make_label(r.country, r.sector)
        if lab not in proj.gx.index:
            continue
        if x0[lab] <= 0:
            skipped.append(lab)
            continue
        if pd.notna(r.output_index):
            ratio = float(r.output_index) / float(proj.gx.loc[lab, r.year])
            proj.gx.loc[lab, r.year] = float(r.output_index)
            proj.gva.loc[lab, r.year] *= ratio
            used += 1
    if skipped:
        log.warning("Zero-base sectors not chained (use project_year to seed them): %s", sorted(set(skipped))[:10])
    proj.notes.append(f"Biofuel sectors {list(sectors)} follow output_index.csv ({used} cells)")
    return proj


def hold_excluded(base: ICIO, proj, countries, growth: dict | None = None) -> object:
    """Hold excluded countries (Canada: proprietary E3MC) at 2022, or follow
    output_index_<C>.csv when supplied, in every method."""
    supplied = (growth or {}).get("output_index")
    for c in countries:
        rows = [k for k in proj.gx.index if k[:3] == c]
        sub = None if supplied is None else supplied[supplied.country == c]
        if sub is not None and len(sub):
            for r in sub.itertuples(index=False):
                lab = make_label(c, r.sector)
                if lab in proj.gx.index and r.year in proj.years and pd.notna(r.output_index):
                    proj.gx.loc[lab, r.year] = float(r.output_index)
                    proj.gva.loc[lab, r.year] = float(r.output_index)
            proj.notes.append(f"{c} follows output_index_{c}.csv")
        else:
            proj.gx.loc[rows] = 1.0
            proj.gva.loc[rows] = 1.0
            proj.notes.append(f"{c} held at 2022 (excluded; add output_index_{c}.csv)")
        proj.px.loc[rows] = 1.0
        fd = [k for k in proj.gfd.index if k[:3] == c]
        proj.gfd.loc[fd] = 1.0
        proj.pfd.loc[[k for k in proj.pfd.index if k[:3] == c]] = 1.0
        if c in proj.gimp.index:
            proj.gimp.loc[c] = 1.0
    return proj


def run_projection(cfg, base_icio: ICIO, results: dict | None = None, panel=None, external=None,
                   out_dir: Path | None = None, methods=("oecd", "arima", "external"),
                   backtest_origins=(2010, 2014, 2017), write_tables: bool = True) -> dict:
    """Project ``base_icio`` (the split 2022 table) to 2030 with each method,
    compare the methods and backtest them. Returns the comparison tables."""
    from .icio_projection import evaluate as E, external as X, methods as M, panel as P, project as PR
    from .icio_projection.table import Table

    pc = cfg.settings.get("projection", {})
    out_dir = Path(out_dir or cfg.path("output_dir") / "projection")
    out_dir.mkdir(parents=True, exist_ok=True)
    if panel is None:
        panel = P.load_panel(cfg.root / pc.get("panel", "data/projection/panel"))
    if external is None:
        ext_path = cfg.root / pc.get("external", "data/projection/external.csv")
        external = X.External.load(ext_path) if ext_path.exists() else None
    if external is None:
        log.warning("No external projections file: only the ARIMA method runs.")
        methods = [m for m in methods if m == "arima"]

    crop_codes = cfg.settings.get("crops", {}).get("crops", []) if cfg.settings.get("crops", {}).get("enabled") else []
    parents = parent_map(base_icio, results, crop_codes=crop_codes,
                         biofuel_codes=tuple(cfg.settings["disaggregation"]["biofuels"]))
    base = Table.from_icio(base_icio)
    ras_kw = {k: cfg.settings["ras"][k] for k in ("max_iter", "tol", "blend") if k in cfg.settings.get("ras", {})}
    theta = pc.get("biojet_theta_2030")
    theta = None if theta in ("", None) else float(theta)
    price_ratio = float(pc.get("price_ratio", cfg.settings.get("projection", {}).get("price_ratio", 2.5)))

    # Your public drivers (projection_inputs) and excluded countries (projections.toml)
    pdir = cfg.root / pc.get("inputs_dir", "data/projections")
    driver = pc.get("biofuel_driver", "auto")
    growth = None
    if (pdir / "output_index.csv").exists():
        growth = load_growth(pdir)
    elif driver == "inputs":
        raise FileNotFoundError(f"{pdir}/output_index.csv missing: run scripts/build_projections.py --inputs-only")
    excluded, can_growth = [], None
    ptoml = cfg.root / "config" / "projections.toml"
    if pc.get("hold_excluded", True) and ptoml.exists():
        import tomllib
        with open(ptoml, "rb") as fh:
            excluded = tomllib.load(fh).get("general", {}).get("exclude_countries", [])
        if excluded:
            can = {}
            for c in excluded:
                f = pdir / f"output_index_{c}.csv"
                if f.exists():
                    can.setdefault("output_index", []).append(pd.read_csv(f))
            can_growth = {"output_index": pd.concat(can["output_index"])} if can else None

    chains, notes = {}, {}
    for m in methods:
        pr = M.project(m, panel, base, YEARS, ext=external, parent_of=parents)
        if growth is not None and driver in ("auto", "inputs"):
            pr = inputs_driver(base_icio, pr, growth, cfg.settings["disaggregation"]["biofuels"])
        else:
            pr = biojet_driver(base_icio, pr, theta, price_ratio)
        if excluded:
            pr = hold_excluded(base_icio, pr, excluded, can_growth)
        chains[m] = PR.run_chain(base, pr, out_dir=out_dir if write_tables else None, ras_kw=ras_kw)
        chains[m].ras_log.to_csv(out_dir / f"ras_log_{m}.csv", index=False)
        chains[m].margins.to_csv(out_dir / f"margins_{m}.csv.gz", index=False)
        notes[m] = pr.notes

    focus = [k for r in ("USA", "BRA", "DEU", "CAN", "IDN") for k in (f"{r}_BJT", f"{r}_C19", f"{r}_H51")]
    cmp_ = E.compare(chains, base, focus_keys=focus)
    E.write(cmp_, out_dir / "comparison")
    try:
        if "focus_paths" in cmp_ and len(cmp_["focus_paths"]):
            E.plot_focus(cmp_["focus_paths"][cmp_["focus_paths"].key.str[:3].isin(["USA", "CAN"])],
                         out_dir / "comparison" / "focus_paths.png", "Gross output by projection method")
    except Exception as exc:  # noqa: BLE001
        log.warning("Plot failed: %s", exc)
    if backtest_origins:
        bt = E.backtest_margins(panel, origins=backtest_origins)
        (out_dir / "backtest").mkdir(exist_ok=True)
        bt.to_csv(out_dir / "backtest" / "backtest_margins.csv", index=False)
        bt.groupby(["method", "horizon"])[["wape_x_cp", "wape_va_cp", "share_ape_gt_20pct"]].mean() \
          .to_csv(out_dir / "backtest" / "backtest_by_horizon.csv")
    cmp_["notes"] = pd.DataFrame([{"method": k, "note": "; ".join(v)} for k, v in notes.items()])
    return cmp_
