"""Build 2023-2030 projection drivers from public sources (Canada excluded).

Writes ``data/projections/output_index.csv`` (year, country, sector, output_index,
output_level_musd, source) and ``fd_index.csv`` (year, country, fd, index), the
inputs read by :mod:`biojet_icio.projection`.

Rules (details and sources in docs/projections_2030.md):

* every sector: (cumulative IMF WEO Oct-2026 real GDP index) ** elasticity
* C19 remainder, and B06 outside the Gulf: IEA global oil-demand index
* B06/C19 in Gulf ICIO regions (SAU, ARE): oil index to 2025, then the
  Gulf output path relative to 2025 (the Hormuz shock)
* H51: IATA world RPK index (Gulf carriers: -11.4% in 2026)
* BGS / BDS: OECD-FAO Agricultural Outlook 2026-2035 ethanol / biomass-based
  diesel production (country, else EU aggregate, else world)
* BJT: global SAF path (IATA history and 2026 estimate; 2030 case) times the
  producer shares, as an absolute level where the 2022 base is zero
* Canada: no rows. Supply E3MC-based ``output_index_CAN.csv``
"""
from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from .icio import ICIO, make_label

log = logging.getLogger(__name__)
FUEL_PATTERN_FD = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"]


def load_settings(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def gdp_index(gdp: pd.DataFrame, countries: list[str], years: list[int], fallback: str) -> pd.DataFrame:
    g = gdp.pivot(index="iso3", columns="year", values="real_gdp_growth_pct")
    g = g.apply(pd.to_numeric, errors="coerce")
    world = g.loc[fallback]
    out = {}
    for c in countries:
        row = g.loc[c] if c in g.index else world
        row = row.fillna(world)
        out[c] = np.cumprod([1 + row[y] / 100 for y in years])
    return pd.DataFrame(out, index=years).T          # country x year


def biofuel_index(bf: pd.DataFrame, commodity: str, countries: list[str], years: list[int],
                  eu: list[str]) -> pd.DataFrame:
    d = bf[(bf.commodity == commodity) & (bf.measure == "QP")]
    p = d.pivot(index="area", columns="year", values="million_litres").astype(float)
    out = {}
    for c in countries:
        src = c if c in p.index and p.loc[c, 2022] > 0 else ("EU" if c in eu else "W")
        out[c] = (p.loc[src, years] / p.loc[src, 2022]).values
    return pd.DataFrame(out, index=years).T


def build(base: ICIO, root: Path, settings_file: Path | None = None,
          scenario: str = "realised") -> tuple[pd.DataFrame, pd.DataFrame]:
    """scenario: 'realised' (post-shock outlook) or 'prewar' (no-war counterfactual)."""
    s = load_settings(settings_file or root / "config" / "projections.toml")
    pw = s.get("prewar", {}) if scenario == "prewar" else None
    years = s["general"]["years"]
    excl = set(s["general"]["exclude_countries"])
    countries = [c for c in base.countries if c not in excl]
    sectors = base.sectors_of(base.countries[0])

    gdp = pd.read_csv(root / s["general"]["gdp_file"])
    if pw is not None:
        gdp = gdp.copy()
        g25 = gdp[gdp.year == 2025].set_index("iso3").real_gdp_growth_pct
        for y in (2026, 2027):
            m = gdp.year == y
            gulfm = m & gdp.iso3.isin(pw["gulf_like"])
            gdp.loc[gulfm, "real_gdp_growth_pct"] = gdp.loc[gulfm, "iso3"].map(g25).values
            if y == 2026:
                other = m & ~gdp.iso3.isin(pw["gulf_like"])
                gdp.loc[other, "real_gdp_growth_pct"] = pd.to_numeric(
                    gdp.loc[other, "real_gdp_growth_pct"], errors="coerce") + pw["gdp_addback_2026_pp"]
    G = gdp_index(gdp, countries, years, s["general"]["fallback_gdp"])
    bfd = pd.read_csv(root / s["general"]["biofuel_file"])
    eu = s["biofuels"]["eu_members"]
    ET = biofuel_index(bfd, "ethanol", countries, years, eu)
    BD = biofuel_index(bfd, "biodiesel_incl_RD_SAF", countries, years, eu)

    oil = pw["demand_mbd"] if pw is not None else s["oil"]["demand_mbd"]
    oil_idx = {y: oil[str(y)] / oil["2022"] for y in years}
    gulf = set(s["oil"]["gulf_regions"])
    gulf_path = {int(k): v for k, v in s["oil"]["gulf_b06_vs_2025"].items()}
    rpk = {int(k): v for k, v in s["aviation"]["rpk_growth"].items()}
    if pw is not None:
        rpk[2026] = pw["rpk_growth_2026"]
    rpk_idx, acc = {}, 1.0
    gulf_rpk_idx, gacc = {}, 1.0
    for y in years:
        acc *= 1 + rpk[y]
        gacc *= 1 + (s["aviation"]["gulf_h51_2026_growth"] if (y == 2026 and pw is None) else rpk[y])
        rpk_idx[y], gulf_rpk_idx[y] = acc, gacc

    el = s["elasticities"]
    x0 = base.output_rows()

    # BJT absolute path
    bj = s["biojet"]
    glob = {int(k): v for k, v in bj["global_mt"].items()}
    g30 = bj["global_mt_2030"][bj["case"]]
    for y in range(2027, 2031):
        glob[y] = glob[2026] + (g30 - glob[2026]) * (y - 2026) / 4
    bjt0 = {c: float(x0.get(make_label(c, "BJT"), 0.0)) for c in base.countries}
    tot0 = sum(v for c, v in bjt0.items() if c not in excl)
    sh22 = {c: (bjt0[c] / tot0 if tot0 > 0 else 0.0) for c in countries}
    sh30 = {c: v for c, v in bj["producer_share_2030"].items() if c in countries}
    z = sum(sh30.values())
    sh30 = {c: v / z for c, v in sh30.items()}
    # USD million per Mt of SAF in the base table (keeps prices at 2022 levels)
    base_mt = glob[2022]
    musd_per_mt = tot0 / base_mt if base_mt > 0 and tot0 > 0 else 2400.0

    rows = []
    for c in countries:
        for sec in sectors:
            lab = make_label(c, sec)
            for i, y in enumerate(years):
                idx, lvl, src = None, np.nan, "IMF_WEO_OCT2026 x elasticity"
                e = float(el.get(sec, el["default"]))
                idx = float(G.loc[c, y] ** e)
                if sec in ("C19", "B06"):
                    idx, src = oil_idx[y], "IEA oil demand index"
                if sec in ("B06", "C19") and c in gulf and y >= 2026 and pw is None:
                    idx, src = oil_idx[2025] * gulf_path[y], "IEA/S&P Gulf output path (Hormuz)"
                if sec == "H51":
                    idx, src = (gulf_rpk_idx[y] if c in gulf else rpk_idx[y]), "IATA RPK"
                if sec == "BGS":
                    idx, src = float(ET.loc[c, y]), "OECD-FAO AO 2026-2035 ethanol"
                if sec == "BDS":
                    idx, src = float(BD.loc[c, y]), "OECD-FAO AO 2026-2035 biomass-based diesel"
                if sec == "BJT":
                    w = 0.0 if y <= 2026 else (y - 2026) / 4
                    share = (1 - w) * sh22.get(c, 0.0) + w * sh30.get(c, 0.0)
                    level = glob[y] * share * musd_per_mt
                    if bjt0[c] > 0:
                        idx = level / bjt0[c]
                    else:
                        idx, lvl = np.nan, level
                    src = f"IATA SAF path ({bj['case']}) x producer shares"
                rows.append({"year": y, "country": c, "sector": sec, "output_index": idx,
                             "output_level_musd": lvl, "source": src})
    out = pd.DataFrame(rows)
    fd = pd.DataFrame([{"year": y, "country": c, "fd": f, "index": float(G.loc[c, y])}
                       for c in countries for y in years for f in FUEL_PATTERN_FD])
    return out, fd


def write(base: ICIO, root: Path, out_dir: Path | None = None, scenario: str = "realised") -> Path:
    out_dir = Path(out_dir or root / "data" / "projections" / scenario)
    out_dir.mkdir(parents=True, exist_ok=True)
    oi, fd = build(base, root, scenario=scenario)
    oi.to_csv(out_dir / "output_index.csv", index=False)
    fd.to_csv(out_dir / "fd_index.csv", index=False)
    log.info("wrote %d output-index rows (Canada excluded) to %s", len(oi), out_dir)
    return out_dir
