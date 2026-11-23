"""External projections used to drive the ICIO to 2030.

Sources (all public)
--------------------
* IMF World Economic Outlook (DataMapper API, no key): real GDP growth
  (NGDP_RPCH), GDP in current USD (NGDPD), import and export volume growth
  (TM_RPCH, TX_RPCH), annual to 2030/2031, every ICIO economy.
* EIA International Energy Outlook (IEO, API v2 route ``ieo/<year>``, free
  key in EIA_API_KEY): gross output by region and sector (Table K2 and the
  services table), delivered energy by end-use sector and fuel, electricity
  generation, primary energy production, 16 world regions.
* EIA Annual Energy Outlook (AEO, route ``aeo/<year>``): US value of
  shipments by industry, energy use by sector and fuel, energy prices.

All sources are reduced to one tidy table

    source, region, item, year, value

with items such as ``GDP_R`` (index or level), ``GDPD_USD`` (USD deflator),
``IMP_R``, ``EXP_R``, ``GO:<ieo_sector>``, ``SHIP:<aeo_industry>``,
``ENERGY:<enduse>:<fuel>``, ``PROD:<fuel>``, ``PRICE:<fuel>``. Fetchers write
that table; it can also be filled by hand from the published XLSX tables.
Only ratios to the base year are used, so units do not matter.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .table import country_of, sector_of

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Concordances
# ----------------------------------------------------------------------------

IEO_REGION = {
    "USA": "United States", "CAN": "Canada", "MEX": "Mexico and Chile", "CHL": "Mexico and Chile",
    "JPN": "Japan", "KOR": "South Korea", "AUS": "Australia and New Zealand", "NZL": "Australia and New Zealand",
    "RUS": "Russia", "CHN": "China", "IND": "India", "BRA": "Brazil",
}
for c in ("AUT BEL CZE DNK EST FIN FRA DEU GRC HUN ISL IRL ITA LVA LTU LUX NLD NOR POL PRT SVK SVN ESP "
          "SWE CHE TUR GBR ISR").split():
    IEO_REGION[c] = "OECD Europe"
for c in "BGR HRV CYP MLT ROU BLR UKR KAZ GEO ARM AZE UZB SRB MKD ALB BIH MNE MDA".split():
    IEO_REGION[c] = "Other non-OECD Europe and Eurasia"
for c in "SAU ARE JOR KWT QAT OMN BHR IRN IRQ LBN SYR YEM".split():
    IEO_REGION[c] = "Middle East"
for c in "EGY MAR TUN ZAF NGA CMR CIV SEN AGO COD KEN ETH GHA TZA DZA LBY ZMB ZWE MOZ".split():
    IEO_REGION[c] = "Africa"
for c in "BGD BRN KHM HKG IDN LAO MYS MMR PAK PHL SGP TWN THA VNM NPL LKA MNG".split():
    IEO_REGION[c] = "Other non-OECD Asia"
for c in "ARG COL CRI PER ECU URY PRY BOL VEN GTM PAN DOM".split():
    IEO_REGION[c] = "Other non-OECD Americas"

# ICIO sector code (regex, first match) -> IEO gross-output sector
IEO_SECTOR = [
    (r"^A", "agriculture"),
    (r"^B", "mining"),
    (r"^C10T12", "food"),
    (r"^C17_18", "paper"),
    (r"^C19", "refining"),
    (r"^C20", "basic_chemicals"),
    (r"^C23", "nonmetallic_minerals"),
    (r"^C24A", "iron_steel"),
    (r"^C24B", "nonferrous_metals"),
    (r"^C24$", "iron_steel"),
    (r"^C", "other_manufacturing"),
    (r"^F", "construction"),
    (r".", "services"),
]

# ICIO sector -> AEO industry (US value of shipments)
AEO_INDUSTRY = [
    (r"^A", "agriculture"), (r"^B", "mining"), (r"^F", "construction"),
    (r"^C10T12", "food"), (r"^C16", "wood"), (r"^C17_18", "paper"), (r"^C20", "bulk_chemicals"),
    (r"^C22", "plastics"), (r"^C23", "nonmetallic_minerals"), (r"^C24A", "iron_steel"),
    (r"^C24B", "aluminum"), (r"^C25", "fabricated_metals"), (r"^C26", "computers_electronics"),
    (r"^C27", "electrical_equipment"), (r"^C28", "machinery"), (r"^C29|^C30", "transportation_equipment"),
    (r"^C19", "refining"), (r"^C", "balance_of_manufacturing"),
]

# Energy sectors whose output volume follows IEO energy production/generation
ENERGY_OUTPUT = [(r"^D35$|^D$", "PROD:electricity"), (r"^B05", "PROD:coal"), (r"^B06", "PROD:liquids_gas")]
# ICIO input sector -> fuel (for input-demand coefficient changes)
ENERGY_INPUT = [(r"^D35$|^D$", "electricity"), (r"^B06", "natural_gas"), (r"^C19", "liquids"), (r"^B05", "coal")]
# ICIO using sector -> end-use sector
END_USE = [(r"^[ABCF]|^D|^E", "industrial"), (r"^H", "transportation"), (r".", "buildings")]
# Energy sector -> price item
ENERGY_PRICE = [(r"^B06|^C19", "PRICE:brent"), (r"^D35$|^D$", "PRICE:electricity"), (r"^B05", "PRICE:coal")]


def first_match(code: str, table: list[tuple[str, str]]) -> str | None:
    for pat, val in table:
        if re.search(pat, code):
            return val
    return None


# ----------------------------------------------------------------------------
# Fetchers (need internet; run on the user's machine)
# ----------------------------------------------------------------------------

WEO_URL = "https://www.imf.org/external/datamapper/api/v1/{ind}"
WEO_ITEMS = {"NGDP_RPCH": "GDP_R_GROWTH", "NGDPD": "GDP_USD", "TM_RPCH": "IMP_R_GROWTH", "TX_RPCH": "EXP_R_GROWTH"}


def fetch_weo(indicators=tuple(WEO_ITEMS)) -> pd.DataFrame:  # pragma: no cover - internet
    rows = []
    for ind in indicators:
        with urllib.request.urlopen(WEO_URL.format(ind=ind), timeout=120) as r:
            data = json.loads(r.read().decode())
        for ctry, series in data.get("values", {}).get(ind, {}).items():
            for y, v in series.items():
                if v is not None:
                    rows.append({"source": "IMF WEO", "region": ctry, "item": WEO_ITEMS[ind], "year": int(y), "value": float(v)})
        log.info("WEO %s: %s countries", ind, len(data.get("values", {}).get(ind, {})))
    return pd.DataFrame(rows)


EIA_BASE = "https://api.eia.gov/v2/{route}"

# Regex on EIA series names -> tidy item (first match wins). Checked against
# the series list printed by `--list-series`; adjust if EIA renames series.
EIA_RULES = [
    (r"gross output.*agricultur", "GO:agriculture"), (r"gross output.*mining", "GO:mining"),
    (r"gross output.*construction", "GO:construction"), (r"gross output.*food", "GO:food"),
    (r"gross output.*paper", "GO:paper"), (r"gross output.*refin", "GO:refining"),
    (r"gross output.*(basic )?chemical", "GO:basic_chemicals"), (r"gross output.*iron", "GO:iron_steel"),
    (r"gross output.*nonferrous", "GO:nonferrous_metals"), (r"gross output.*nonmetallic", "GO:nonmetallic_minerals"),
    (r"gross output.*non-?energy.?intensive|gross output.*other manufactur", "GO:other_manufacturing"),
    (r"gross output.*services", "GO:services"),
    (r"gross domestic product", "GDP_R"),
    (r"industrial.*electricity", "ENERGY:industrial:electricity"),
    (r"industrial.*natural gas", "ENERGY:industrial:natural_gas"),
    (r"industrial.*(liquids|petroleum)", "ENERGY:industrial:liquids"), (r"industrial.*coal", "ENERGY:industrial:coal"),
    (r"transportation.*electricity", "ENERGY:transportation:electricity"),
    (r"transportation.*natural gas", "ENERGY:transportation:natural_gas"),
    (r"transportation.*(liquids|petroleum)", "ENERGY:transportation:liquids"),
    (r"(residential|commercial|buildings).*electricity", "ENERGY:buildings:electricity"),
    (r"(residential|commercial|buildings).*natural gas", "ENERGY:buildings:natural_gas"),
    (r"(residential|commercial|buildings).*(liquids|petroleum)", "ENERGY:buildings:liquids"),
    (r"net (electricity )?generation", "PROD:electricity"), (r"coal production", "PROD:coal"),
    (r"(liquids|crude oil|natural gas) production", "PROD:liquids_gas"),
    (r"brent", "PRICE:brent"), (r"electricity.*price|price.*electricity", "PRICE:electricity"),
    (r"value of shipments.*food", "SHIP:food"), (r"value of shipments.*paper", "SHIP:paper"),
    (r"value of shipments.*bulk chemical", "SHIP:bulk_chemicals"), (r"value of shipments.*iron", "SHIP:iron_steel"),
    (r"value of shipments.*alumin", "SHIP:aluminum"), (r"value of shipments.*glass|value of shipments.*cement", "SHIP:nonmetallic_minerals"),
    (r"value of shipments.*fabricated", "SHIP:fabricated_metals"), (r"value of shipments.*machinery", "SHIP:machinery"),
    (r"value of shipments.*computer", "SHIP:computers_electronics"), (r"value of shipments.*transportation equip", "SHIP:transportation_equipment"),
    (r"value of shipments.*electrical equip", "SHIP:electrical_equipment"), (r"value of shipments.*wood", "SHIP:wood"),
    (r"value of shipments.*plastic", "SHIP:plastics"), (r"value of shipments.*balance of manufactur", "SHIP:balance_of_manufacturing"),
    (r"value of shipments.*agricultur", "SHIP:agriculture"), (r"value of shipments.*mining", "SHIP:mining"),
    (r"value of shipments.*construction", "SHIP:construction"), (r"value of shipments.*refin", "SHIP:refining"),
]


def _eia_get(route: str, api_key: str, params: dict) -> dict:  # pragma: no cover - internet
    q = urllib.parse.urlencode({"api_key": api_key, **params}, doseq=True)
    with urllib.request.urlopen(f"{EIA_BASE.format(route=route)}?{q}", timeout=180) as r:
        return json.loads(r.read().decode())


def list_eia_series(route: str, api_key: str | None = None) -> pd.DataFrame:  # pragma: no cover
    key = api_key or os.environ["EIA_API_KEY"]
    d = _eia_get(f"{route}/facet/seriesId", key, {})
    return pd.DataFrame(d["response"]["facets"])


def fetch_eia(route: str, source: str, api_key: str | None = None, scenario_regex: str = r"ref",
              rules=EIA_RULES) -> pd.DataFrame:  # pragma: no cover - internet
    """Fetch the series matched by ``rules`` from an EIA AEO/IEO route."""
    key = api_key or os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError("Set EIA_API_KEY (free key: https://www.eia.gov/opendata/register.php)")
    series = list_eia_series(route, key)
    series["item"] = [first_match(str(n).lower(), [(p, i) for p, i in rules]) for n in series["name"]]
    series = series.dropna(subset=["item"])
    scen = pd.DataFrame(_eia_get(f"{route}/facet/scenario", key, {})["response"]["facets"])
    scen_ids = scen.loc[scen["name"].str.lower().str.contains(scenario_regex) | scen["id"].str.lower().str.contains(scenario_regex), "id"].tolist()[:1]
    rows = []
    for i in range(0, len(series), 50):
        chunk = series.iloc[i:i + 50]
        params = {"frequency": "annual", "data[0]": "value", "facets[seriesId][]": chunk["id"].tolist(),
                  "length": 5000}
        if scen_ids:
            params["facets[scenario][]"] = scen_ids
        d = _eia_get(f"{route}/data", key, params)
        for r in d["response"]["data"]:
            item = dict(zip(chunk["id"], chunk["item"])).get(r.get("seriesId"))
            region = r.get("regionName") or r.get("regionId") or "United States"
            try:
                rows.append({"source": source, "region": region, "item": item,
                             "year": int(r["period"]), "value": float(r["value"])})
            except (TypeError, ValueError):
                continue
    log.info("%s: %s rows for %s items", source, len(rows), series["item"].nunique())
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Access
# ----------------------------------------------------------------------------

@dataclass
class External:
    tidy: pd.DataFrame
    base_year: int = 2022

    @classmethod
    def load(cls, path: Path, base_year: int = 2022) -> "External":
        df = pd.read_csv(path)
        need = {"region", "item", "year", "value"}
        if not need <= set(df.columns):
            raise ValueError(f"{path} needs columns {sorted(need)}")
        df["region"] = df["region"].astype(str)
        return cls(df, base_year)

    def _series(self, region: str, item: str) -> pd.Series | None:
        s = self.tidy[(self.tidy["region"] == region) & (self.tidy["item"] == item)]
        if s.empty:
            return None
        s = s.groupby("year")["value"].mean().sort_index()
        full = pd.Series(np.nan, index=range(int(s.index.min()), int(s.index.max()) + 1))
        full.loc[s.index] = s.to_numpy()
        # IEO publishes 2022, 2025, 2030: interpolate in logs
        if (full > 0).all() or full.dropna().gt(0).all():
            full = np.exp(np.log(full).interpolate())
        else:
            full = full.interpolate()
        return full

    def index(self, region: str, item: str, years) -> pd.Series | None:
        """Level ratio to the base year for ``years`` (None if not covered)."""
        s = self._series(region, item)
        if s is None or self.base_year not in s.index or s[self.base_year] == 0:
            return None
        s = s / s[self.base_year]
        years = list(years)
        if max(years) > s.index.max():
            # extend with the last 5-year average growth
            g = (s.iloc[-1] / s.iloc[max(0, len(s) - 6)]) ** (1 / max(1, min(5, len(s) - 1)))
            for y in range(s.index.max() + 1, max(years) + 1):
                s.loc[y] = s.iloc[-1] * g
        return s.reindex(years)

    def growth_index(self, region: str, item: str, years) -> pd.Series | None:
        """Index from annual percentage growth series (WEO *_GROWTH items)."""
        s = self._series(region, item)
        if s is None:
            return None
        years = list(years)
        idx, level = {}, 1.0
        for y in range(self.base_year + 1, max(years) + 1):
            g = s.get(y, np.nan)
            if not np.isfinite(g):
                g = s.dropna().iloc[-1]
            level *= 1 + g / 100.0
            idx[y] = level
        idx[self.base_year] = 1.0
        return pd.Series(idx).reindex(years)

    # --- convenience ----------------------------------------------------
    def gdp_real(self, country: str, years) -> pd.Series | None:
        return self.growth_index(country, "GDP_R_GROWTH", years)

    def usd_deflator(self, country: str, years) -> pd.Series | None:
        nominal = self.index(country, "GDP_USD", years)
        real = self.gdp_real(country, years)
        if nominal is None or real is None:
            return None
        return nominal / real

    def trade_volume(self, country: str, flow: str, years) -> pd.Series | None:
        return self.growth_index(country, f"{flow}_R_GROWTH", years)


def world_fallback(ext: External, countries: list[str], years, fn) -> dict[str, pd.Series]:
    """Apply fn(country, years) with a GDP-weighted 'World' fallback."""
    out = {}
    for c in countries:
        s = fn(c, years)
        if s is None:
            s = fn("WEOWORLD", years)
        if s is None:
            s = pd.Series(1.0, index=list(years))
        out[c] = s
    return out


def write_tidy(frames: list[pd.DataFrame], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def fetch_all(path: Path, ieo_route: str = "ieo/2023", aeo_route: str = "aeo/2026") -> Path:  # pragma: no cover
    frames = [fetch_weo()]
    try:
        frames.append(fetch_eia(ieo_route, "EIA IEO"))
        frames.append(fetch_eia(aeo_route, "EIA AEO"))
    except Exception as exc:  # noqa: BLE001
        log.warning("EIA fetch failed (%s): external method will use WEO only.", exc)
    return write_tidy(frames, path)


__all__ = ["External", "IEO_REGION", "IEO_SECTOR", "AEO_INDUSTRY", "first_match", "fetch_all",
           "fetch_weo", "fetch_eia", "list_eia_series", "country_of", "sector_of"]
