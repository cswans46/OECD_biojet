"""UNSD Energy Statistics Database (UNdata 'EDATA') liquid-biofuel balances.

These are the series FAOSTAT's Bioenergy domain (BE) republishes in TJ, so they
cover the same fuels named in the proposal: biogasoline (AL), biodiesel (BD),
bio jet kerosene (BJ) and other liquid biofuels (OL). Units: thousand tonnes.

Two input layouts are accepted in ``data/raw/unsd_energy``:

* the compact per-transaction files collected for this repository
  (``{CM}_{TR}_*.csv`` with columns ``country,quantity_kt`` and
  ``BJ_all_2022.csv`` with ``country,transaction,quantity_kt``);
* a full UNdata download (``UNdata_Export_*.csv``; columns
  ``Country or Area, Commodity - Transaction, Year, Unit, Quantity``), as
  saved by ``scripts/download_data.py --unsd``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .regions import map_names, unsd_to_icio

log = logging.getLogger(__name__)

CMID_TO_BIOFUEL = {"AL": "BGS", "BD": "BDS", "BJ": "BJT", "OL": "OLB"}
COMMODITY_NAME_TO_BIOFUEL = {
    "biogasoline": "BGS", "biodiesel": "BDS", "bio jet kerosene": "BJT",
    "other liquid biofuels": "OLB",
}
TRID_TO_FLOW = {"01": "production", "03": "imports", "04": "exports", "1221": "road"}
TRANSACTION_TO_FLOW = {
    "production": "production", "imports": "imports", "exports": "exports",
    "consumption by road": "road",
    "consumption by domestic aviation": "aviation",
    "international aviation bunkers": "aviation",
    "stock changes": "stock_changes",
}
FLOWS = ["production", "imports", "exports", "road", "aviation"]


def _parse_compact(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(raw_dir.glob("*_*_*.csv")):
        m = re.match(r"([A-Z]{2})_(\d+)_", f.name)
        if not m:
            continue
        cm, tr = m.groups()
        if cm not in CMID_TO_BIOFUEL or tr not in TRID_TO_FLOW:
            continue
        d = pd.read_csv(f)
        if d.empty:
            continue
        frames.append(pd.DataFrame({"unsd_name": d.country, "biofuel": CMID_TO_BIOFUEL[cm],
                                    "flow": TRID_TO_FLOW[tr], "kt": d.quantity_kt.astype(float)}))
    for f in sorted(raw_dir.glob("*_all_*.csv")):
        cm = f.name[:2]
        d = pd.read_csv(f)
        flow = d.transaction.str.strip().str.lower().map(TRANSACTION_TO_FLOW)
        d = d[flow.notna()]
        frames.append(pd.DataFrame({"unsd_name": d.country, "biofuel": CMID_TO_BIOFUEL[cm],
                                    "flow": flow[flow.notna()], "kt": d.quantity_kt.astype(float)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["unsd_name", "biofuel", "flow", "kt"])


def _parse_undata_export(raw_dir: Path, year: int) -> pd.DataFrame:
    frames = []
    for f in sorted(raw_dir.glob("UNdata_Export*.csv")):
        d = pd.read_csv(f)
        d = d[pd.to_numeric(d.get("Year"), errors="coerce") == year]
        ct = d["Commodity - Transaction"].str.split(" - ", n=1, expand=True)
        bio = ct[0].str.strip().str.lower().map(COMMODITY_NAME_TO_BIOFUEL)
        flow = ct[1].str.strip().str.lower().map(TRANSACTION_TO_FLOW)
        keep = bio.notna() & flow.notna()
        frames.append(pd.DataFrame({"unsd_name": d.loc[keep, "Country or Area"],
                                    "biofuel": bio[keep], "flow": flow[keep],
                                    "kt": pd.to_numeric(d.loc[keep, "Quantity"], errors="coerce")}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["unsd_name", "biofuel", "flow", "kt"])


def load_unsd(raw_dir: Path, year: int = 2022) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    full = _parse_undata_export(raw_dir, year)
    compact = _parse_compact(raw_dir)
    data = full if not full.empty else compact
    if not full.empty and not compact.empty:
        # prefer the full export, fill gaps from the compact files
        key = ["unsd_name", "biofuel", "flow"]
        data = pd.concat([full, compact]).drop_duplicates(subset=key, keep="first")
    return data.groupby(["unsd_name", "biofuel", "flow"], as_index=False).kt.sum()


def apply_supplements(data: pd.DataFrame, supplement_file: Path) -> pd.DataFrame:
    """Add biojet observations missing from UNSD (e.g. US SAF reported to EPA).

    Rows with ``reclassify_from`` set move the same quantity out of another
    biofuel (e.g. US SAF is inside UNSD 'other liquid biofuels')."""
    if not Path(supplement_file).exists():
        return data
    sup = pd.read_csv(supplement_file)
    data = data.copy()
    for _, s in sup.iterrows():
        key = (data.unsd_name == s.unsd_name) & (data.biofuel == s.biofuel) & (data.flow == s.flow)
        if key.any():
            if str(s.get("mode", "replace_if_missing")) != "override":
                continue
            data = data[~key]
        data = pd.concat([data, pd.DataFrame([{"unsd_name": s.unsd_name, "biofuel": s.biofuel,
                                               "flow": s.flow, "kt": float(s.kt)}])])
        src = s.get("reclassify_from")
        if isinstance(src, str) and src:
            k2 = (data.unsd_name == s.unsd_name) & (data.biofuel == src) & (data.flow == s.flow)
            data.loc[k2, "kt"] = (data.loc[k2, "kt"] - float(s.kt)).clip(lower=0)
        log.info("Supplement: %s %s %s = %.1f kt (%s)", s.unsd_name, s.biofuel, s.flow, s.kt, s.source)
    return data.reset_index(drop=True)


def balances_by_region(data: pd.DataFrame, regions: pd.DataFrame,
                       icio_countries: list[str]) -> pd.DataFrame:
    """Aggregate to ICIO regions; returns rows (icio, biofuel) x FLOWS + apparent consumption."""
    d = data.copy()
    d["icio"] = map_names(d.unsd_name, unsd_to_icio(regions, icio_countries), "UNSD country")
    wide = d.pivot_table(index=["icio", "biofuel"], columns="flow", values="kt",
                         aggfunc="sum", fill_value=0.0)
    for f in FLOWS:
        if f not in wide.columns:
            wide[f] = 0.0
    wide = wide[FLOWS]
    wide["apparent_consumption"] = (wide.production + wide.imports - wide.exports).clip(lower=0.0)
    return wide.reset_index()
