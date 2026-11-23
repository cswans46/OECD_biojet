"""Environmental (GHG) extension of the (disaggregated) ICIO.

Input: production-based emissions by ICIO country x industry, e.g. the OECD
"Greenhouse gas footprints (GHGFP) 2025" emissions-by-industry series, or air
emission accounts (AEA) mapped to ICIO industries. Accepted layouts:

* tidy: ``country, industry, value`` (kt CO2e), optional ``gas``;
* OECD SDMX csv export: ``REF_AREA, ACTIVITY, OBS_VALUE`` (+ ``TIME_PERIOD``,
  ``POLLUTANT``/``MEASURE``). Activity "_T" and aggregates are dropped.
  Household direct emissions can be passed with industry ``HH``.

Splitting rule: a parent's emissions go to the remainder and the new sectors in
proportion to their purchases of energy products (B05, B06, C19, D; default) or
to output (``rule='output'``, used for crop splits, where CH4/N2O follow
activity). Biogenic CO2 from burning biofuels is a memo item in inventories,
so road and aviation emissions need no base-year change. Substitution
scenarios adjust user-sector combustion explicitly (see ``scenarios``).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .icio import ICIO, make_label, split_label

log = logging.getLogger(__name__)
ENERGY_PRODUCTS = ("B05", "B06", "C19", "D")


def load_emissions(path: Path, year: int | None = None, gas: str | None = None) -> pd.Series:
    d = pd.read_csv(path)
    cols = {c.upper(): c for c in d.columns}
    if "REF_AREA" in cols:
        if year is not None and "TIME_PERIOD" in cols:
            d = d[d[cols["TIME_PERIOD"]].astype(int) == year]
        for g in ("POLLUTANT", "MEASURE"):
            if gas and g in cols:
                d = d[d[cols[g]] == gas]
        d = pd.DataFrame({"country": d[cols["REF_AREA"]], "industry": d[cols["ACTIVITY"]],
                          "value": d[cols["OBS_VALUE"]]})
    else:
        if gas and "gas" in d.columns:
            d = d[d.gas == gas]
        if year is not None and "year" in d.columns:
            d = d[d.year == year]
    d = d[~d.industry.astype(str).str.startswith("_")]
    s = d.groupby(["country", "industry"]).value.sum()
    s.index = [make_label(c, i) for c, i in s.index]
    return s


def split_emissions(e: pd.Series, new: ICIO, parent_of: dict[str, str], rule: str = "energy",
                    energy_products=ENERGY_PRODUCTS) -> pd.Series:
    """Map an emissions vector on the original labels to the disaggregated labels.

    ``parent_of`` maps each new sector code (e.g. 'BJT') to its parent code in
    every region (or a dict region->parent via 'BJT@USA' keys)."""
    out = pd.Series(0.0, index=new.Z.index)
    common = out.index.intersection(e.index)
    out[common] = e[common]
    hh = e[[i for i in e.index if i.endswith("_HH")]]
    groups: dict[str, list[str]] = {}
    for lab in new.Z.index:
        c, code = split_label(lab)
        p = parent_of.get(f"{code}@{c}", parent_of.get(code))
        if p is not None:
            groups.setdefault(make_label(c, p), []).append(lab)
    erows = [l for l in new.Z.index if split_label(l)[1] in energy_products]
    x = new.output_rows()
    for plab, kids in groups.items():
        members = [plab] + kids
        total = float(e.get(plab, 0.0))
        if total == 0:
            continue
        if rule == "energy":
            w = new.Z.loc[erows, members].sum()
            if w.sum() <= 0:
                w = x[members]
        else:
            w = x[members]
        w = w.clip(lower=0)
        out[members] = total * (w / w.sum()).values if w.sum() > 0 else 0.0
    if len(hh):
        out = pd.concat([out, hh])
    return out


def intensities(e: pd.Series, icio: ICIO) -> pd.Series:
    x = icio.output_rows()
    e = e.reindex(x.index).fillna(0.0)
    return (e / x.where(x > 0)).fillna(0.0)
