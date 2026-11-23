"""Biojet substitution and OPEC petroleum supply-shock scenarios.

Parameters live in ``config/scenarios.toml``. The Oxford Economics estimates of
lost oil and gas production and trade from the Hormuz closure are entered
there; they are not bundled.

* ``substitute_biojet``: moves a share theta of each region's H51 purchases of
  C19 (fossil jet, from every origin) to BJT. The energy-equivalent value is
  scaled by the BJT/fossil price ratio, and BJT origins follow the region's BJT
  supply mix (domestic output plus imports, or the global BJT export mix where
  the region has none). H51 direct CO2 falls by theta x its fuel-combustion CO2.
* ``opec_cost_push``: Leontief price model. B06 (and optionally C19) prices
  in the shocked regions rise by the shock size; returns price changes for
  every industry, e.g. H51.
"""
from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from .icio import ICIO, make_label
from .indicators import cost_push, leontief

log = logging.getLogger(__name__)


def load_scenarios(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def bjt_supply_mix(icio: ICIO, region: str) -> pd.Series:
    rows = [make_label(c, "BJT") for c in icio.countries]
    cols = [l for l in list(icio.Z.columns) if l[:3] == region]
    v = icio.Z.loc[rows, cols].sum(axis=1) + icio.Y.loc[rows, [c for c in icio.Y.columns if c[:3] == region]].sum(axis=1)
    if v.sum() <= 0:
        v = icio.output_rows()[rows]
    if v.sum() <= 0:
        raise ValueError("No BJT output anywhere: run the biofuel disaggregation first")
    return v / v.sum()


def substitute_biojet(icio: ICIO, theta: float | dict[str, float], price_ratio: float,
                      emissions: pd.Series | None = None, h51_fuel_co2_share: float = 0.95,
                      regions: list[str] | None = None) -> tuple[ICIO, pd.Series | None]:
    """Return a new ICIO (not re-balanced: H51 columns change cost; BJT/C19 rows
    change sales) and the adjusted emissions vector."""
    out = icio.copy()
    e = emissions.copy() if emissions is not None else None
    regions = regions or icio.countries
    c19_rows = [make_label(c, "C19") for c in icio.countries]
    for r in regions:
        th = theta.get(r, 0.0) if isinstance(theta, dict) else theta
        if th <= 0:
            continue
        h = make_label(r, "H51")
        fossil = out.Z.loc[c19_rows, h]
        moved = th * fossil
        out.Z.loc[c19_rows, h] -= moved
        mix = bjt_supply_mix(icio, r)
        out.Z.loc[mix.index, h] += moved.sum() * price_ratio * mix.values
        # the extra cost lowers H51's operating surplus (prices fixed, short run)
        out.W.loc["VA", h] -= moved.sum() * (price_ratio - 1.0)
        if e is not None and h in e.index:
            e[h] = e[h] * (1 - th * h51_fuel_co2_share)
    return out, e


def opec_cost_push(icio: ICIO, regions: list[str], shock: float,
                   include_c19: bool = False, rest_of_world_weight: float = 0.0) -> pd.Series:
    A, _ = leontief(icio)
    sh = {make_label(r, "B06"): shock for r in regions}
    if rest_of_world_weight > 0:
        sh[make_label("ROW", "B06")] = shock * rest_of_world_weight
    if include_c19:
        sh.update({make_label(r, "C19"): shock for r in regions})
    return cost_push(A, list(sh), sh)
