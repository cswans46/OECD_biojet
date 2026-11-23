"""Country concordances: ICIO codes <-> UNSD names <-> UN Comtrade numeric codes."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)
ROW = "ROW"


def unsd_to_icio(regions: pd.DataFrame, icio_countries: list[str] | None = None) -> dict[str, str]:
    m = dict(zip(regions.unsd_name, regions.icio))
    if icio_countries is not None:
        m = {k: (v if v in icio_countries else ROW) for k, v in m.items()}
    return m


def comtrade_to_icio(regions: pd.DataFrame, icio_countries: list[str] | None = None) -> dict[int, str]:
    m = dict(zip(regions.comtrade_code.astype(int), regions.icio))
    # M49 codes that Comtrade sometimes uses instead of its own reporter codes
    m.update({250: "FRA", 356: "IND", 578: "NOR", 756: "CHE", 840: "USA", 158: "TWN"})
    if icio_countries is not None:
        m = {k: (v if v in icio_countries else ROW) for k, v in m.items()}
    return m


def map_names(names: pd.Series, mapping: dict, what: str = "name") -> pd.Series:
    out = names.map(mapping)
    unmatched = sorted(set(names[out.isna()]))
    if unmatched:
        log.info("%d %ss mapped to ROW: %s", len(unmatched), what, ", ".join(map(str, unmatched[:40])))
    return out.fillna(ROW)
