"""UN Comtrade bilateral trade in biofuel HS codes.

Accepted inputs in ``data/raw/comtrade``:

* ``{X|M}_{reporterCode}_{hs6}_{year}.csv`` with ``partnerCode,primaryValue,netWgt``
  (the layout of the files collected through the public preview API);
* ``comtrade_*.csv`` files written by ``scripts/download_data.py --comtrade``
  (full API layout: ``reporterCode, partnerCode, cmdCode, flowCode,
  primaryValue, netWgt`` and optionally ``partner2Code, customsCode, motCode``).

Partner 0 (World) is used only as a checksum.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .regions import ROW, comtrade_to_icio

log = logging.getLogger(__name__)
COLUMNS = ["reporterCode", "partnerCode", "cmdCode", "flowCode", "primaryValue", "netWgt"]


def load_comtrade(raw_dir: Path, year: int = 2022) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    frames = []
    for f in sorted(raw_dir.glob("*.csv")):
        m = re.match(r"([XM])_(\d+)_(\d{6})_(\d{4})\.csv$", f.name)
        if m:
            flow, rep, hs, yr = m.groups()
            if int(yr) != year:
                continue
            d = pd.read_csv(f)
            d["reporterCode"], d["cmdCode"], d["flowCode"] = int(rep), hs, flow
            frames.append(d)
        elif f.name.startswith("comtrade_"):
            d = pd.read_csv(f, dtype={"cmdCode": str})
            if "period" in d.columns:
                d = d[d.period.astype(int) == year]
            for col, val in (("partner2Code", 0), ("motCode", 0)):
                if col in d.columns:
                    d = d[d[col] == val]
            if "customsCode" in d.columns:
                d = d[d.customsCode == "C00"]
            frames.append(d)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    d = pd.concat(frames, ignore_index=True)[COLUMNS]
    d["cmdCode"] = d.cmdCode.astype(str).str.zfill(6)
    d = d.drop_duplicates(subset=["reporterCode", "partnerCode", "cmdCode", "flowCode"])
    return d


def checksum(d: pd.DataFrame) -> pd.DataFrame:
    """Compare partner sums with the reported World total per reporter/hs/flow."""
    g = d.groupby(["reporterCode", "cmdCode", "flowCode"])
    world = g.apply(lambda x: x.loc[x.partnerCode == 0, "primaryValue"].sum(), include_groups=False)
    parts = g.apply(lambda x: x.loc[x.partnerCode != 0, "primaryValue"].sum(), include_groups=False)
    return pd.DataFrame({"world": world, "partners": parts, "ratio": parts / world.replace(0, np.nan)})


def to_icio(d: pd.DataFrame, regions: pd.DataFrame, icio_countries: list[str],
            special: pd.DataFrame) -> pd.DataFrame:
    """Map reporter/partner codes to ICIO regions and drop special partners."""
    m = comtrade_to_icio(regions, icio_countries)
    drop = set(special.loc[special.treatment.str.startswith(("drop", "total")), "comtrade_code"])
    d = d[~d.partnerCode.isin(drop)].copy()
    d["reporter"] = d.reporterCode.map(m).fillna(ROW)
    d["partner"] = d.partnerCode.map(m).fillna(ROW)
    return d.groupby(["reporter", "partner", "cmdCode", "flowCode"], as_index=False)[
        ["primaryValue", "netWgt"]].sum()


def unit_values(d_icio: pd.DataFrame, hs_codes: list[str], flow: str = "X") -> pd.Series:
    """USD per tonne by reporter (exports, all partners) for a set of HS codes."""
    s = d_icio[(d_icio.cmdCode.isin(hs_codes)) & (d_icio.flowCode == flow)]
    g = s.groupby("reporter")[["primaryValue", "netWgt"]].sum()
    g = g[g.netWgt > 0]
    return (g.primaryValue / (g.netWgt / 1000.0)).rename("usd_per_t")
