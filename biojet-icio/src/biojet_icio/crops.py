"""Split A01 (Agriculture and hunting) into biofuel feedstock crops.

The proposal asks for crop-specific supply, use and trade carved out of A01
with FAOSTAT production data and UN Comtrade crop trade. Inputs:

* FAOSTAT QV (Value of Agricultural Production), element "Gross Production
  Value (current thousand US$)", for the items in config/feedstock_crops.csv.
  Download with ``scripts/download_data.py --faostat``.
* Optional Comtrade bilateral exports for the crops' HS codes.

Method (used when crop-level cost data are missing): a crop's cost column is
proportional to A01's column (crop output / A01 output). Its sales row follows
A01's row, rescaled so exports match Comtrade where it is available. The
biofuel split then sends ethanol and oil feedstock purchases to the crop
sectors using config/feedstock_mix.csv.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .disaggregate import SectorSplit
from .icio import ICIO, make_label
from .regions import map_names

log = logging.getLogger(__name__)
FAO_TO_ICIO_NAME_FIX = {"United States of America": "United States",
                        "Netherlands (Kingdom of the)": "Netherlands (Kingd. of the)",
                        "Republic of Korea": "Korea, Republic of",
                        "China, mainland": "China", "China, Taiwan Province of": "Other Asia",
                        "Türkiye": "Türkiye", "Russian Federation": "Russian Federation",
                        "Slovakia": "Slovakia", "Czechia": "Czechia"}


def load_faostat_qv(path: Path, year: int, item_codes: dict[int, str]) -> pd.DataFrame:
    """Return (unsd-style country name, crop_code, value_musd)."""
    d = pd.read_csv(path, encoding="latin-1")
    d = d[(d.Year == year) & d["Item Code"].isin(item_codes) &
          d.Element.str.contains("Gross Production Value", case=False) &
          d.Unit.str.contains("1000 USD|thousand US", case=False, regex=True)]
    d = d[~d.Unit.str.contains("constant", case=False)]
    return pd.DataFrame({"name": d.Area.replace(FAO_TO_ICIO_NAME_FIX),
                         "crop": d["Item Code"].map(item_codes),
                         "value_musd": d.Value / 1000.0})


def build_crop_splits(icio: ICIO, cfg, fao: pd.DataFrame, regions_tbl: pd.DataFrame,
                      exports_musd: pd.DataFrame | None = None) -> list[SectorSplit]:
    crops = cfg.settings["crops"]["crops"]
    m = dict(zip(regions_tbl.unsd_name, regions_tbl.icio))
    fao = fao.copy()
    fao["icio"] = map_names(fao.name, {k: v for k, v in m.items() if v in icio.countries}, "FAO area")
    val = fao.groupby(["icio", "crop"]).value_musd.sum()
    old_cols = list(icio.Z.columns) + list(icio.Y.columns)
    x = icio.output_rows()
    splits = []
    for crop in crops:
        sp = SectorSplit(code=crop, insert_after="A01", parent={}, output={})
        for r in icio.countries:
            sp.parent[r] = "A01"
            xa = float(x[make_label(r, "A01")])
            v = float(val.get((r, crop), 0.0))
            if v <= 0 or xa <= 0:
                continue
            tot_crops = sum(float(val.get((r, c), 0.0)) for c in crops)
            f = min(1.0, 0.9 * xa / tot_crops) if tot_crops > 0 else 1.0
            v *= f
            sh = v / xa
            sp.output[r] = v
            lab = make_label(r, "A01")
            row = np.concatenate([icio.Z.loc[lab].values, icio.Y.loc[lab].values]).clip(min=0)
            row = row / row.sum() * v if row.sum() > 0 else np.zeros(len(old_cols))
            if exports_musd is not None and (r, crop) in exports_musd.index:
                # rescale foreign destinations to the Comtrade export value
                foreign = np.array([c[:3] != r for c in old_cols])
                e = float(exports_musd.loc[(r, crop)])
                if row[foreign].sum() > 0:
                    e = min(e, 0.95 * v)
                    row[foreign] *= e / row[foreign].sum()
                    row[~foreign] *= (v - e) / max(row[~foreign].sum(), 1e-12)
            sp.sales[r] = row
            sp.inputs[r] = icio.Z[lab].values * sh
            sp.va[r] = float(icio.W.loc["VA", lab]) * sh
            sp.tls[r] = float(icio.W.loc["TLS", lab]) * sh
        splits.append(sp)
    return splits


def remap_feedstock_inputs(cost: pd.DataFrame, mix: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Replace an A01 feedstock line in cost_structures with crop lines per region."""
    rows = []
    for b in cost.biofuel.unique():
        for r in regions:
            reg = cost[(cost.biofuel == b) & (cost.region == r)]
            base = reg if len(reg) else cost[(cost.biofuel == b) & (cost.region == "DEFAULT")]
            mx = mix[(mix.biofuel == b) & (mix.icio == r)]
            if mx.empty:
                mx = mix[(mix.biofuel == b) & (mix.icio == "DEFAULT")]
            replaced = mx.input_replaced.iloc[0] if len(mx) else None
            for _, line in base.iterrows():
                if replaced is not None and line.input == replaced:
                    for _, c in mx.iterrows():
                        rows.append({"biofuel": b, "region": r, "input": c.crop,
                                     "share": line.share * c.share, "source_note": "feedstock_mix"})
                    rest = 1 - mx.share.sum()
                    if rest > 1e-9:
                        rows.append({**line.to_dict(), "region": r, "share": line.share * rest})
                else:
                    rows.append({**line.to_dict(), "region": r})
    return pd.DataFrame(rows)
