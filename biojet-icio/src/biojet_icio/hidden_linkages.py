"""Supply-chain linkages that bilateral customs (Comtrade) data do not show.

1. Look-through vs face-value exposure (Baldwin, Freeman & Theodorakopoulos,
   2022 NBER WP 30525 and 2023 BPEA). Face value: the share of a buyer's direct
   purchases of a product group sourced from a supplier country. Look-through:
   the same share computed on total (direct + indirect) requirements, A L,
   which shows e.g. Gulf crude reaching an airline through Singaporean, Korean
   or Indian jet fuel.

2. Embodied Gulf energy per unit of output: Gulf B06 (+ C19, C20) content of
   one USD of each sector's output, sum over Gulf origins of (A L)_ij, split
   into direct and indirect parts.

3. Feedstock trade for HEFA/FAME: used cooking oil (HS 1518.00) and tallow
   (HS 1502.10) imports re-route the origin of BJT/BDS lipid inputs. ICIO
   origins come from the average C10T12 import mix, which misses the
   China->EU/US UCO trade. A sensitivity moves a share of Chinese "UCO" to
   palm-oil origins (IDN/MYS) to reflect suspected mislabelling (T&E 2023:
   about a third likely not genuine). The changed table is re-balanced with RAS.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .icio import ICIO, make_label, split_label
from .indicators import leontief
from .ras import ras_signed

log = logging.getLogger(__name__)


def exposure(icio: ICIO, buyers: list[str], products: list[str], suppliers: list[str],
             A: pd.DataFrame | None = None, L: pd.DataFrame | None = None) -> pd.DataFrame:
    """Face-value and look-through shares of ``suppliers`` in each buyer's sourcing of ``products``."""
    if A is None or L is None:
        A, L = leontief(icio)
    T = A.values @ L.values                                    # total input requirements per unit output
    lab = list(icio.Z.index)
    pos = {l: i for i, l in enumerate(lab)}
    prod_rows = [i for i, l in enumerate(lab) if split_label(l)[1] in products]
    sup_rows = [i for i in prod_rows if lab[i][:3] in suppliers]
    out = []
    for b in buyers:
        if b not in pos:
            continue
        j = pos[b]
        fv_den = A.values[prod_rows, j].sum()
        lt_den = T[prod_rows, j].sum()
        fv = A.values[sup_rows, j].sum() / fv_den if fv_den > 0 else 0.0
        lt = T[sup_rows, j].sum() / lt_den if lt_den > 0 else 0.0
        out.append({"buyer": b, "face_value": fv, "look_through": lt,
                    "ratio": lt / fv if fv > 0 else np.inf if lt > 0 else np.nan,
                    "direct_req": A.values[sup_rows, j].sum(), "total_req": T[sup_rows, j].sum()})
    return pd.DataFrame(out).set_index("buyer")


def embodied_gulf(icio: ICIO, gulf: list[str], products=("B06", "C19", "C20"),
                  sectors=("H51", "C19", "BJT", "BDS", "H49"), row_share: float = 0.0,
                  A: pd.DataFrame | None = None, L: pd.DataFrame | None = None) -> pd.DataFrame:
    """USD of Gulf-origin ``products`` embodied (direct + indirect) per USD of output.
    ``row_share`` adds that share of ROW rows (Iran, Iraq, Kuwait, Qatar... sit in ROW)."""
    if A is None or L is None:
        A, L = leontief(icio)
    T = A.values @ L.values
    lab = list(icio.Z.index)
    w = np.zeros(len(lab))
    for i, l in enumerate(lab):
        c, p = split_label(l)
        if p in products:
            w[i] = 1.0 if c in gulf else (row_share if c == "ROW" else 0.0)
    rows = []
    for j, l in enumerate(lab):
        c, p = split_label(l)
        if p not in sectors:
            continue
        rows.append({"country": c, "sector": p, "direct": float(w @ A.values[:, j]),
                     "total": float(w @ T[:, j])})
    d = pd.DataFrame(rows)
    d["indirect"] = d.total - d.direct
    d["hidden_share"] = d.indirect / d.total.where(d.total > 0)
    return d


def feedstock_import_shares(ct_icio: pd.DataFrame, hs=("151800", "150210")) -> pd.DataFrame:
    """Importer x origin shares (by value) of waste-lipid feedstock imports."""
    s = ct_icio[ct_icio.cmdCode.isin(hs) & (ct_icio.flowCode == "M")]
    m = s.pivot_table(index="reporter", columns="partner", values="primaryValue", aggfunc="sum", fill_value=0.0)
    for r in m.index:
        if r in m.columns:
            m.loc[r, r] = 0.0
    return m.div(m.sum(axis=1).where(m.sum(axis=1) > 0), axis=0).fillna(0.0)


def apply_feedstock_linkages(icio: ICIO, shares: pd.DataFrame, imported_share: dict[str, float],
                             fuels=("BJT", "BDS"), products=("C10T12", "E"),
                             mislabel_share: float = 0.0, mislabel_from: str = "CHN",
                             mislabel_to: dict[str, float] | None = None,
                             ras: bool = True) -> ICIO:
    """Re-route the origin of lipid inputs of ``fuels`` in importing regions.

    For buyer (r, fuel), a fraction ``imported_share[r]`` (default 0) of its
    ``products`` purchases is re-sourced with r's UCO/tallow import origins.
    The rest keeps the ICIO origin mix. ``mislabel_share`` of the volume
    attributed to ``mislabel_from`` is moved to palm-oil origins
    ``mislabel_to`` (default IDN 0.6, MYS 0.4). Row totals of the origin rows
    are then restored by RAS (sales of other buyers adjust)."""
    mislabel_to = mislabel_to or {"IDN": 0.6, "MYS": 0.4}
    out = icio.copy()
    Z = out.Z
    for r, sh in imported_share.items():
        if r not in shares.index or sh <= 0:
            continue
        orig = shares.loc[r].copy()
        if mislabel_share > 0 and mislabel_from in orig.index:
            moved = orig[mislabel_from] * mislabel_share
            orig[mislabel_from] -= moved
            for k, v in mislabel_to.items():
                if k in orig.index:
                    orig[k] += moved * v
                else:
                    orig[k] = moved * v
        orig = orig[orig > 0]
        for f in fuels:
            col = make_label(r, f)
            if col not in Z.columns:
                continue
            for p in products:
                rows = [make_label(c, p) for c in out.countries]
                total = Z.loc[rows, col].sum()
                if total <= 0:
                    continue
                re = total * sh
                Z.loc[rows, col] *= (1 - sh)
                for o, w in orig.items():
                    lab = make_label(o, p)
                    if lab in Z.index:
                        Z.loc[lab, col] += re * w
    if ras:
        x = icio.output_rows().values
        M = np.hstack([Z.values, out.Y.values])
        col_t = np.concatenate([x - out.W[list(Z.columns)].sum().values, out.Y.values.sum(axis=0)])
        Mb, info = ras_signed(M, x, col_t, label="feedstock linkages", max_iter=5000, tol=1e-7)
        nZ = Z.shape[0]
        out.Z = pd.DataFrame(Mb[:, :nZ], Z.index, Z.columns)
        out.Y = pd.DataFrame(Mb[:, nZ:], out.Y.index, out.Y.columns)
    return out
