"""Build the biofuel sector splits (BGS, BDS, BJT, OLB) for the ICIO.

Per region r and biofuel b:

* output   x_rb = production_kt x price_rb / 1000 / (1 - byproduct_share)
                  (USD million; the by-product share covers DDGS, glycerol and
                  renewable naphtha, sold with the use pattern of another product)
* sales    fuel sold to destination s = trade value r->s (or domestic sales
           for s = r), spread over s's purchasers with a key built from UNSD
           consumption shares (road / aviation / other) times s's purchases
           of the fuel-pattern product (C19) from all origins;
* costs    cost_structures.csv shares x x_rb, by input product, split by origin
           with r's intermediate import mix of that product; TLS follows the
           parent's TLS/output ratio and VA is capped so the parent keeps
           non-negative VA.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .comtrade import unit_values
from .disaggregate import SectorSplit, column_groups, origin_mix
from .icio import ICIO, make_label

log = logging.getLogger(__name__)
PATTERN_PRODUCT = "C19"


def parent_for(cfg, region: str, biofuel: str) -> str:
    mode = cfg.dis.get("parent_mode", "proposal")
    if mode == "proposal":
        return cfg.table("biofuels").loc[biofuel, "default_parent"]
    t = cfg.table("parents")
    if mode == "table":
        hit = t[(t.icio == region) & (t.biofuel == biofuel)]
        if len(hit):
            return hit.parent.iloc[0]
    hit = t[(t.icio == "DEFAULT") & (t.biofuel == biofuel)]
    return hit.parent.iloc[0] if len(hit) else "C19"


def prices(cfg, ct_icio: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """USD/t by region and biofuel; Comtrade export unit values within 0.5x-2x of the default."""
    bf = cfg.table("biofuels")
    hs = cfg.table("hs")
    out = pd.DataFrame(index=regions, columns=cfg.dis["biofuels"], dtype=float)
    for b in cfg.dis["biofuels"]:
        default = float(bf.loc[b, "default_price_usd_per_t"])
        out[b] = default
        codes = list(hs.loc[(hs.biofuel == b) & (hs.hs6 != "271020"), "hs6"])
        if cfg.dis.get("use_comtrade_prices", True) and codes and not ct_icio.empty:
            uv = unit_values(ct_icio, codes)
            uv = uv[(uv > 0.5 * default) & (uv < 2.0 * default)]
            out.loc[out.index.intersection(uv.index), b] = uv
    return out


def _cost_shares(cfg, biofuel: str, region: str) -> pd.Series:
    t = cfg.table("cost")
    t = t[t.biofuel == biofuel]
    reg = t[t.region == region]
    use = reg if len(reg) else t[t.region == "DEFAULT"]
    s = use.groupby("input").share.sum()
    if not np.isclose(s.sum(), 1.0, atol=1e-6):
        log.warning("cost shares for %s/%s sum to %.4f; normalising", biofuel, region, s.sum())
        s = s / s.sum()
    return s


def _user_key(icio: ICIO, cfg, dest: str, biofuel: str, bal_row: pd.Series,
              sectors: list[str], col_index: dict[str, int]) -> np.ndarray:
    """Vector over old Z+Y columns, summing to one, for biofuel use in ``dest``."""
    groups = cfg.table("user_groups")
    use = cfg.table("use")
    cats = use[use.biofuel == biofuel]
    cons = float(bal_row.get("consumption", 0.0))
    shares = {}
    for _, c in cats.iterrows():
        if c.category == "road":
            shares[c.user_group] = min(bal_row.get("road", 0.0) / cons, 1.0) if cons > 0 else 0.0
        elif c.category == "aviation":
            shares[c.user_group] = 1.0
    rest = [c.user_group for _, c in cats.iterrows() if c.category == "other"]
    remaining = max(0.0, 1.0 - sum(shares.values()))
    for g in rest:
        shares[g] = shares.get(g, 0.0) + remaining / len(rest)
    if not shares:
        shares = {"all_users": 1.0}

    prow = [make_label(o, PATTERN_PRODUCT) for o in icio.countries]
    n = len(col_index)
    key = np.zeros(n)
    blend = float(cfg.dis.get("blend_to_refiners", 0.0))
    for g, w in shares.items():
        if w <= 0:
            continue
        inds, fds = column_groups(icio, dest, groups[g], sectors)
        cols = inds + fds
        pat = pd.concat([icio.Z.loc[prow, inds].sum(), icio.Y.loc[prow, fds].sum()]).clip(lower=0)
        if pat.sum() <= 0:
            pat = pd.Series(1.0, index=cols)
        if g == "road_users" and blend > 0:
            ref = make_label(dest, PATTERN_PRODUCT)
            key[col_index[ref]] += w * blend
            w = w * (1 - blend)
        idx = [col_index[c] for c in pat.index]
        key[idx] += w * (pat.values / pat.sum())
    return key / key.sum()


def build_splits(icio: ICIO, cfg, bal: pd.DataFrame, trade: dict[str, pd.DataFrame],
                 price: pd.DataFrame) -> tuple[list[SectorSplit], pd.DataFrame]:
    bfinfo = cfg.table("biofuels")
    regions = icio.countries
    sectors = icio.sectors_of(regions[0])
    old_cols = list(icio.Z.columns) + list(icio.Y.columns)
    col_index = {c: i for i, c in enumerate(old_cols)}
    row_index = {r: i for i, r in enumerate(icio.Z.index)}
    x = icio.output_rows()
    va_all = icio.W.loc["VA"]
    tls_all = icio.W.loc["TLS"]
    max_share = float(cfg.dis.get("max_share_of_parent", 0.9))
    min_out = float(cfg.dis.get("min_output_musd", 0.05))
    balr = bal.set_index(["icio", "biofuel"])

    # ---- 1. outputs and caps ---------------------------------------------------
    outputs = {}
    for b in cfg.dis["biofuels"]:
        bp = float(bfinfo.loc[b, "byproduct_share"])
        for r in regions:
            q = balr.loc[(r, b), "production"] if (r, b) in balr.index else 0.0
            fuel_val = q * price.loc[r, b] / 1000.0
            outputs[(r, b)] = (fuel_val, fuel_val / (1 - bp) if bp < 1 else fuel_val)
    by_parent: dict[tuple[str, str], float] = {}
    for (r, b), (_, tot) in outputs.items():
        by_parent[(r, parent_for(cfg, r, b))] = by_parent.get((r, parent_for(cfg, r, b)), 0.0) + tot
    scale = {}
    for (r, p), tot in by_parent.items():
        xp = float(x.get(make_label(r, p), 0.0))
        cap = max_share * max(xp, 0.0)
        scale[(r, p)] = min(1.0, cap / tot) if tot > 0 else 1.0
        if tot > cap:
            log.warning("%s: biofuel output %.0f exceeds %.0f%% of %s output %.0f; scaled by %.2f",
                        r, tot, 100 * max_share, p, xp, scale[(r, p)])

    # ---- 2. per-fuel splits ---------------------------------------------------------
    splits, report = [], []
    for b in cfg.dis["biofuels"]:
        bp = float(bfinfo.loc[b, "byproduct_share"])
        bp_prod = bfinfo.loc[b, "byproduct_pattern_product"]
        T = trade[b].reindex(index=regions, columns=regions).fillna(0.0)
        sp = SectorSplit(code=b, insert_after="C19", parent={}, output={})
        # destination keys (independent of the exporter)
        keys = {}
        for s in regions:
            brow = balr.loc[(s, b)] if (s, b) in balr.index else pd.Series(dtype=float)
            keys[s] = _user_key(icio, cfg, s, b, brow, sectors, col_index)
        # parent VA/TLS shared among fuels with the same parent
        for r in regions:
            p = parent_for(cfg, r, b)
            sp.parent[r] = p
            fuel_val, tot = outputs[(r, b)]
            f = scale[(r, p)]
            fuel_val, tot = fuel_val * f, tot * f
            if tot < min_out:
                continue
            sp.output[r] = tot
            # ------------- sales
            pr_lab = make_label(r, p)
            exports_val = T.loc[r].values * price.loc[r, b] / 1000.0 * f
            exp_tot = exports_val.sum()
            if exp_tot > fuel_val:
                exports_val *= fuel_val / exp_tot
                exp_tot = fuel_val
            row = np.zeros(len(old_cols))
            for j, s in enumerate(regions):
                amt = (fuel_val - exp_tot) if s == r else exports_val[j]
                if amt > 0:
                    row += amt * keys[s]
            if bp > 0 and isinstance(bp_prod, str):
                pat = np.clip(np.concatenate([icio.Z.loc[make_label(r, bp_prod)].values,
                                              icio.Y.loc[make_label(r, bp_prod)].values]), 0, None)
                if pat.sum() > 0:
                    row += (tot - fuel_val) * pat / pat.sum()
                else:
                    row += (tot - fuel_val) * keys[r]
            sp.sales[r] = row
            # ------------- costs
            cs = _cost_shares(cfg, b, r)
            xp = float(x[pr_lab])
            tls_ratio = float(tls_all[pr_lab] / xp) if xp > 0 else 0.0
            va_target = float(cs.get("VA", 0.0)) * tot
            va_parent = float(va_all[pr_lab])
            share_of_parent = tot / xp if xp > 0 else 0.0
            if va_parent <= 0:
                va = va_parent * share_of_parent
            else:
                va = min(va_target, va_parent * min(1.0, share_of_parent / max_share))
            tls = tls_all[pr_lab] * share_of_parent
            inter_total = tot - va - tls
            ish = cs.drop(labels=["VA", "TLS"], errors="ignore")
            ish = ish / ish.sum()
            col = np.zeros(len(icio.Z.index))
            for prod, w in ish.items():
                mix = origin_mix(icio, r, prod)
                col[[row_index[l] for l in mix.index]] += w * inter_total * mix.values
            sp.inputs[r], sp.va[r], sp.tls[r] = col, va, tls
            report.append({"region": r, "biofuel": b, "parent": p, "price_usd_t": price.loc[r, b],
                           "fuel_output_musd": fuel_val, "total_output_musd": tot,
                           "exports_musd": exp_tot, "va_musd": va, "va_target_musd": va_target,
                           "tls_ratio": tls_ratio, "cap_scale": f})
        splits.append(sp)
    return splits, pd.DataFrame(report)
