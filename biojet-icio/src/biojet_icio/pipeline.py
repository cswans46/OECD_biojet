"""End-to-end routine: ICIO 2022 -> biofuel (and crop) disaggregation -> RAS ->
GHG extension -> TiVA / footprint indicators."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import biofuels, comtrade, crops, environment, indicators, trade, unsd
from .config import Config, load_config
from .disaggregate import apply_splits
from .icio import ICIO, find_icio_file, read_icio, write_icio

log = logging.getLogger(__name__)


def prepare_biofuel_data(cfg: Config, regions: list[str]) -> dict:
    raw = cfg.path("raw_dir")
    data = unsd.load_unsd(raw / "unsd_energy", cfg.year)
    data = unsd.apply_supplements(data, raw / "supplements" / "biojet_supplement.csv")
    bal = unsd.balances_by_region(data, cfg.table("regions"), regions)
    ct = comtrade.load_comtrade(raw / "comtrade", cfg.year)
    ct_icio = comtrade.to_icio(ct, cfg.table("regions"), regions, cfg.table("special_partners")) \
        if len(ct) else ct.assign(reporter=[], partner=[])
    tr = trade.build_all(cfg, bal, ct_icio, regions)
    bal2 = trade.reconcile_balances(bal, tr, regions)
    price = biofuels.prices(cfg, ct_icio, regions)
    return {"unsd": data, "balances_unsd": bal, "comtrade": ct, "comtrade_icio": ct_icio,
            "trade_kt": tr, "balances": bal2, "prices": price}


def run(cfg: Config | None = None, icio: ICIO | None = None, write: bool = True) -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.path("output_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    if icio is None:
        icio = read_icio(find_icio_file(cfg.path("icio_dir"), cfg.year),
                         merge_cn_mx=cfg.settings["base"].get("merge_cn_mx", True))
    regions = icio.countries
    log.info("ICIO: %d regions x %d industries", len(regions), len(icio.sectors_of(regions[0])))
    d = prepare_biofuel_data(cfg, regions)
    ras_kw = {k: cfg.settings["ras"][k] for k in ("max_iter", "tol", "blend") if k in cfg.settings["ras"]}

    results = {"data": d}
    base = icio
    # ---- optional crop split of A01 (proposal: crop-specific feedstock sectors)
    if cfg.settings["crops"].get("enabled", False):
        qv = cfg.path("raw_dir") / "faostat" / "Value_of_Production_E_All_Data_(Normalized).csv"
        if not qv.exists():
            raise FileNotFoundError(f"crops.enabled=true but {qv} is missing (scripts/download_data.py --faostat)")
        ctab = cfg.table("crops")
        codes = dict(zip(ctab.faostat_item_code.astype(int), ctab.crop_code))
        fao = crops.load_faostat_qv(qv, cfg.year, codes)
        csplits = crops.build_crop_splits(base, cfg, fao, cfg.table("regions"))
        cres = apply_splits(base, csplits, ras=True, ras_kw=ras_kw)
        base = cres.icio
        mix = pd.read_csv(cfg.root / "config" / "feedstock_mix.csv")
        cfg.tables["cost"] = crops.remap_feedstock_inputs(cfg.table("cost"), mix, regions)
        results["crop_diagnostics"] = cres.diagnostics

    # ---- biofuel split
    splits, report = biofuels.build_splits(base, cfg, d["balances"], d["trade_kt"], d["prices"])
    res = apply_splits(base, splits, ras=True, ras_kw=ras_kw, keep_pre_ras=True)
    results.update({"icio": res.icio, "pre_ras": res.pre_ras, "diagnostics": res.diagnostics,
                    "biofuel_report": report, "ras_info": res.ras_info})
    log.info("RAS: %s", res.ras_info)

    # ---- environmental extension
    ef = cfg.path("emissions_file")
    if ef.exists():
        e0 = environment.load_emissions(ef, cfg.year)
        parent_of = {f"{s.code}@{r}": p for s in splits for r, p in s.parent.items()}
        e = environment.split_emissions(e0, res.icio, parent_of, rule="energy")
        _, L = indicators.leontief(res.icio)
        results["emissions"] = e
        results["ghg"] = indicators.ghg_footprints(res.icio, e, L)
        results["tiva"] = indicators.tiva(res.icio, L)
    else:
        log.warning("No emissions file at %s: skipping the GHG extension", ef)

    if write:
        write_outputs(cfg, results)
    return results


def write_outputs(cfg: Config, r: dict) -> None:
    out = cfg.path("output_dir")
    y = cfg.year
    write_icio(r["icio"], out / f"ICIO_biofuels_{y}.csv")
    r["diagnostics"].to_csv(out / f"disaggregation_diagnostics_{y}.csv", index=False)
    r["biofuel_report"].to_csv(out / f"biofuel_sectors_{y}.csv", index=False)
    r["data"]["balances"].to_csv(out / f"biofuel_balances_kt_{y}.csv", index=False)
    r["data"]["prices"].to_csv(out / f"biofuel_prices_usd_per_t_{y}.csv")
    for bf, T in r["data"]["trade_kt"].items():
        T.to_csv(out / f"trade_kt_{bf}_{y}.csv")
    if "emissions" in r:
        r["emissions"].to_csv(out / f"emissions_kt_co2e_{y}.csv", header=["kt_co2e"])
        r["ghg"]["summary"].to_csv(out / f"ghg_pba_cba_{y}.csv")
        r["ghg"]["multipliers"].to_csv(out / f"ghg_multipliers_{y}.csv")
        r["tiva"]["exports"].to_csv(out / f"tiva_exports_{y}.csv")
        r["tiva"]["FFD_DVA"].to_csv(out / f"tiva_ffd_dva_{y}.csv")
    log.info("Outputs written to %s", out)
