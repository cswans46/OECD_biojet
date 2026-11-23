"""Hidden supply-chain linkages: look-through vs face-value Gulf exposure of aviation,
embodied Gulf energy in fossil vs bio jet, and UCO/tallow feedstock re-routing.

    python scripts/run_hidden_linkages.py [--icio output/ICIO_biofuels_2022.csv]
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio import comtrade, hidden_linkages as hl  # noqa: E402
from biojet_icio.config import load_config  # noqa: E402
from biojet_icio.icio import make_label, read_icio  # noqa: E402
from biojet_icio.indicators import leontief  # noqa: E402

GULF = ["SAU", "ARE"]
# share of BJT/BDS lipid feedstock that is imported UCO/tallow (EU ~80%, T&E 2024; others assumption)
IMPORTED = {"NLD": 0.8, "BEL": 0.8, "DEU": 0.8, "ESP": 0.8, "FRA": 0.8, "ITA": 0.8, "FIN": 0.8,
            "GBR": 0.8, "SGP": 0.9, "USA": 0.3, "KOR": 0.5}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--icio", default=str(ROOT / "output" / "ICIO_biofuels_2022.csv"))
    ap.add_argument("--out", default=str(ROOT / "output" / "hidden_linkages"))
    ap.add_argument("--mislabel", type=float, default=0.33)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    icio = read_icio(a.icio, merge_cn_mx=False)
    A, L = leontief(icio)
    buyers = [make_label(c, "H51") for c in icio.countries]
    ex = hl.exposure(icio, buyers, ["B06", "C19"], GULF, A, L)
    ex.to_csv(out / "h51_gulf_petroleum_exposure_face_vs_lookthrough.csv")
    eg = hl.embodied_gulf(icio, GULF, row_share=0.3, A=A, L=L)
    eg.to_csv(out / "embodied_gulf_energy.csv", index=False)
    ct = comtrade.load_comtrade(cfg.path("raw_dir") / "comtrade", 2024)
    if len(ct):
        ci = comtrade.to_icio(ct, cfg.table("regions"), icio.countries, cfg.table("special_partners"))
        sh = hl.feedstock_import_shares(ci)
        sh.to_csv(out / "uco_tallow_import_origin_shares_2024.csv")
        imp = {k: v for k, v in IMPORTED.items() if k in icio.countries}
        rows = []
        for lab, mis in (("feedstock_trade", 0.0), ("feedstock_trade_mislabel", a.mislabel)):
            new = hl.apply_feedstock_linkages(icio, sh, imp, mislabel_share=mis)
            A2, L2 = leontief(new)
            e2 = hl.embodied_gulf(new, GULF, row_share=0.3, A=A2, L=L2)
            e2["case"] = lab
            rows.append(e2)
        base = eg.assign(case="icio_default")
        pd.concat([base] + rows).to_csv(out / "embodied_gulf_energy_feedstock_cases.csv", index=False)
    print(ex.sort_values("ratio", ascending=False).head(15).round(4))


if __name__ == "__main__":
    main()
