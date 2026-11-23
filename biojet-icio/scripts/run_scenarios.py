"""OPEC shock x biojet-share grid on the disaggregated ICIO.

    python scripts/run_scenarios.py [--icio output/ICIO_biofuels_2022.csv]

For each biojet share theta (config/scenarios.toml):
  * substitute fossil jet (C19 -> H51) with BJT;
  * Leontief cost push from the OPEC B06 price shock: price change of H51
    and of every product (sensitivity of air transport to petroleum);
  * pass-through frequency of shocked-region B06 value added through H51 into
    world final demand;
  * total GHG intensity of H51 output (if an emissions vector exists).
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio import indicators, scenarios  # noqa: E402
from biojet_icio.config import load_config  # noqa: E402
from biojet_icio.icio import make_label, read_icio  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--icio", default=None)
    ap.add_argument("--emissions", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    y = cfg.year
    out = cfg.path("output_dir")
    icio = read_icio(a.icio or out / f"ICIO_biofuels_{y}.csv", merge_cn_mx=False)
    sc = scenarios.load_scenarios(ROOT / "config" / "scenarios.toml")
    em_path = Path(a.emissions) if a.emissions else out / f"emissions_kt_co2e_{y}.csv"
    e = pd.read_csv(em_path, index_col=0).iloc[:, 0] if em_path.exists() else None
    op, bj = sc["opec_shock"], sc["biojet"]
    if op["price_shock"] == 0:
        logging.warning("opec_shock.price_shock is 0 (placeholder): enter the Oxford Economics estimate")
    rows = []
    for th in bj["theta"]:
        ic, e2 = scenarios.substitute_biojet(icio, th, bj["price_ratio"], e, bj["h51_fuel_co2_share"],
                                             bj["regions"] or None)
        dp = scenarios.opec_cost_push(ic, op["regions"], op["price_shock"] or 1.0, op["include_c19"],
                                      op["rest_of_world_weight"])
        A, L = indicators.leontief(ic)
        v = indicators.va_coefficients(ic)
        yw = ic.Y.sum(axis=1)
        origins = [make_label(r, "B06") for r in op["regions"]]
        for c in ic.countries:
            h = make_label(c, "H51")
            row = {"theta": th, "country": c, "dp_H51_per_unit_shock": dp[h],
                   "ptf_opecB06_through_H51": indicators.pass_through_frequency(L, v, yw, origins, [h])}
            if e2 is not None:
                f = (e2.reindex(ic.Z.index).fillna(0) / ic.output_rows()).fillna(0)
                row["ghg_multiplier_H51"] = float(f.values @ L[h].values)
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"scenario_grid_{y}.csv", index=False)
    print(res.groupby("theta").mean(numeric_only=True))


if __name__ == "__main__":
    main()
