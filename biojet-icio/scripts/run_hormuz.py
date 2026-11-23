"""Hormuz shock: total 2026-2030 system cost, baseline vs accelerated biojet.

    python scripts/run_hormuz.py [--tables output/projections] [--base output/ICIO_biofuels_2022.csv]

Uses projected tables ICIO_biofuels_{year}.csv when present, else the 2022 base for every year.
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio import hormuz  # noqa: E402
from biojet_icio.icio import read_icio  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "output" / "projections" / "prewar"),
                    help="counterfactual (pre-war) projected tables; the shock is applied on top")
    ap.add_argument("--repeat-year", type=int, default=2029, help="also run a repeat closure starting this year (0 = off)")
    ap.add_argument("--base", default=str(ROOT / "output" / "ICIO_biofuels_2022.csv"))
    ap.add_argument("--out", default=str(ROOT / "output" / "hormuz"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    h = hormuz.load(ROOT / "config" / "hormuz.toml")
    years = range(2026, 2031)
    tables = {}
    for y in years:
        f = Path(a.tables) / f"ICIO_biofuels_{y}.csv"
        tables[y] = read_icio(f if f.exists() else a.base, merge_cn_mx=False)
        if not f.exists():
            logging.warning("%d: projected table missing, using 2022 base", y)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {"realised_2026": h}
    if a.repeat_year:
        runs[f"repeat_{a.repeat_year}"] = hormuz.shifted_shock(h, a.repeat_year)
    for name, spec in runs.items():
        res = hormuz.compare_cases(tables, spec)
        for k, v in res.items():
            v.to_csv(out / f"hormuz_{name}_{k}.csv")
        print(f"== {name}")
        print(res["totals"].round(0))


if __name__ == "__main__":
    main()
