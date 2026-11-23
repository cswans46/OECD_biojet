"""Build public 2023-2030 drivers (Canada excluded) and project the disaggregated ICIO.

    python scripts/build_projections.py --scenario realised   # post-shock outlook (main 2030 projection)
    python scripts/build_projections.py --scenario prewar     # no-war counterfactual (Hormuz cost base)
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio import projection, projection_inputs  # noqa: E402
from biojet_icio.icio import read_icio, write_icio  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "output" / "ICIO_biofuels_2022.csv"))
    ap.add_argument("--years", type=int, nargs="*", default=list(range(2023, 2031)))
    ap.add_argument("--out", default=None, help="default output/projections/<scenario>")
    ap.add_argument("--scenario", choices=["realised", "prewar"], default="realised")
    ap.add_argument("--inputs-only", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base = read_icio(a.base, merge_cn_mx=False)
    pdir = projection_inputs.write(base, ROOT, scenario=a.scenario)
    projection.write_templates(pdir, base)
    if a.inputs_only:
        return
    growth = projection.load_growth(pdir)
    out = Path(a.out or ROOT / "output" / "projections" / a.scenario)
    for y in a.years:
        t = projection.project_year(base, y, growth)
        write_icio(t, out / f"ICIO_biofuels_{y}.csv")


if __name__ == "__main__":
    main()
