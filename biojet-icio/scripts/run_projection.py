"""Project the biofuel-split ICIO 2023-2030 (OECD, ARIMA, external) and compare.

    python scripts/run_projection.py build-panel --cp-dir data/icio [--pyp-dir data/icio_pyp]
    python scripts/run_projection.py fetch-external          # needs EIA_API_KEY for AEO/IEO
    python scripts/run_projection.py run [--parent-mode proposal|isic|table]

``run`` first runs the 2022 pipeline (biofuel split, RAS), then projects the
split table year by year with RAS in previous year's and current prices.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biojet_icio.config import load_config  # noqa: E402
from biojet_icio.icio_projection import external as X, panel as P  # noqa: E402
from biojet_icio.pipeline import run  # noqa: E402
from biojet_icio.projection import run_projection  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--settings", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-panel")
    b.add_argument("--cp-dir", type=Path, required=True, help="OECD ICIO csv files 1995-2022")
    b.add_argument("--pyp-dir", type=Path, default=None, help="OECD PYP ICIO csv files")
    b.add_argument("--pattern", default="*.csv")
    f = sub.add_parser("fetch-external")
    f.add_argument("--ieo-route", default="ieo/2023")
    f.add_argument("--aeo-route", default="aeo/2026")
    r = sub.add_parser("run")
    r.add_argument("--parent-mode", choices=["proposal", "isic", "table"], default=None)
    r.add_argument("--no-backtest", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.settings)
    pc = cfg.settings.get("projection", {})
    if a.cmd == "build-panel":
        pan = P.build_panel(a.cp_dir, a.pyp_dir, pattern=a.pattern,
                            merge_cn_mx=cfg.settings["base"].get("merge_cn_mx", True))
        print(P.save_panel(pan, cfg.root / pc.get("panel", "data/projection/panel")))
    elif a.cmd == "fetch-external":
        print(X.fetch_all(cfg.root / pc.get("external", "data/projection/external.csv"),
                          ieo_route=a.ieo_route, aeo_route=a.aeo_route))
    else:
        if a.parent_mode:
            cfg.settings["disaggregation"]["parent_mode"] = a.parent_mode
        res = run(cfg)
        run_projection(cfg, res["icio"], res, methods=pc.get("methods", ["oecd", "arima", "external"]),
                       backtest_origins=() if a.no_backtest else pc.get("backtest_origins", [2010, 2014, 2017]))


if __name__ == "__main__":
    main()
