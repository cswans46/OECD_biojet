"""Run the full routine:  python scripts/run_pipeline.py [--parent-mode proposal|isic|table]"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biojet_icio.config import load_config  # noqa: E402
from biojet_icio.pipeline import run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", default=None, help="alternative settings.toml")
    ap.add_argument("--parent-mode", choices=["proposal", "isic", "table"], default=None)
    ap.add_argument("--crops", action="store_true", help="also split A01 into feedstock crops")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.settings)
    if a.parent_mode:
        cfg.settings["disaggregation"]["parent_mode"] = a.parent_mode
    if a.crops:
        cfg.settings["crops"]["enabled"] = True
    run(cfg)


if __name__ == "__main__":
    main()
