"""Run the disaggregation under each parent mode and compare how much of the
parent row/column had to be clipped and how far RAS moved the untouched block.

    python scripts/compare_parent_modes.py            # real ICIO in data/icio
    python scripts/compare_parent_modes.py --synthetic
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio.config import load_config  # noqa: E402
from biojet_icio.icio import find_icio_file, read_icio  # noqa: E402
from biojet_icio.pipeline import run  # noqa: E402
from biojet_icio.synthetic import make_synthetic  # noqa: E402

NEW = ("_BGS", "_BDS", "_BJT", "_OLB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg0 = load_config()
    icio = make_synthetic() if a.synthetic else read_icio(find_icio_file(cfg0.path("icio_dir"), cfg0.year))
    rows = []
    for mode in ("proposal", "isic", "table"):
        cfg = load_config()
        cfg.settings["disaggregation"]["parent_mode"] = mode
        r = run(cfg, icio=icio, write=False)
        d = r["diagnostics"].drop_duplicates(["region", "parent"])
        pre, post = r["pre_ras"], r["icio"]
        old = [l for l in post.Z.index if not l.endswith(NEW)]
        dz = np.abs(post.Z.loc[old, old].values - pre.Z.loc[old, old].values).sum()
        rows.append({"mode": mode, "clipped_row_musd": d.parent_row_clipped.sum(),
                     "clipped_col_musd": d.parent_col_clipped.sum(),
                     "ras_abs_change_untouched_musd": dz,
                     "ras_abs_change_pct_of_Z": 100 * dz / pre.Z.values.sum()})
    res = pd.DataFrame(rows)
    out = cfg0.path("output_dir")
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "parent_mode_comparison.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
