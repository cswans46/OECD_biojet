"""Projection 2023-2030 of a biofuel-split table (synthetic, no config csvs)."""
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio.config import Config  # noqa: E402
from biojet_icio.icio import ICIO, make_label  # noqa: E402
from biojet_icio.icio_projection import external as X, panel as P, synthetic as S  # noqa: E402
from biojet_icio.projection import YEARS, biojet_driver, parent_map, run_projection  # noqa: E402
from biojet_icio.synthetic import DEFAULT_REGIONS, SECTORS, make_synthetic  # noqa: E402

REGIONS = ["USA", "BRA", "CAN", "ROW"]


def carve(icio: ICIO, region: str, code: str = "BJT", parent: str = "C19", share: float = 0.1) -> ICIO:
    """Proportional split of parent into parent + code (keeps the table balanced)."""
    keys = list(icio.Z.index)
    p = keys.index(make_label(region, parent))
    new = keys[:p + 1] + [make_label(region, code)] + keys[p + 1:]
    Smat = np.zeros((len(new), len(keys)))
    for i, k in enumerate(keys):
        Smat[new.index(k), i] = 1.0
    Smat[p, p] = 1 - share
    Smat[p + 1, p] = share
    Z = Smat @ icio.Z.to_numpy() @ Smat.T
    Y = Smat @ icio.Y.to_numpy()
    Wz = icio.W[keys].to_numpy() @ Smat.T
    W = pd.DataFrame(np.hstack([Wz, icio.W[list(icio.Y.columns)].to_numpy()]), index=icio.W.index,
                     columns=new + list(icio.Y.columns))
    return ICIO(pd.DataFrame(Z, new, new), pd.DataFrame(Y, new, icio.Y.columns), W)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("bj")
    hist = S.make_history(countries=REGIONS, sectors=SECTORS, seed=5)
    S.write_history(hist, d / "cp", d / "pyp")
    pan = P.build_panel(d / "cp", d / "pyp")
    base = make_synthetic({r: DEFAULT_REGIONS[r] for r in REGIONS}, seed=1)
    for r in ("USA", "BRA"):
        base = carve(base, r)
    ext = X.External(S.synthetic_external(countries=REGIONS))
    with open(ROOT / "config" / "settings.toml", "rb") as fh:
        settings = tomllib.load(fh)
    cfg = Config(settings=settings, root=d)
    return base, pan, ext, cfg, d


def test_parent_map(setup):
    base, *_ = setup
    pm = parent_map(base)
    assert pm == {"USA_BJT": "USA_C19", "BRA_BJT": "BRA_C19"}


def test_run_projection(setup):
    base, pan, ext, cfg, d = setup
    cmp_ = run_projection(cfg, base, panel=pan, external=ext, out_dir=d / "out", backtest_origins=(2014,))
    by = cmp_["by_industry"]
    assert {"oecd", "arima", "external"} <= set(by.columns)
    assert {"USA_BJT", "BRA_BJT"} <= set(by.index)
    for m in ("oecd", "arima", "external"):
        ras = pd.read_csv(d / "out" / f"ras_log_{m}.csv")
        assert (ras["error"] < 1e-4).all()
        assert (d / "out" / m / "2030_cp.npz").exists()
    assert (d / "out" / "backtest" / "backtest_by_horizon.csv").exists()


def test_biojet_driver_raises_bjt_and_cuts_c19(setup):
    base, pan, ext, cfg, d = setup
    from biojet_icio.icio_projection import methods as M
    from biojet_icio.icio_projection.table import Table
    pm = parent_map(base)
    p0 = M.project("arima", pan, Table.from_icio(base), YEARS, parent_of=pm)
    g_bjt, g_c19 = p0.gx.loc["USA_BJT", 2030], p0.gx.loc["USA_C19", 2030]
    p1 = biojet_driver(base, p0, theta_2030=0.30, price_ratio=2.5)
    assert p1.gx.loc["USA_BJT", 2030] > g_bjt
    assert p1.gx.loc["USA_C19", 2030] < g_c19
    # at 2030 the biojet share of H51 jet spending (volume, 2022 prices) reaches theta
    x0 = base.output_rows()
    h51_fuel = base.Z.loc[[f"{c}_C19" for c in REGIONS] + ["USA_BJT", "BRA_BJT"], "USA_H51"].sum()
    added = (0.30 - base.Z.loc[["USA_BJT", "BRA_BJT"], "USA_H51"].sum() / h51_fuel) * h51_fuel * p0.gx.loc["USA_H51", 2030]
    assert x0["USA_BJT"] * p1.gx.loc["USA_BJT", 2030] == pytest.approx(x0["USA_BJT"] * g_bjt + 2.5 * added)


def test_inputs_driver_and_hold_excluded(setup):
    base, pan, ext, cfg, d = setup
    from biojet_icio.icio_projection import methods as M
    from biojet_icio.icio_projection.table import Table
    from biojet_icio.projection import hold_excluded, inputs_driver
    pm = parent_map(base)
    pr = M.project("external", pan, Table.from_icio(base), YEARS, ext=ext, parent_of=pm)
    growth = {"output_index": pd.DataFrame([{"year": y, "country": "USA", "sector": "BJT",
                                              "output_index": 1 + 0.5 * (y - 2022)} for y in YEARS])}
    pr = inputs_driver(base, pr, growth, ["BGS", "BDS", "BJT", "OLB"])
    assert pr.gx.loc["USA_BJT", 2030] == pytest.approx(5.0)
    pr = hold_excluded(base, pr, ["CAN"])
    can = [k for k in pr.gx.index if k.startswith("CAN_")]
    assert (pr.gx.loc[can] == 1.0).all().all() and (pr.px.loc[can] == 1.0).all().all()
    can_g = {"output_index": pd.DataFrame([{"year": 2030, "country": "CAN", "sector": "H51", "output_index": 1.3}])}
    pr = hold_excluded(base, pr, ["CAN"], can_g)
    assert pr.gx.loc["CAN_H51", 2030] == pytest.approx(1.3)


def test_run_projection_uses_inputs_and_excludes(setup, tmp_path):
    base, pan, ext, cfg, d = setup
    import shutil
    from biojet_icio.config import Config
    root = tmp_path / "root"
    (root / "config").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "projections.toml", root / "config" / "projections.toml")
    (root / "data" / "projections").mkdir(parents=True)
    pd.DataFrame([{"year": y, "country": "USA", "sector": "BJT", "output_index": 2.0, "output_level_musd": np.nan}
                  for y in YEARS]).to_csv(root / "data" / "projections" / "output_index.csv", index=False)
    cfg2 = Config(settings=cfg.settings, root=root)
    cmp_ = run_projection(cfg2, base, panel=pan, external=ext, out_dir=tmp_path / "o", methods=("arima",),
                          backtest_origins=(), write_tables=False)
    by = cmp_["by_industry"]
    assert by.loc["CAN_H51", "arima"] == pytest.approx(by.loc["CAN_H51", "x_2022"], rel=1e-6)
    assert "output_index.csv" in cmp_["notes"].iloc[0]["note"]
