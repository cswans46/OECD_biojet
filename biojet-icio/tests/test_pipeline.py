import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio import environment, indicators, scenarios  # noqa: E402
from biojet_icio.config import load_config  # noqa: E402
from biojet_icio.icio import from_wide, make_label  # noqa: E402
from biojet_icio.pipeline import run  # noqa: E402
from biojet_icio.ras import ras_signed  # noqa: E402
from biojet_icio.synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return make_synthetic(seed=1)


@pytest.fixture(scope="module", params=["proposal", "table"])
def result(request, base, tmp_path_factory):
    cfg = load_config()
    cfg.settings["paths"]["output_dir"] = str(tmp_path_factory.mktemp(request.param))
    cfg.settings["disaggregation"]["parent_mode"] = request.param
    return run(cfg, icio=base, write=True), cfg


def test_synthetic_is_balanced(base):
    assert base.check_balanced()


def test_roundtrip(base):
    again = from_wide(base.to_wide())
    assert np.allclose(again.Z.values, base.Z.values)
    assert np.allclose(again.Y.values, base.Y.values)


def test_ras_keeps_negatives():
    seed = np.array([[1.0, -0.5, 2.0], [3.0, 1.0, 0.5]])
    out, info = ras_signed(seed, [4.0, 5.0], [4.0, 1.0, 4.0])
    assert out[0, 1] == -0.5
    assert np.allclose(out.sum(axis=1), [4.0, 5.0])
    assert np.allclose(out.sum(axis=0), [4.0, 1.0, 4.0])


def test_balanced_after_split(result):
    r, _ = result
    assert r["icio"].check_balanced(atol=1e-2)


def test_output_conservation(result, base):
    r, cfg = result
    new = r["icio"]
    x0, x1 = base.output_rows(), new.output_rows()
    assert np.isclose(x0.sum(), x1.sum(), rtol=1e-9)
    for c in base.countries:
        fam = [make_label(c, k) for k in ["C19", "C20", "BGS", "BDS", "BJT", "OLB"]]
        old = [make_label(c, k) for k in ["C19", "C20"]]
        assert np.isclose(x1[fam].sum(), x0[old].sum(), rtol=1e-8), c


def test_new_sector_outputs_hit_targets(result):
    r, _ = result
    x1 = r["icio"].output_rows()
    rep = r["biofuel_report"]
    for _, row in rep.iterrows():
        assert np.isclose(x1[make_label(row.region, row.biofuel)], row.total_output_musd, rtol=1e-6)


def test_va_and_fd_preserved(result, base):
    r, _ = result
    new = r["icio"]
    assert np.isclose(new.W.loc["VA"].sum(), base.W.loc["VA"].sum())
    assert np.allclose(new.Y.sum().values, base.Y.sum().values, atol=1e-3)


def test_biojet_sold_to_air_transport(result):
    r, _ = result
    new = r["icio"]
    rows = [l for l in new.Z.index if l.endswith("_BJT")]
    h51 = [c for c in new.Z.columns if c.endswith("_H51")]
    total = new.Z.loc[rows].values.sum() + new.Y.loc[rows].values.sum()
    to_air = new.Z.loc[rows, h51].values.sum()
    # by-products (10%) go elsewhere; RAS may move a little more
    assert to_air / total > 0.7


def test_indicators_identities(result):
    r, _ = result
    new = r["icio"]
    e0 = pd.Series(new.output_rows().values * 0.1, index=new.Z.index)
    _, L = indicators.leontief(new)
    g = indicators.ghg_footprints(new, e0, L)
    assert np.isclose(g["summary"].production_based.sum(), g["summary"].consumption_based.sum(), rtol=1e-6)
    t = indicators.tiva(new, L)
    assert np.isclose(t["va_fd"].values.sum(), new.W.loc["VA", new.Z.columns].sum(), rtol=1e-6)


def test_emission_split_conserves(result, base):
    r, _ = result
    e0 = pd.Series(base.output_rows().values * 0.2, index=base.Z.index)
    parent_of = {}
    for _, row in r["biofuel_report"].iterrows():
        parent_of[f"{row.biofuel}@{row.region}"] = row.parent
    e1 = environment.split_emissions(e0, r["icio"], parent_of)
    assert np.isclose(e1.sum(), e0.sum())


def test_biojet_substitution_cuts_h51_emissions(result):
    r, _ = result
    new = r["icio"]
    e = pd.Series(1.0, index=new.Z.index)
    out, e2 = scenarios.substitute_biojet(new, 0.1, 2.5, e)
    assert all(e2[l] < 1.0 for l in e2.index if l.endswith("_H51"))
    shock = scenarios.opec_cost_push(new, ["SAU"], 0.5)
    assert shock["USA_H51"] > 0
