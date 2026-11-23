import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biojet_icio import hidden_linkages as hl, hormuz, projection, projection_inputs  # noqa: E402
from biojet_icio.config import load_config  # noqa: E402
from biojet_icio.icio import make_label  # noqa: E402
from biojet_icio.indicators import cost_push, leontief  # noqa: E402
from biojet_icio.pipeline import run  # noqa: E402
from biojet_icio.synthetic import make_synthetic  # noqa: E402

REGS = {"USA": 26e6, "BRA": 2.0e6, "NLD": 1.1e6, "FIN": 0.3e6, "SAU": 1.1e6, "SGP": 0.5e6,
        "CHN": 18e6, "IDN": 1.3e6, "MYS": 0.4e6, "CAN": 2.1e6, "ROW": 40e6}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    cfg = load_config()
    cfg.settings["paths"]["output_dir"] = str(tmp_path_factory.mktemp("b"))
    return run(cfg, icio=make_synthetic(REGS, seed=3), write=False)["icio"]


@pytest.fixture(scope="module")
def growth(base, tmp_path_factory):
    d = tmp_path_factory.mktemp("proj")
    projection_inputs.write(base, ROOT, d, scenario="prewar")
    return projection.load_growth(d)


def test_inputs_exclude_canada(growth):
    assert "CAN" not in set(growth["output_index"].country)
    assert "CAN" not in set(growth["fd_index"].country)


def test_projection_balanced_canada_held(base, growth):
    t = projection.project_year(base, 2030, growth)
    assert t.check_balanced(atol=1e-1)
    x0, x1 = base.output_rows(), t.output_rows()
    can = [l for l in x0.index if l.startswith("CAN_")]
    assert np.allclose(x1[can], x0[can], rtol=1e-6)
    assert x1["SGP_BJT"] > 0 and x0["SGP_BJT"] == 0          # new entrant seeded


def test_cost_push_general_matches_simple(base):
    A, _ = leontief(base)
    lab = list(A.index)
    s = [i for i, l in enumerate(lab) if l.endswith("_B06")]
    a = hormuz.cost_push_general(A.values, {i: 0.3 for i in s}, np.zeros(len(lab)))
    b = cost_push(A, [lab[i] for i in s], 0.3).values
    assert np.allclose(a, b)


def test_zero_shock_zero_cost(base):
    h = hormuz.load(ROOT / "config" / "hormuz.toml")
    q = hormuz.quantity_channel(base, h, 2029)            # 2029 has no supply loss
    assert q.va_loss.sum() == 0
    p = hormuz.price_channel(base, h, 2029)
    assert abs(p["consumer_cost"].sum()) < 1e-6


def test_biojet_lowers_h51_shock_exposure(base):
    h = hormuz.load(ROOT / "config" / "hormuz.toml")
    hi, _ = hormuz.set_biojet_share(base, 0.2, 2.0)
    q0 = hormuz.quantity_channel(base, h, 2026).q
    q1 = hormuz.quantity_channel(hi, h, 2026).q
    h51 = [l for l in q0.index if l.endswith("_H51")]
    assert (q1[h51] <= q0[h51] + 1e-12).all() and (q1[h51] < q0[h51]).any()


def test_lookthrough_at_least_direct(base):
    A, L = leontief(base)
    ex = hl.exposure(base, [make_label(c, "H51") for c in base.countries], ["B06", "C19"], ["SAU"], A, L)
    assert (ex.total_req >= ex.direct_req - 1e-12).all()


def test_feedstock_relinking_balanced_and_rerouted(base):
    shares = pd.DataFrame({"CHN": [0.6], "MYS": [0.4]}, index=["USA"])
    new = hl.apply_feedstock_linkages(base, shares, {"USA": 0.8}, ras=True, mislabel_share=0.3)
    assert new.check_balanced(atol=1e-1)
    col = "USA_BJT"
    before = base.Z.loc[["CHN_C10T12", "MYS_C10T12", "IDN_C10T12"], col].sum()
    after = new.Z.loc[["CHN_C10T12", "MYS_C10T12", "IDN_C10T12"], col].sum()
    assert after > before
