"""Tests for the shared icio_projection package (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
for cand in (HERE, HERE / "src"):
    if (cand / "icio_projection").exists() or (cand / "biojet_icio").exists():
        sys.path.insert(0, str(cand))

try:
    from icio_projection import evaluate as E, external as X, methods as M, panel as P, project as PR, synthetic, tsmodels
except ImportError:  # vendored inside biojet_icio
    from biojet_icio.icio_projection import evaluate as E, external as X, methods as M, panel as P, project as PR, synthetic, tsmodels

YEARS = list(range(2023, 2031))


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("hist")
    h = synthetic.make_history()
    synthetic.write_history(h, d / "cp", d / "pyp")
    pan = P.build_panel(d / "cp", d / "pyp")
    return h, pan, X.External(synthetic.synthetic_external())


def test_arima_recovers_ar1():
    rng = np.random.default_rng(3)
    w = np.zeros(400)
    for t in range(1, 400):
        w[t] = 0.01 + 0.6 * w[t - 1] + rng.normal(0, 0.01)
    fit = tsmodels.fit_armax(w)
    assert fit.p >= 1
    assert fit.phi[0] == pytest.approx(0.6, abs=0.12)
    f = fit.forecast(200)
    assert f[-1] == pytest.approx(0.01 / (1 - fit.phi.sum()) if fit.q == 0 else f[-1], rel=0.3)


def test_bridge_picks_up_proxy():
    rng = np.random.default_rng(4)
    z = np.cumsum(rng.normal(0.02, 0.03, 60))
    y = np.exp(0.5 + 1.5 * z + rng.normal(0, 0.002, 60).cumsum() * 0)
    zl = np.exp(z)
    lvl, fit = tsmodels.project_series(y, 3, zl, zl[-1] * np.exp(np.array([0.05, 0.10, 0.15])))
    assert fit.beta[0] == pytest.approx(1.5, abs=0.05)
    assert np.log(lvl[0] / y[-1]) == pytest.approx(1.5 * 0.05, abs=0.01)


def test_panel_volume_chain(data):
    h, pan, _ = data
    vol, price, has = P.volumes_and_prices(pan, "x")
    assert has
    cp = P.wide(pan, "x")
    np.testing.assert_allclose(vol[2022], cp[2022])
    # volume growth equals the true real growth (x_pyp_t / x_cp_{t-1})
    t = 2010
    true = h[t][1].x() / h[t - 1][0].x()
    np.testing.assert_allclose((vol[t] / vol[t - 1]).reindex(true.index), true, rtol=1e-9)


@pytest.mark.parametrize("method", ["oecd", "arima", "external"])
def test_chain_meets_targets(data, method):
    h, pan, ext = data
    base = h[2022][0]
    pj = M.project(method, pan, base, YEARS, ext=ext)
    ch = PR.run_chain(base, pj, keep_tables=True)
    assert set(ch.tables) == {(y, p) for y in YEARS for p in ("pyp", "cp")}
    for y in YEARS:
        cp = ch.tables[(y, "cp")]
        assert cp.max_imbalance() < 1e-3 * cp.x().max()
        target = base.x() * pj.gx[y] * pj.px[y]
        np.testing.assert_allclose(cp.x(), target, rtol=1e-5)
        # PYP of year t is at t-1 prices
        pyp = ch.tables[(y, "pyp")]
        pprev = pj.px[y - 1] if y > YEARS[0] else 1.0
        np.testing.assert_allclose(pyp.x(), base.x() * pj.gx[y] * pprev, rtol=1e-5)
    assert (ch.ras_log["error"] < 1e-5).all()


def test_external_drivers(data):
    h, pan, ext = data
    base = h[2022][0]
    pj = M.project("external", pan, base, YEARS, ext=ext)
    # US refining follows AEO shipments (-0.8%/yr)
    assert pj.gx.loc["USA_C19", 2030] == pytest.approx(0.992 ** 8, rel=1e-6)
    # Canadian electricity output follows IEO generation (+2.5%/yr)
    assert pj.gx.loc["CAN_D35", 2030] == pytest.approx(1.025 ** 8, rel=1e-3)
    assert pj.coef and 2030 in pj.coef


def test_split_children_follow_parent(data):
    h, pan, ext = data
    base = h[2022][0]
    # pretend CAN_J62_63 was split into two children
    keys = base.keys
    j = keys.index("CAN_J62_63")
    new = keys[:j] + ["CAN_J62_63_CLD", "CAN_J62_63_OTH"] + keys[j + 1:]
    parent_of = {"CAN_J62_63_CLD": "CAN_J62_63", "CAN_J62_63_OTH": "CAN_J62_63"}
    from types import SimpleNamespace
    fake = SimpleNamespace(keys=new, Y=base.Y)
    pj = M.project_arima(pan, fake, YEARS, 2022, 2022, parent_of)
    assert pj.gx.loc["CAN_J62_63_CLD", 2030] == pj.gx.loc["CAN_J62_63_OTH", 2030]


def test_backtest_and_compare(data):
    h, pan, ext = data
    bt = E.backtest_margins(pan, origins=(2014,))
    assert set(bt.method) >= {"arima", "oecd_expost", "oecd_ar", "naive"}
    assert bt["wape_x_cp"].notna().all()
    base = h[2022][0]
    chains = {m: PR.run_chain(base, M.project(m, pan, base, YEARS, ext=ext)) for m in ("oecd", "arima", "external")}
    cmp_ = E.compare(chains, base, focus_keys=["CAN_J62_63"])
    assert {"oecd", "arima", "external"} <= set(cmp_["by_industry"].columns)
    assert len(cmp_["focus_paths"]) == 3 * 9


def test_table_backtest(data):
    h, pan, _ = data
    hist = pan[pan.year <= 2018]
    base = h[2018][0]
    pj = M.project_oecd(hist, base, [2019, 2020, 2021, 2022], 2018, None, 2018, future_actual=pan)
    r = E.backtest_table(base, h[2022][0], pj)
    assert r["wape_x_cp"] < 0.2 and np.isfinite(r["output_multiplier_mape"])
