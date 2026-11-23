"""Medium-term (2026-2030) system cost of the Strait of Hormuz shock, with and
without faster biojet uptake.

Two channels are evaluated on each year's (projected) ICIO:

1. **Price channel**: Leontief cost-push model. The Brent increase over the
   pre-war outlook (World Bank CMO Oct-2025 vs Oct-2026) is an exogenous price
   change for B06 everywhere (world market), blended with the LNG shock. Extra
   refining margins (C19) and fertiliser (C20) enter as unit cost-push terms.
   Cost to each country's final demand at fixed quantities is
   ``sum_i dp_i * Y_ic``. Oil exporters' windfall (dp_B06 * x_B06) is reported
   separately, so global gross and net costs can both be shown.

2. **Quantity channel**: supply-side inoperability. This is a static,
   per-year version of the inoperability input-output model (Haimes & Jiang
   2001; Santos & Haimes 2004), with critical-input logic for energy chains.
   Gulf B06/C19/C20 lose a share q0 of output. Buyers lose availability of a
   product in proportion to the share of their purchases from inoperable
   origins. Critical inputs pass that loss one-for-one (crude -> refiners;
   jet fuel and biojet -> airlines; fertiliser -> crops); other inputs pass it
   in proportion to their cost share. The propagated part is scaled by
   (1 - substitution) and, in the first round, by (1 - inventory damping),
   then iterated to a fixed point. Loss = sum_j VA_j * q_j.

Biojet enters through H51's critical "jet fuel" bundle {C19, BJT}. A higher
BJT share lowers H51's exposure to the C19 loss and to the C19 price shock,
but BJT carries its own exposure (hydrogen from gas, fertiliser for crops,
C10T12 inputs), which is computed rather than assumed to be zero. The biojet
premium (price ratio x shocked prices) is charged every year, so the result is
a net system-cost comparison.
"""
from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .icio import ICIO, make_label, split_label
from .indicators import leontief

log = logging.getLogger(__name__)

CRITICAL = {            # buyer sector -> product group whose availability binds output
    "C19": ["B06"],
    "D": ["B06"],
    "C20": ["B06"],
    "H49": ["C19", "BJT", "BDS", "BGS"],
    "H50": ["C19", "BDS"],
    "H51": ["C19", "BJT"],
    "A01": ["C20"],
    "BJT": ["C10T12", "E", "C20"],
    "BDS": ["C10T12", "C20"],
}


def load(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _yr(d: dict, y: int, default=0.0):
    return float(d.get(str(y), d.get(y, default)))


# ------------------------------------------------------------------- price channel
def cost_push_general(A: np.ndarray, exo: dict[int, float], dv: np.ndarray) -> np.ndarray:
    """Delta p with exogenous price changes on ``exo`` and unit cost-push ``dv``
    on the remaining (endogenous) sectors: dp_N = (dp_S A_SN + dv_N)(I - A_NN)^-1."""
    n = A.shape[0]
    S = np.array(sorted(exo), dtype=int)
    N = np.setdiff1d(np.arange(n), S)
    dps = np.array([exo[i] for i in S])
    rhs = dps @ A[np.ix_(S, N)] + dv[N]
    dpn = np.linalg.solve((np.eye(len(N)) - A[np.ix_(N, N)]).T, rhs)
    dp = np.zeros(n)
    dp[S], dp[N] = dps, dpn
    return dp


def price_channel(icio: ICIO, h: dict, year: int, A: np.ndarray | None = None) -> dict:
    pr = h["prices"]
    rho = _yr(pr["brent_shock"], year) / _yr(pr["brent_prewar"], year) - 1
    rho_g = _yr(pr["lng_price_shock"], year)
    labels = list(icio.Z.index)
    if A is None:
        x = icio.output_rows().values
        A = icio.Z.values / np.where(x > 0, x, 1)[None, :]
    exo = {i: 0.8 * rho + 0.2 * rho_g for i, l in enumerate(labels) if split_label(l)[1] == "B06"}
    dv = np.zeros(len(labels))
    m = _yr(pr["c19_margin_shock"], year)
    f = _yr(pr["fertiliser_price_shock"], year)
    for i, l in enumerate(labels):
        sec = split_label(l)[1]
        if sec == "C19":
            dv[i] = m
        elif sec == "C20":
            dv[i] = 0.08 * f            # fertilisers ~8% of C20 output (assumption)
    dp = cost_push_general(A, exo, dv)
    dp = pd.Series(dp, index=labels)
    Yc = icio.Y.T.groupby(lambda c: c[:3]).sum().T          # rows x destination country
    consumer = (Yc.mul(dp, axis=0)).sum()                    # USD m by country
    x = icio.output_rows()
    windfall = (dp * x)[[l for l in labels if split_label(l)[1] == "B06"]]
    windfall = windfall.groupby(lambda l: l[:3]).sum()
    return {"dp": dp, "consumer_cost": consumer, "b06_windfall": windfall}


# ---------------------------------------------------------------- quantity channel
@dataclass
class Inoperability:
    q: pd.Series
    va_loss: pd.Series      # by country
    out_loss: pd.Series     # by country


def quantity_channel(icio: ICIO, h: dict, year: int, max_iter: int = 200, tol: float = 1e-10) -> Inoperability:
    qd = h["quantities"]
    labels = list(icio.Z.index)
    n = len(labels)
    pos = {l: i for i, l in enumerate(labels)}
    q0 = np.zeros(n)
    for r in qd["regions"]:
        for sec, key in (("B06", "b06_supply_loss"), ("C19", "c19_supply_loss"), ("C20", "c20_supply_loss")):
            l = make_label(r, sec)
            if l in pos:
                q0[pos[l]] = _yr(qd[key], year)
    rl = make_label("ROW", "B06")
    if rl in pos:
        q0[pos[rl]] = _yr(qd["b06_supply_loss"], year) * qd["row_gulf_share_of_row_b06"]
    sub = _yr(qd["substitution"], year, 1.0)
    damp = float(qd.get("inventory_damping", 0.0))
    if q0.max() == 0:
        z = pd.Series(0.0, index=labels)
        return Inoperability(z, z.groupby(lambda l: l[:3]).sum(), z.groupby(lambda l: l[:3]).sum())

    Z = np.clip(icio.Z.values, 0, None)
    prod = np.array([split_label(l)[1] for l in labels])
    colsum = Z.sum(axis=0)
    # critical-input mask: crit[i, j] = 1 if row i's product binds buyer j's output
    crit = np.zeros_like(Z, dtype=bool)
    for j, b in enumerate(prod):
        grp = CRITICAL.get(b)
        if grp:
            crit[:, j] = np.isin(prod, grp)
    Zc = np.where(crit, Z, 0.0)
    Zn = Z - Zc
    den_c = Zc.sum(axis=0)
    xo = icio.output_rows().values
    Wn = Zn / np.where(xo > 0, xo, 1)[None, :]                   # cost-share (per unit output) pass-through
    Wc = Zc / np.where(den_c > 0, den_c, 1)[None, :]             # bundle availability (one-for-one)
    W = Wn + Wc
    q = q0.copy()
    for it in range(max_iter):
        prop = q @ W
        factor = (1 - sub) * (1 - damp if it == 0 else 1.0)
        new = np.clip(np.maximum(q0, q0 + factor * prop), 0, 1)
        if np.abs(new - q).max() < tol:
            q = new
            break
        q = new
    qs = pd.Series(q, index=labels)
    va = icio.W.loc["VA", labels]
    x = icio.output_rows()
    return Inoperability(qs, (va * qs).groupby(lambda l: l[:3]).sum(), (x * qs).groupby(lambda l: l[:3]).sum())


# --------------------------------------------------------------- biojet adjustment
def h51_bio_share(icio: ICIO, country: str) -> float:
    h = make_label(country, "H51")
    c19 = icio.Z.loc[[make_label(c, "C19") for c in icio.countries], h].sum()
    bjt = icio.Z.loc[[make_label(c, "BJT") for c in icio.countries], h].sum() if \
        make_label(icio.countries[0], "BJT") in icio.Z.index else 0.0
    return float(bjt / (bjt + c19)) if (bjt + c19) > 0 else 0.0


def set_biojet_share(icio: ICIO, target: float, price_ratio: float) -> tuple[ICIO, float]:
    """Raise every region's H51 BJT share (value basis on energy-equivalent volumes) to
    ``target``. Returns the new table and the annual extra fuel cost (USD m) at
    pre-shock prices."""
    from .scenarios import substitute_biojet
    th = {}
    for c in icio.countries:
        s0 = h51_bio_share(icio, c)
        # energy share of BJT implied by value share s0 and the price ratio
        e0 = s0 / price_ratio / (s0 / price_ratio + (1 - s0)) if s0 > 0 else 0.0
        if target > e0:
            th[c] = (target - e0) / (1 - e0)
    new, _ = substitute_biojet(icio, th, price_ratio)
    c19_rows = [make_label(c, "C19") for c in icio.countries]
    h51 = [make_label(c, "H51") for c in icio.countries]
    moved = icio.Z.loc[c19_rows, h51].sum() - new.Z.loc[c19_rows, h51].sum()
    return new, float(moved.sum() * (price_ratio - 1))


# --------------------------------------------------------------------- driver
def system_cost(tables: dict[int, ICIO], h: dict, case: str = "baseline",
                discount: float = 0.03, base_year: int = 2026) -> dict[str, pd.DataFrame]:
    """Run both channels for each year under a biojet case; returns yearly and total tables."""
    theta = h["biojet_cases"][case]
    ratio = float(h["biojet_cases"]["price_ratio_prewar"])
    rows, by_country = [], []
    for y, icio in sorted(tables.items()):
        if str(y) not in theta and y not in theta:
            continue
        has_bjt = make_label(icio.countries[0], "BJT") in icio.Z.index
        prem0 = 0.0
        tab = icio
        if has_bjt:
            tab, prem0 = set_biojet_share(icio, _yr(theta, y), ratio)
        x = tab.output_rows().values
        A = tab.Z.values / np.where(x > 0, x, 1)[None, :]
        pc = price_channel(tab, h, y, A)
        qc = quantity_channel(tab, h, y)
        # biojet premium relative to an all-fossil bundle, without and with the shock
        prem_ns = prem_s = 0.0
        h51 = [make_label(c, "H51") for c in tab.countries]
        if has_bjt:
            bj = [make_label(c, "BJT") for c in tab.countries]
            c19 = [make_label(c, "C19") for c in tab.countries]
            Zb, Zf = tab.Z.loc[bj, h51], tab.Z.loc[c19, h51]
            v_bjt = Zb.values.sum(axis=0)
            dpv_bjt = Zb.mul(pc["dp"][bj].values, axis=0).values.sum(axis=0)
            fs = Zf.values.sum(axis=0)
            dp_c19 = np.divide(Zf.mul(pc["dp"][c19].values, axis=0).values.sum(axis=0), fs,
                               out=np.zeros(len(h51)), where=fs > 0)
            fossil_equiv = v_bjt / ratio
            prem_ns = float((v_bjt - fossil_equiv).sum())
            prem_s = float((v_bjt + dpv_bjt - fossil_equiv * (1 + dp_c19)).sum())
        xs = tab.output_rows()
        h51_price = float((pc["dp"][h51] * xs[h51]).sum())
        h51_va = float((qc.q[h51] * tab.W.loc["VA", h51]).sum())
        df = (1 + discount) ** -(y - base_year)
        shock = float(pc["consumer_cost"].sum() - pc["b06_windfall"].sum() + qc.va_loss.sum() + prem_s - prem_ns)
        rows.append({"year": y, "case": case, "biojet_share": _yr(theta, y),
                     "consumer_cost_musd": float(pc["consumer_cost"].sum()),
                     "b06_windfall_musd": float(pc["b06_windfall"].sum()),
                     "net_price_cost_musd": float(pc["consumer_cost"].sum() - pc["b06_windfall"].sum()),
                     "va_loss_musd": float(qc.va_loss.sum()),
                     "premium_shock_effect_musd": prem_s - prem_ns,
                     "shock_cost_musd": shock,
                     "biojet_policy_cost_musd": prem_ns,
                     "h51_price_cost_musd": h51_price, "h51_va_loss_musd": h51_va,
                     "dp_H51_mean": float(pc["dp"][h51].mean()),
                     "q_H51_mean": float(qc.q[h51].mean()),
                     "bjt_value_musd": float(v_bjt.sum()) if has_bjt else 0.0,
                     "discount_factor": df})
        cc = pd.DataFrame({"consumer_cost_musd": pc["consumer_cost"], "va_loss_musd": qc.va_loss})
        cc["year"], cc["case"] = y, case
        by_country.append(cc.rename_axis("country").reset_index())
    yr = pd.DataFrame(rows)
    yr["system_cost_musd"] = yr.shock_cost_musd + yr.biojet_policy_cost_musd
    yr["shock_cost_pv_musd"] = yr.shock_cost_musd * yr.discount_factor
    yr["system_cost_pv_musd"] = yr.system_cost_musd * yr.discount_factor
    return {"by_year": yr, "by_country": pd.concat(by_country, ignore_index=True) if by_country else pd.DataFrame()}


def compare_cases(tables: dict[int, ICIO], h: dict, cases=("baseline", "accelerated")) -> dict[str, pd.DataFrame]:
    res = {c: system_cost(tables, h, c) for c in cases}
    cols = ["consumer_cost_musd", "b06_windfall_musd", "net_price_cost_musd", "va_loss_musd",
            "premium_shock_effect_musd", "shock_cost_musd", "biojet_policy_cost_musd", "system_cost_musd",
            "shock_cost_pv_musd", "system_cost_pv_musd", "h51_price_cost_musd", "h51_va_loss_musd",
            "bjt_value_musd"]
    tot = pd.DataFrame({c: r["by_year"][cols].sum() for c, r in res.items()})
    if len(cases) == 2:
        a, b = cases
        tot["difference"] = tot[b] - tot[a]
        avoided = tot.loc["shock_cost_musd", a] - tot.loc["shock_cost_musd", b]
        extra_policy = tot.loc["biojet_policy_cost_musd", b] - tot.loc["biojet_policy_cost_musd", a]
        extra_bjt = tot.loc["bjt_value_musd", b] - tot.loc["bjt_value_musd", a]
        r = float(h["biojet_cases"]["price_ratio_prewar"])
        # premium is (1 - 1/ratio) x BJT value: breakeven when avoided shock cost equals it
        frac = avoided / extra_bjt if extra_bjt > 0 else np.nan
        tot.loc["avoided_shock_cost_musd", "difference"] = avoided
        tot.loc["extra_policy_cost_musd", "difference"] = extra_policy
        tot.loc["net_benefit_of_faster_biojet_musd", "difference"] = avoided - extra_policy
        tot.loc["breakeven_price_ratio", "difference"] = 1 / (1 - frac) if frac < 1 else np.inf
    return {"totals": tot, "by_year": pd.concat([r["by_year"] for r in res.values()]),
            "by_country": pd.concat([r["by_country"] for r in res.values()])}


def shifted_shock(h: dict, start: int = 2029, years=range(2026, 2031)) -> dict:
    """Copy of the shock spec with the 2026-27 profile moved to ``start`` (a repeat closure).
    Biojet shares are unchanged, so faster uptake meets the repeat shock at a higher share."""
    import copy
    g = copy.deepcopy(h)
    lag = start - 2026

    def shift(d, fill):
        return {str(y): float(d.get(str(y - lag), fill)) if y - lag >= 2026 else fill for y in years}
    pr = g["prices"]
    pr["brent_shock"] = {str(y): (_yr(h["prices"]["brent_shock"], y - lag) / _yr(h["prices"]["brent_prewar"], y - lag)
                                  * _yr(h["prices"]["brent_prewar"], y)) if y - lag >= 2026 else _yr(h["prices"]["brent_prewar"], y)
                         for y in years}
    for k in ("c19_margin_shock", "lng_price_shock", "fertiliser_price_shock"):
        pr[k] = shift(h["prices"][k], 0.0)
    q = g["quantities"]
    for k in ("b06_supply_loss", "c19_supply_loss", "c20_supply_loss"):
        q[k] = shift(h["quantities"][k], 0.0)
    q["substitution"] = shift(h["quantities"]["substitution"], 1.0)
    return g
