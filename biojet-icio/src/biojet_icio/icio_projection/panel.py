"""Historical margins panel from the OECD ICIO time series (1995-2022).

Reading 28 full tables for every model run is slow, so ``build_panel`` reads
each yearly table once and keeps only the margins the projections need:

    x[key]        gross output (row total)                 per industry
    va[key]       value added                              per industry
    tls[key]      taxes less subsidies on intermediates    per industry
    fd[c_k]       final demand column totals               per FD column
    imp[c]        imports (intermediate + final)           per country
    exp[c]        exports (intermediate + final)           per country
    gdp[c]        VA + TLS on industries + TLS on FD       per country

for current prices ("cp") and, when available, previous year's prices
("pyp"). From both it derives chain-linked volume indices and price
indices (2022 = 1), the series the ARIMA and bridge regressions model.

Without PYP tables the panel is nominal only: volumes then equal current
values and prices are flat. ``--pyp-dir`` (or download_pyp) fixes that.
"""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .table import Table, country_of, read_wide

log = logging.getLogger(__name__)

VARS = ("x", "va", "tls", "fd", "imp", "exp", "gdp")


def margins(t: Table) -> dict[str, pd.Series]:
    keys = t.keys
    ctry = np.array([country_of(k) for k in keys])
    zc = np.array([country_of(c) for c in t.Z.columns])
    yc = np.array([country_of(c) for c in t.Y.columns])
    Z, Y = t.Z.to_numpy(), t.Y.to_numpy()
    foreign_z = ctry[:, None] != zc[None, :]
    foreign_y = ctry[:, None] != yc[None, :]
    exp_i = (Z * foreign_z).sum(axis=1) + (Y * foreign_y).sum(axis=1)
    countries = list(dict.fromkeys(ctry))
    imp = {}
    for c in countries:
        cz, cy = zc == c, yc == c
        home = ctry == c
        imp[c] = Z[np.ix_(~home, cz)].sum() + Y[np.ix_(~home, cy)].sum()
    va = t.W.loc["VA", keys] if "VA" in t.W.index else pd.Series(0.0, index=keys)
    tls = t.W.loc["TLS", keys] if "TLS" in t.W.index else pd.Series(0.0, index=keys)
    tls_fd = t.W.loc["TLS", t.Y.columns] if "TLS" in t.W.index else pd.Series(0.0, index=t.Y.columns)
    gdp = (va + tls).groupby(ctry).sum().add(tls_fd.groupby(yc).sum(), fill_value=0.0)
    return {
        "x": t.x(),
        "va": va.astype(float),
        "tls": tls.astype(float),
        "fd": t.Y.sum(axis=0),
        "imp": pd.Series(imp),
        "exp": pd.Series(exp_i, index=keys).groupby(ctry).sum(),
        "gdp": gdp,
    }


def _year_of(path: Path) -> int | None:
    m = re.search(r"(19|20)\d{2}", path.stem)
    return int(m.group(0)) if m else None


def table_files(folder: Path, pattern: str = "*.csv") -> dict[int, Path]:
    out = {}
    for p in sorted(Path(folder).glob(pattern)):
        y = _year_of(p)
        if y is not None:
            out[y] = p
    return out


def build_panel(cp_dir: Path, pyp_dir: Path | None = None, years=range(1995, 2023),
                pattern: str = "*.csv", merge_cn_mx: bool = True, relabel=None) -> pd.DataFrame:
    """Long panel: columns year, prices ('cp'|'pyp'), var, key, value."""
    rows = []
    for prices, folder in (("cp", cp_dir), ("pyp", pyp_dir)):
        if folder is None:
            continue
        files = table_files(Path(folder), pattern)
        for y in years:
            if y not in files:
                if prices == "cp":
                    raise FileNotFoundError(f"No current-price ICIO table for {y} in {folder}")
                log.warning("No PYP table for %s in %s", y, folder)
                continue
            t = read_wide(files[y], merge_cn_mx=merge_cn_mx)
            if relabel is not None:
                t = relabel(t)
            for var, s in margins(t).items():
                rows.append(pd.DataFrame({"year": y, "prices": prices, "var": var,
                                          "key": s.index.astype(str), "value": s.to_numpy(float)}))
            log.info("panel: %s %s read", prices, y)
    return pd.concat(rows, ignore_index=True)


def save_panel(panel: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        panel.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception:  # pyarrow missing
        panel.to_csv(path.with_suffix(".csv.gz"), index=False)
        return path.with_suffix(".csv.gz")


def load_panel(path: Path) -> pd.DataFrame:
    path = Path(path)
    for cand in (path, path.with_suffix(".parquet"), path.with_suffix(".csv.gz")):
        if cand.exists():
            return pd.read_parquet(cand) if cand.suffix == ".parquet" else pd.read_csv(cand)
    raise FileNotFoundError(path)


def wide(panel: pd.DataFrame, var: str, prices: str = "cp") -> pd.DataFrame:
    """key x year matrix for one variable."""
    p = panel[(panel["var"] == var) & (panel["prices"] == prices)]
    return p.pivot_table(index="key", columns="year", values="value", aggfunc="sum").sort_index(axis=1)


def volumes_and_prices(panel: pd.DataFrame, var: str, base_year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Chain-linked volume (in base-year prices) and price index (base = 1).

    vol_t / vol_{t-1} = pyp_t / cp_{t-1};  price = cp / vol.
    Returns (volume, price, has_pyp).
    """
    cp = wide(panel, var, "cp")
    has_pyp = bool(((panel["prices"] == "pyp") & (panel["var"] == var)).any())
    if not has_pyp:
        return cp.copy(), pd.DataFrame(1.0, index=cp.index, columns=cp.columns), False
    pyp = wide(panel, var, "pyp").reindex_like(cp)
    years = list(cp.columns)
    link = pd.DataFrame(1.0, index=cp.index, columns=years)
    for a, b in zip(years[:-1], years[1:]):
        prev = cp[a].where(cp[a].abs() > 1e-9)
        link[b] = (pyp[b] / prev).fillna(1.0)
    idx = link.cumprod(axis=1)
    # volume in base-year prices: vol_base = cp_base
    vol = idx.div(idx[base_year], axis=0).mul(cp[base_year], axis=0)
    price = (cp / vol.where(vol.abs() > 1e-9)).fillna(1.0)
    return vol, price, True


def download_tables(urls: list[str], dest: Path) -> list[Path]:  # pragma: no cover - needs internet
    """Download and unzip OECD ICIO / PYP ICIO zip files (pass the links from
    https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html
    and the PYP ICIO page)."""
    import urllib.request
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for url in urls:
        zpath = dest / Path(url.split("?")[0]).name
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, zpath)
        if zipfile.is_zipfile(zpath):
            with zipfile.ZipFile(zpath) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(".csv"):
                        p = dest / Path(n).name
                        p.write_bytes(zf.read(n))
                        out.append(p)
        else:
            out.append(zpath)
    return out
