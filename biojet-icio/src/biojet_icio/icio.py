"""Read, hold and write OECD Inter-Country Input-Output (ICIO) tables.

The 2025 edition is distributed as one wide csv per year (e.g. ``2022.csv``).
The first column holds row labels; industry rows/columns are labelled
``<CTRY>_<IND>`` (``USA_C19``), final-demand columns ``<CTRY>_<FD>``
(``USA_HFCE``, ``USA_NPISH``, ``USA_GGFC``, ``USA_GFCF``, ``USA_INVNT``,
``USA_DPABR``), and the accounting rows/column are ``TLS`` (taxes less
subsidies on products), ``VA`` (value added) and ``OUT`` (gross output).
The extended version splits China and Mexico into CN1/CN2 and MX1/MX2.

Values are USD million at basic prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ACCOUNT_ROWS = ("TLS", "VA")
OUT = "OUT"
MERGE_PREFIX = {"CN1": "CHN", "CN2": "CHN", "MX1": "MEX", "MX2": "MEX"}


def split_label(label: str) -> tuple[str, str]:
    """``'USA_C17_18' -> ('USA', 'C17_18')``. Country codes are 3 characters."""
    return label[:3], label[4:]


def make_label(country: str, code: str) -> str:
    return f"{country}_{code}"


@dataclass
class ICIO:
    Z: pd.DataFrame        # intermediate use: industry rows x industry cols
    Y: pd.DataFrame        # final demand: industry rows x FD cols
    W: pd.DataFrame        # TLS and VA rows x (industry cols + FD cols)

    # ------------------------------------------------------------------ basics
    @property
    def industries(self) -> list[str]:
        return list(self.Z.index)

    @property
    def countries(self) -> list[str]:
        seen, out = set(), []
        for lab in self.Z.index:
            c = lab[:3]
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def sectors_of(self, country: str) -> list[str]:
        return [split_label(l)[1] for l in self.Z.index if l[:3] == country]

    @property
    def fd_codes(self) -> list[str]:
        seen, out = set(), []
        for lab in self.Y.columns:
            c = split_label(lab)[1]
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def output_rows(self) -> pd.Series:
        """Gross output from the use side: Z row sums + Y row sums."""
        return self.Z.sum(axis=1) + self.Y.sum(axis=1)

    def output_cols(self) -> pd.Series:
        """Gross output from the cost side: Z col sums + TLS + VA."""
        return self.Z.sum(axis=0) + self.W[self.Z.columns].sum(axis=0)

    @property
    def x(self) -> pd.Series:
        return self.output_rows()

    def copy(self) -> "ICIO":
        return ICIO(self.Z.copy(), self.Y.copy(), self.W.copy())

    def balance_report(self) -> pd.DataFrame:
        r, c = self.output_rows(), self.output_cols()
        diff = (r - c)
        return pd.DataFrame({"x_rows": r, "x_cols": c, "diff": diff,
                             "rel": diff / r.where(r.abs() > 1e-9)})

    def check_balanced(self, atol: float = 1e-3, rtol: float = 1e-6) -> bool:
        rep = self.balance_report()
        return bool(np.allclose(rep.x_rows, rep.x_cols, atol=atol, rtol=rtol))

    # ------------------------------------------------------------ aggregation
    def to_wide(self) -> pd.DataFrame:
        """Rebuild the OECD wide layout (rows: industries + TLS, VA, OUT)."""
        top = pd.concat([self.Z, self.Y], axis=1)
        bottom = self.W.reindex(columns=top.columns).fillna(0.0)
        wide = pd.concat([top, bottom]).copy()
        out_col = pd.Series(0.0, index=wide.index)
        out_col.loc[self.Z.index] = self.output_rows()
        wide[OUT] = out_col
        out_row = pd.Series(0.0, index=wide.columns)
        out_row.loc[self.Z.columns] = self.output_cols()
        wide.loc[OUT] = out_row
        return wide


# --------------------------------------------------------------------------- IO
def _merge_split_countries(df: pd.DataFrame, axis: int) -> pd.DataFrame:
    labels = df.index if axis == 0 else df.columns

    def fix(lab: str) -> str:
        pre = lab[:3]
        return MERGE_PREFIX[pre] + lab[3:] if pre in MERGE_PREFIX else lab

    new = [fix(l) for l in labels]
    if new == list(labels):
        return df
    order = list(dict.fromkeys(new))
    if axis == 0:
        out = df.groupby(new, sort=False).sum()
        return out.reindex(order)
    out = df.T.groupby(new, sort=False).sum().T
    return out[order]


def from_wide(wide: pd.DataFrame, merge_cn_mx: bool = True) -> ICIO:
    wide = wide.copy()
    wide.index = wide.index.astype(str).str.strip()
    wide.columns = wide.columns.astype(str).str.strip()
    if OUT in wide.index:
        wide = wide.drop(index=OUT)
    if OUT in wide.columns:
        wide = wide.drop(columns=OUT)
    acct = [r for r in wide.index if r in ACCOUNT_ROWS]
    ind_rows = [r for r in wide.index if r not in ACCOUNT_ROWS]
    ind_cols = [c for c in wide.columns if c in set(ind_rows)]
    fd_cols = [c for c in wide.columns if c not in set(ind_rows)]
    if len(ind_cols) != len(ind_rows):
        missing = set(ind_rows) - set(ind_cols)
        raise ValueError(f"Industry rows without matching columns: {sorted(missing)[:10]}")
    wide = wide.astype(float)
    if merge_cn_mx:
        wide = _merge_split_countries(_merge_split_countries(wide, 0), 1)
        ind_rows = [r for r in wide.index if r not in ACCOUNT_ROWS]
        ind_cols = list(ind_rows)
        fd_cols = [c for c in wide.columns if c not in set(ind_rows)]
    Z = wide.loc[ind_rows, ind_cols]
    Y = wide.loc[ind_rows, fd_cols]
    W = wide.loc[acct, ind_cols + fd_cols].reindex(list(ACCOUNT_ROWS)).fillna(0.0)
    return ICIO(Z=Z, Y=Y, W=W)


def read_icio(path: str | Path, merge_cn_mx: bool = True) -> ICIO:
    """Read one OECD ICIO 2025 csv (e.g. ``data/icio/2022.csv``)."""
    path = Path(path)
    if path.suffix == ".parquet":
        wide = pd.read_parquet(path)
    else:
        wide = pd.read_csv(path, index_col=0)
    return from_wide(wide, merge_cn_mx=merge_cn_mx)


def write_icio(icio: ICIO, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    icio.to_wide().to_csv(path)
    return path


def find_icio_file(icio_dir: Path, year: int) -> Path:
    for pat in (f"{year}.csv", f"*{year}*.csv", f"{year}.parquet"):
        hits = sorted(Path(icio_dir).glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No ICIO table for {year} in {icio_dir}. Download the OECD ICIO 2025 edition "
        "(2016-2022 zip) with scripts/download_data.py --icio, or copy {year}.csv there."
    )
