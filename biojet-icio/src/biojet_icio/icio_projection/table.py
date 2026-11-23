"""Minimal ICIO container and OECD wide-csv reader shared by both repositories.

The container has the same fields as ``biojet_icio.icio.ICIO`` (``Z``, ``Y``,
``W``) so tables pass between the two without conversion:

    Z  intermediate use, industry rows x industry columns (USD million)
    Y  final demand, industry rows x final-demand columns (``CAN_HFCE`` ...)
    W  rows ``TLS`` and ``VA`` x (industry columns + final-demand columns)

Country of a label = text before the first underscore, except ``EU27_2020``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ACCOUNT_ROWS = ("TLS", "VA")
OUT = "OUT"
MERGE_PREFIX = {"CN1": "CHN", "CN2": "CHN", "MX1": "MEX", "MX2": "MEX"}
FD_CODES = ("HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR")


def country_of(label: str) -> str:
    label = str(label)
    if label.startswith("EU27_2020"):
        return "EU27_2020"
    return label.split("_", 1)[0]


def sector_of(label: str) -> str:
    label = str(label)
    if label.startswith("EU27_2020_"):
        return label[len("EU27_2020_"):]
    return label.split("_", 1)[1] if "_" in label else label


@dataclass
class Table:
    Z: pd.DataFrame
    Y: pd.DataFrame
    W: pd.DataFrame

    @property
    def keys(self) -> list[str]:
        return list(self.Z.index)

    def x(self) -> pd.Series:
        return self.Z.sum(axis=1) + self.Y.sum(axis=1)

    def x_cols(self) -> pd.Series:
        return self.Z.sum(axis=0) + self.W[self.Z.columns].sum(axis=0)

    def copy(self) -> "Table":
        return Table(self.Z.copy(), self.Y.copy(), self.W.copy())

    def max_imbalance(self) -> float:
        r, c = self.x(), self.x_cols()
        return float((r - c).abs().max())

    @classmethod
    def from_icio(cls, icio) -> "Table":
        """Accept any object with Z, Y, W (e.g. biojet_icio.icio.ICIO)."""
        return cls(icio.Z, icio.Y, icio.W)

    @classmethod
    def from_axfd(cls, A: pd.DataFrame, X: pd.Series, FD: pd.DataFrame) -> "Table":
        """Build from A, X, FD (Datacenters RData layout). VA absorbs TLS."""
        keys = list(A.index)
        x = X.reindex(keys).to_numpy(float)
        Z = pd.DataFrame(A.loc[keys, keys].to_numpy(float) * x[None, :], index=keys, columns=keys)
        Y = FD.reindex(keys).astype(float)
        va = x - Z.sum(axis=0).to_numpy()
        cols = keys + list(Y.columns)
        W = pd.DataFrame(0.0, index=list(ACCOUNT_ROWS), columns=cols)
        W.loc["VA", keys] = va
        return cls(Z, Y, W)

    def to_axfd(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        x = self.x()
        xv = x.to_numpy()
        A = pd.DataFrame(np.divide(self.Z.to_numpy(), xv[None, :], out=np.zeros(self.Z.shape), where=xv[None, :] > 0),
                         index=self.Z.index, columns=self.Z.columns)
        return A, x, self.Y.copy()


def _merge_split_countries(df: pd.DataFrame, axis: int) -> pd.DataFrame:
    labels = df.index if axis == 0 else df.columns
    new = [MERGE_PREFIX[l[:3]] + l[3:] if l[:3] in MERGE_PREFIX else l for l in labels]
    if new == list(labels):
        return df
    order = list(dict.fromkeys(new))
    if axis == 0:
        return df.groupby(new, sort=False).sum().reindex(order)
    return df.T.groupby(new, sort=False).sum().T[order]


def from_wide(wide: pd.DataFrame, merge_cn_mx: bool = True) -> Table:
    wide = wide.copy()
    wide.index = wide.index.astype(str).str.strip()
    wide.columns = wide.columns.astype(str).str.strip()
    wide = wide.drop(index=[OUT], errors="ignore").drop(columns=[OUT], errors="ignore")
    if merge_cn_mx:
        wide = _merge_split_countries(_merge_split_countries(wide, 0), 1)
    acct = [r for r in wide.index if r in ACCOUNT_ROWS]
    rows = [r for r in wide.index if r not in ACCOUNT_ROWS]
    rowset = set(rows)
    fd = [c for c in wide.columns if c not in rowset]
    wide = wide.astype(float)
    W = wide.loc[acct, rows + fd].reindex(list(ACCOUNT_ROWS)).fillna(0.0)
    return Table(Z=wide.loc[rows, rows], Y=wide.loc[rows, fd], W=W)


def read_wide(path: str | Path, merge_cn_mx: bool = True) -> Table:
    path = Path(path)
    if path.suffix == ".parquet":
        wide = pd.read_parquet(path)
    else:
        wide = pd.read_csv(path, index_col=0)
    return from_wide(wide, merge_cn_mx=merge_cn_mx)


def save_table(t: Table, path: str | Path) -> Path:
    """Compact binary store (npz) of one table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, Z=t.Z.to_numpy(), Y=t.Y.to_numpy(), W=t.W.to_numpy(),
        keys=np.array(t.Z.index, dtype=object), fd=np.array(t.Y.columns, dtype=object),
        wrows=np.array(t.W.index, dtype=object), wcols=np.array(t.W.columns, dtype=object),
    )
    return path


def load_table(path: str | Path) -> Table:
    d = np.load(path, allow_pickle=True)
    keys, fd = list(d["keys"]), list(d["fd"])
    return Table(
        Z=pd.DataFrame(d["Z"], keys, keys),
        Y=pd.DataFrame(d["Y"], keys, fd),
        W=pd.DataFrame(d["W"], list(d["wrows"]), list(d["wcols"])),
    )


def to_wide(t: Table) -> pd.DataFrame:
    """OECD wide layout: industry rows + TLS, VA, OUT; columns industries + FD + OUT."""
    top = pd.concat([t.Z, t.Y], axis=1)
    bottom = t.W.reindex(columns=top.columns).fillna(0.0)
    w = pd.concat([top, bottom])
    out_col = pd.Series(0.0, index=w.index)
    out_col.loc[t.Z.index] = t.x()
    w[OUT] = out_col
    out_row = pd.Series(0.0, index=w.columns)
    out_row.loc[t.Z.columns] = t.x_cols()
    w.loc[OUT] = out_row
    return w
