"""A small balanced synthetic ICIO with the 2025 sector and FD layout, for tests.

Not data: magnitudes are random apart from a stylised energy and transport
structure (B06 -> C19 -> H49/H51/households), sized so that the real UNSD
biofuel volumes fit inside each region's C19 output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .icio import ICIO, make_label

SECTORS = ["A01", "A02", "A03", "B05", "B06", "B07", "B08", "B09", "C10T12", "C13T15", "C16",
           "C17_18", "C19", "C20", "C21", "C22", "C23", "C24A", "C24B", "C25", "C26", "C27", "C28",
           "C29", "C301", "C302T309", "C31T33", "D", "E", "F", "G", "H49", "H50", "H51", "H52", "H53",
           "I", "J58T60", "J61", "J62_63", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
FD = ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"]
DEFAULT_REGIONS = {"USA": 26e6, "BRA": 2.0e6, "DEU": 4.3e6, "NLD": 1.1e6, "FIN": 0.3e6,
                   "IDN": 1.3e6, "SAU": 1.1e6, "CAN": 2.1e6, "ROW": 40e6}   # ~GDP, USD million


def make_synthetic(regions: dict[str, float] | None = None, seed: int = 0) -> ICIO:
    rng = np.random.default_rng(seed)
    regions = regions or DEFAULT_REGIONS
    R, S = list(regions), SECTORS
    n = len(R) * len(S)
    labels = [make_label(r, s) for r in R for s in S]
    si = {s: i for i, s in enumerate(S)}
    A = np.zeros((n, n))
    for a, r in enumerate(R):
        for b, c in enumerate(R):
            blk = rng.uniform(0, 1, (len(S), len(S))) * (rng.uniform(0, 1, (len(S), len(S))) < 0.35)
            blk *= 1.0 if a == b else 0.04
            A[a*len(S):(a+1)*len(S), b*len(S):(b+1)*len(S)] = blk
    # stylised energy chain
    for b in range(len(R)):
        col = lambda s: b*len(S) + si[s]  # noqa: E731
        for a in range(len(R)):
            w = 1.0 if a == b else 0.15
            A[a*len(S) + si["B06"], col("C19")] = 6.0 * w
            A[a*len(S) + si["C19"], col("H51")] = 4.0 * w
            A[a*len(S) + si["C19"], col("H49")] = 2.0 * w
            A[a*len(S) + si["A01"], col("C10T12")] = 3.0 * w
            A[a*len(S) + si["C10T12"], col("A01")] = 1.0 * w
    A = A / A.sum(axis=0, keepdims=True) * 0.55
    A[:, [i for i, l in enumerate(labels) if l.endswith("_C19")]] *= 0.85 / 0.55
    f = np.concatenate([rng.uniform(0.5, 1.5, len(S)) * regions[r] / len(S) for r in R])
    for i, l in enumerate(labels):
        if l.endswith("_C19"):
            f[i] *= 3.0
    x = np.linalg.solve(np.eye(n) - A, f)
    Z = A * x[None, :]
    fd_labels = [make_label(r, k) for r in R for k in FD]
    Y = np.zeros((n, len(fd_labels)))
    w = np.array([0.6, 0.02, 0.15, 0.2, 0.0, 0.03])
    for i in range(n):
        dest = rng.dirichlet(np.ones(len(R)) * 0.3)
        home = R.index(labels[i][:3])
        dest = 0.7 * np.eye(len(R))[home] + 0.3 * dest
        Y[i] = np.kron(dest, w) * f[i]
        # small inventory change of either sign, offset in GFCF
        inv = rng.normal(0, 0.005) * f[i]
        Y[i, home*len(FD) + 4] += inv
        Y[i, home*len(FD) + 3] -= inv
    tls = 0.03 * x
    va = x - Z.sum(axis=0) - tls
    Wz = np.vstack([tls, va])
    Zdf = pd.DataFrame(Z, labels, labels)
    Ydf = pd.DataFrame(Y, labels, fd_labels)
    W = pd.DataFrame(np.hstack([Wz, np.vstack([np.zeros(len(fd_labels)), np.zeros(len(fd_labels))])]),
                     index=["TLS", "VA"], columns=labels + fd_labels)
    return ICIO(Z=Zdf, Y=Ydf, W=W)
