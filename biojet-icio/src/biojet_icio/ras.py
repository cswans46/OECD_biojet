"""Biproportional (RAS) balancing.

``ras_signed`` is the signed-matrix routine from cswans46/baltic
(``application._ras_balance``): negative cells (e.g. inventory changes) are held
at their seed values and the non-negative part is balanced to the residual row
and column targets, with an optional proportional-seed blend for sensitivity
runs. It is reused here for the bilateral trade matrices and for re-balancing
the disaggregated ICIO.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def ras_signed(seed, row_targets, col_targets, blend: float = 0.0, max_iter: int = 5000,
               tol: float = 1e-9, label: str = "RAS", rescale_cols: bool = True):
    seed = np.asarray(seed, dtype=float).copy()
    row_targets = np.asarray(row_targets, dtype=float).copy()
    col_targets = np.asarray(col_targets, dtype=float).copy()
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"{label}: blend must be in [0, 1]")
    if not (np.isfinite(seed).all() and np.isfinite(row_targets).all() and np.isfinite(col_targets).all()):
        raise ValueError(f"{label}: non-finite input")

    rt, ct = row_targets.sum(), col_targets.sum()
    if rescale_cols and ct != 0:
        col_targets *= rt / ct

    negative = np.minimum(seed, 0.0)
    pos = np.maximum(seed, 0.0)
    prow = np.maximum(row_targets - negative.sum(axis=1), 0.0)
    pcol = np.maximum(col_targets - negative.sum(axis=0), 0.0)
    total = prow.sum()
    if total <= 0:
        return negative, {"iterations": 0, "error": 0.0}
    pcol *= total / pcol.sum() if pcol.sum() > 0 else 1.0

    if blend > 0:
        prop = np.outer(prow, pcol) / total
        prop[seed < 0] = 0.0
        pos = (1 - blend) * pos + blend * prop

    # Targets without seed support cannot be met: report and fall back to a
    # proportional seed on that row/column so RAS can still converge.
    bad_r = np.where((prow > tol) & (pos.sum(axis=1) <= 0))[0]
    bad_c = np.where((pcol > tol) & (pos.sum(axis=0) <= 0))[0]
    if len(bad_r) or len(bad_c):
        log.warning("%s: %d rows / %d cols without seed support; using proportional seed there",
                    label, len(bad_r), len(bad_c))
        prop = np.outer(prow, pcol) / total
        prop[seed < 0] = 0.0
        pos[bad_r, :] = prop[bad_r, :]
        pos[:, bad_c] = prop[:, bad_c]

    err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        rs = pos.sum(axis=1)
        pos *= np.divide(prow, rs, out=np.ones_like(prow), where=rs > 0)[:, None]
        cs = pos.sum(axis=0)
        pos *= np.divide(pcol, cs, out=np.ones_like(pcol), where=cs > 0)[None, :]
        err = max(np.abs(pos.sum(axis=1) - prow).max(), np.abs(pos.sum(axis=0) - pcol).max())
        if err < tol * max(1.0, total * 1e-9):
            break
    if err > 1e-6 * max(1.0, total * 1e-6):
        log.warning("%s: did not fully converge after %d iterations (max error %.3e)", label, it, err)
    return pos + negative, {"iterations": it, "error": float(err)}
