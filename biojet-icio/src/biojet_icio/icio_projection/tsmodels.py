"""ARIMA(p, 1, q) and bridge regressions with ARIMA errors, written for short
annual series (the 28-year ICIO panel) and many of them (~10,000).

Model (on w_t = first difference of the series, usually of its log):

    w_t = c + b' z_t + sum_i phi_i w_{t-i} + sum_j theta_j e_{t-j} + e_t

* ARIMA:  no exogenous z (univariate, drift c).
* Bridge: z_t = first difference of the (log) proxy, e.g. the country's real
  GDP or broad-industry value added (the OECD's gap-filling regressions).

The lag orders p, q in {0, 1, 2} ("endogenous lags") are chosen by AICc.
Estimation is Hannan-Rissanen (two-stage least squares: a long AR gives
innovation estimates, then OLS on lagged w and lagged innovations), which
is closed-form and fast enough for the full panel. If statsmodels is
installed, ``fit_statsmodels`` refits a chosen order by exact ML as a check.

Non-stationary AR or non-invertible MA estimates are discarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ORDERS = [(p, q) for p in range(3) for q in range(3)]


@dataclass
class ArmaxFit:
    p: int
    q: int
    const: float
    beta: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    sigma2: float
    aicc: float
    w_hist: np.ndarray            # differenced history used
    e_hist: np.ndarray            # innovation estimates aligned to w_hist
    nobs: int
    info: dict = field(default_factory=dict)

    def forecast(self, h: int, z_future: np.ndarray | None = None) -> np.ndarray:
        """h-step forecast of the differenced series."""
        w = list(self.w_hist)
        e = list(self.e_hist)
        out = np.empty(h)
        for k in range(h):
            val = self.const
            if self.beta.size:
                if z_future is None:
                    raise ValueError("Bridge model needs future proxy values.")
                val += float(np.dot(self.beta, np.atleast_1d(z_future[k])))
            for i in range(self.p):
                val += self.phi[i] * w[-1 - i]
            for j in range(self.q):
                val += self.theta[j] * e[-1 - j]
            out[k] = val
            w.append(val)
            e.append(0.0)
        return out


def _lagmat(v: np.ndarray, lags: int, start: int) -> np.ndarray:
    n = len(v)
    return np.column_stack([v[start - i - 1:n - i - 1] for i in range(lags)]) if lags else np.empty((n - start, 0))


def _roots_ok(coefs: np.ndarray) -> bool:
    if coefs.size == 0:
        return True
    # 1 - c1 z - c2 z^2 ... roots must lie outside the unit circle
    poly = np.r_[-coefs[::-1], 1.0]
    roots = np.roots(poly)
    return bool(np.all(np.abs(roots) > 1.0 + 1e-6))


def _ols(Xm: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return coef, y - Xm @ coef


def fit_armax(w: np.ndarray, z: np.ndarray | None = None, orders=ORDERS, drift: bool = True) -> ArmaxFit:
    """Fit all candidate orders and return the AICc-best admissible model."""
    w = np.asarray(w, float)
    n = len(w)
    Z = None if z is None else np.asarray(z, float).reshape(n, -1)
    kz = 0 if Z is None else Z.shape[1]

    # Stage 1: long AR (with exog) for innovation estimates.
    m = int(min(4, max(1, n // 5)))
    X1 = [np.ones(n - m)] if drift else []
    if Z is not None:
        X1.append(Z[m:])
    X1.append(_lagmat(w, m, m))
    X1 = np.column_stack(X1)
    _, res1 = _ols(X1, w[m:])
    ehat = np.r_[np.zeros(m), res1]

    best = None
    for p, q in orders:
        start = m + max(p, q)
        nn = n - start
        k = int(drift) + kz + p + q + 1  # +1 for sigma2
        if nn - k - 1 <= 2:
            continue
        cols = [np.ones(nn)] if drift else []
        if Z is not None:
            cols.append(Z[start:])
        if p:
            cols.append(_lagmat(w, p, start))
        if q:
            cols.append(_lagmat(ehat, q, start))
        Xm = np.column_stack(cols) if cols else np.empty((nn, 0))
        if Xm.shape[1]:
            coef, res = _ols(Xm, w[start:])
        else:
            coef, res = np.empty(0), w[start:].copy()
        pos = 0
        c = coef[pos] if drift else 0.0
        pos += int(drift)
        beta = coef[pos:pos + kz]
        pos += kz
        phi = coef[pos:pos + p]
        pos += p
        theta = coef[pos:pos + q]
        if not _roots_ok(phi) or not _roots_ok(-theta):
            continue
        sigma2 = max(float(res @ res) / nn, 1e-14)
        aicc = nn * np.log(sigma2) + 2 * k + 2 * k * (k + 1) / (nn - k - 1)
        if best is None or aicc < best.aicc:
            e_hist = ehat.copy()
            e_hist[start:] = res
            best = ArmaxFit(p, q, float(c), np.asarray(beta), np.asarray(phi), np.asarray(theta),
                            sigma2, float(aicc), w.copy(), e_hist, nn)
    if best is None:
        # Fall back to drift-only random walk.
        c = float(w.mean()) if drift else 0.0
        res = w - c
        best = ArmaxFit(0, 0, c, np.zeros(kz), np.empty(0), np.empty(0),
                        max(float(res @ res) / max(n, 1), 1e-14), np.inf, w.copy(), res, n,
                        info={"fallback": True})
    return best


def project_series(y: np.ndarray, h: int, z_hist: np.ndarray | None = None,
                   z_future: np.ndarray | None = None, log: bool = True,
                   clip_growth: float = 0.5) -> tuple[np.ndarray, ArmaxFit | None]:
    """Project a level series h steps ahead with ARIMA(p,1,q) (+ bridge proxy).

    log=True models log levels (strictly positive series). Series that are
    all zero are returned as zeros. Annual growth is clipped to +/- clip_growth
    in log terms to avoid explosive paths from very short noisy series.
    """
    y = np.asarray(y, float)
    if np.allclose(y, 0.0):
        return np.zeros(h), None
    use_log = log and np.all(y > 0)
    lvl = np.log(y) if use_log else y
    w = np.diff(lvl)
    zh = None if z_hist is None else np.diff(np.log(z_hist) if use_log else z_hist, axis=0)
    zf = None
    if z_future is not None:
        zz = np.r_[np.atleast_2d(np.asarray(z_hist, float).reshape(len(y), -1))[-1:],
                   np.asarray(z_future, float).reshape(h, -1)]
        zf = np.diff(np.log(zz) if use_log else zz, axis=0)
    fit = fit_armax(w, zh)
    dw = fit.forecast(h, zf)
    if use_log:
        dw = np.clip(dw, -clip_growth, clip_growth)
        return np.exp(lvl[-1] + np.cumsum(dw)), fit
    return lvl[-1] + np.cumsum(dw), fit


def fit_statsmodels(y: np.ndarray, p: int, q: int, exog=None):  # pragma: no cover - optional
    """Exact-ML refit of a chosen order (requires statsmodels)."""
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(np.log(y), exog=exog, order=(p, 1, q), trend="t").fit()
