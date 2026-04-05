# idtools/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.signal
from sklearn.preprocessing import MaxAbsScaler, StandardScaler


class IdentityScaler:
    """
    No-op state scaling for SINDy: ``transform(X) == X``, ``scale_ = 1``, ``mean_ = 0``.

    Duck-types like sklearn scalers so :func:`affine_from_sklearn_scaler` and the pipeline work unchanged.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.shape}")
        self.scale_ = np.ones(X.shape[1], dtype=float)
        self.mean_ = np.zeros(X.shape[1], dtype=float)
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float)


ScalerLike = Union[MaxAbsScaler, StandardScaler, IdentityScaler]


@dataclass
class PreprocessResult:
    X_scaled: np.ndarray                 # (n_samples, n_vars)
    X_dot_scaled: Optional[np.ndarray]   # (n_samples, n_vars) or None
    scaler: ScalerLike
    dt: float
    meta: Dict


def estimate_dt(t: np.ndarray, method: str = "median") -> float:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size < 2:
        raise ValueError("t must have at least 2 samples")

    dts = np.diff(t)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        raise ValueError("Could not estimate dt (non-positive or non-finite diffs)")

    if method == "median":
        return float(np.median(dts))
    if method == "mean":
        return float(np.mean(dts))
    raise ValueError(f"Unknown method: {method}")


def fit_scaler(X: np.ndarray, kind: str = "maxabs") -> ScalerLike:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")

    if kind == "maxabs":
        scaler: ScalerLike = MaxAbsScaler()
    elif kind == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    elif kind in ("identity", "none"):
        scaler = IdentityScaler()
    else:
        raise ValueError(f"Unknown scaler kind: {kind}")

    scaler.fit(X)
    return scaler


def scale(X: np.ndarray, scaler: ScalerLike) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return scaler.transform(X)


def estimate_derivatives_savgol(
    X_scaled: np.ndarray,
    dt: float,
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 1,
    axis: int = 0,
) -> np.ndarray:
    """
    Savitzky–Golay derivative estimate on already-scaled data.

    Notes:
    - window_length must be odd and > polyorder and <= n_samples.
    - If your data is (n_samples, n_vars), use axis=0 (default).
    """
    X_scaled = np.asarray(X_scaled, dtype=float)
    if X_scaled.ndim != 2:
        raise ValueError(f"X_scaled must be 2D, got {X_scaled.shape}")
    if float(dt) <= 0:
        raise ValueError("dt must be positive")
    if int(polyorder) < 0:
        raise ValueError("polyorder must be >= 0")
    if int(deriv) < 0:
        raise ValueError("deriv must be >= 0")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    n = int(X_scaled.shape[axis])

    wl = int(window_length)
    if wl > n:
        wl = n
    if wl % 2 == 0:
        wl -= 1

    min_wl = int(polyorder) + 2
    if min_wl % 2 == 0:
        min_wl += 1
    if wl < min_wl:
        wl = min_wl

    if wl > n:
        raise ValueError(
            f"Not enough samples for SavGol: need n>={wl} (got n={n}) "
            f"for polyorder={polyorder}, window_length request={window_length}."
        )

    return scipy.signal.savgol_filter(
        X_scaled,
        window_length=wl,
        polyorder=int(polyorder),
        deriv=int(deriv),
        delta=float(dt),
        axis=int(axis),
        mode="interp",
    )


def preprocess_timeseries(
    X: np.ndarray,
    t: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    scaler_kind: str = "maxabs",
    savgol_window: int = 11,
    savgol_poly: int = 3,
    deriv: int = 1,
    compute_derivatives: bool = True,
    X_dot_phys: Optional[np.ndarray] = None,
) -> PreprocessResult:
    """
    Shared preprocessing:
      1) scale X (``scaler_kind``: ``maxabs``, ``standard``, or ``identity`` / ``none`` for no scaling)
      2) derivatives: if X_dot_phys is provided use it (exact derivatives, e.g. from known ODE);
         else if compute_derivatives, estimate via SavGol on scaled data.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_vars), got {X.shape}")

    if dt is None:
        if t is None:
            raise ValueError("Provide either dt or t")
        dt = estimate_dt(t, method="median")

    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")

    scaler = fit_scaler(X, kind=scaler_kind)
    X_scaled = scaler.transform(X)

    X_dot_scaled: Optional[np.ndarray] = None
    if X_dot_phys is not None:
        X_dot_phys = np.asarray(X_dot_phys, dtype=float)
        if X_dot_phys.shape != X.shape:
            raise ValueError(f"X_dot_phys must have same shape as X: {X.shape} vs {X_dot_phys.shape}")
        # Scale derivatives by 1/scale only. Do NOT subtract mean: d/dt[(x-mu)/sigma] = (1/sigma)*dx/dt.
        # scaler.transform(X_dot_phys) would use (X_dot - mean_state)/scale for StandardScaler, which is wrong.
        scale = np.asarray(scaler.scale_, dtype=float).reshape(1, -1)
        scale = np.where(scale == 0, 1.0, scale)
        X_dot_scaled = X_dot_phys / scale
    elif compute_derivatives:
        X_dot_scaled = estimate_derivatives_savgol(
            X_scaled,
            dt=dt,
            window_length=int(savgol_window),
            polyorder=int(savgol_poly),
            deriv=int(deriv),
            axis=0,
        )

    meta = {
        "scaler_kind": str(scaler_kind),
        "savgol_window": int(savgol_window),
        "savgol_poly": int(savgol_poly),
        "deriv": int(deriv),
        "compute_derivatives": bool(compute_derivatives),
        "used_exact_derivatives": X_dot_phys is not None,
    }

    return PreprocessResult(
        X_scaled=X_scaled,
        X_dot_scaled=X_dot_scaled,
        scaler=scaler,
        dt=dt,
        meta=meta,
    )


def normalize_columns(X: np.ndarray, eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    norms = np.linalg.norm(X, axis=0)
    norms = np.where(norms == 0.0, 1.0, norms)
    if float(eps) > 0:
        norms = np.maximum(norms, float(eps))

    return X / norms, norms


@dataclass(frozen=True)
class AffineScaler:
    # x_scaled = (x - offset) / scale
    offset: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.offset) / self.scale

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        Xs = np.asarray(Xs, dtype=float)
        return Xs * self.scale + self.offset


def affine_from_sklearn_scaler(sklearn_scaler) -> AffineScaler:
    """
    Supports MaxAbsScaler and StandardScaler in the common affine form.
    Returns an object with:
      x_scaled = (x - offset) / scale
    """
    if hasattr(sklearn_scaler, "scale_") and hasattr(sklearn_scaler, "mean_"):
        scale = np.asarray(sklearn_scaler.scale_, dtype=float)
        offset = np.asarray(sklearn_scaler.mean_, dtype=float)
    elif hasattr(sklearn_scaler, "scale_"):
        scale = np.asarray(sklearn_scaler.scale_, dtype=float)
        offset = np.zeros_like(scale, dtype=float)
    else:
        raise TypeError(f"Unsupported scaler type: {type(sklearn_scaler)}")

    scale = np.where(scale == 0.0, 1.0, scale)
    return AffineScaler(offset=offset, scale=scale)

