"""
Validation metrics: compare an identified or measured (x, y, z) trajectory to the reference
mean-field simulator (PNAS Eq. 8).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .mean_field_simulator import (
    CylinderMeanFieldParams,
    DEFAULT_ILLUSTRATIVE_PARAMS,
    simulate_cylinder_mean_field,
)


def trajectory_validation_metrics(
    t: np.ndarray,
    y_reference: np.ndarray,
    y_candidate: np.ndarray,
) -> Dict[str, float]:
    """
    Pointwise comparison on aligned time grids (same shape).

    Parameters
    ----------
    t : shape (n,)
    y_reference, y_candidate : shape (n, 3) or (3, n); rows = time if (n, 3).

    Returns
    -------
    dict with rmse_total, rmse_x, rmse_y, rmse_z, r2_total, corr_x, corr_y, corr_z
    """
    t = np.asarray(t, dtype=float).ravel()
    yr = np.asarray(y_reference, dtype=float)
    yc = np.asarray(y_candidate, dtype=float)
    if yr.shape != yc.shape:
        raise ValueError("y_reference and y_candidate must have the same shape")

    if yr.ndim != 2:
        raise ValueError("y_reference must be 2-D (n, 3) or transpose")

    if yr.shape[1] == 3:
        pass
    elif yr.shape[0] == 3:
        yr = yr.T
        yc = yc.T
    else:
        raise ValueError("expected state dimension 3")

    if yr.shape[0] != t.size:
        raise ValueError("t length must match number of state rows")

    diff = yc - yr
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    rmse_total = float(np.sqrt(np.mean(diff**2)))

    var = np.var(yr, axis=0)
    mse = np.mean(diff**2, axis=0)
    r2_per = 1.0 - mse / (var + 1e-30)
    r2_total = float(1.0 - np.mean(diff**2) / (np.var(yr) + 1e-30))

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-15 or np.std(b) < 1e-15:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "rmse_total": rmse_total,
        "rmse_x": float(rmse[0]),
        "rmse_y": float(rmse[1]),
        "rmse_z": float(rmse[2]),
        "r2_total": r2_total,
        "r2_x": float(r2_per[0]),
        "r2_y": float(r2_per[1]),
        "r2_z": float(r2_per[2]),
        "corr_x": _corr(yr[:, 0], yc[:, 0]),
        "corr_y": _corr(yr[:, 1], yc[:, 1]),
        "corr_z": _corr(yr[:, 2], yc[:, 2]),
    }


def compare_to_mean_field_reference(
    t: np.ndarray,
    y_candidate: np.ndarray,
    *,
    params: CylinderMeanFieldParams = DEFAULT_ILLUSTRATIVE_PARAMS,
    y0: Optional[np.ndarray] = None,
    integrator_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Simulate Eq. 8 from ``y0`` (default: first row of ``y_candidate``) and score ``y_candidate``.

    If ``t`` is not uniform, the reference is still evaluated at the same ``t`` samples.

    Returns
    -------
    metrics : dict from :func:`trajectory_validation_metrics`
    y_reference : shape (n, 3) aligned to ``t``
    """
    t = np.asarray(t, dtype=float).ravel()
    yc = np.asarray(y_candidate, dtype=float)
    if yc.ndim != 2:
        raise ValueError("y_candidate must be 2-D")
    if yc.shape[1] == 3:
        pass
    elif yc.shape[0] == 3:
        yc = yc.T
    else:
        raise ValueError("expected state dimension 3")
    if yc.shape[0] != t.size:
        raise ValueError("t and y_candidate must have matching lengths")

    if y0 is None:
        y0_use = yc[0].copy()
    else:
        y0_use = np.asarray(y0, dtype=float).ravel()
        if y0_use.size != 3:
            raise ValueError("y0 must have length 3")

    kw = dict(rtol=1e-8, atol=1e-10)
    if integrator_kwargs:
        kw.update(integrator_kwargs)

    sol = simulate_cylinder_mean_field(
        (float(t[0]), float(t[-1])),
        y0_use,
        t_eval=t,
        params=params,
        **kw,
    )
    if not sol.success:
        raise RuntimeError(f"reference integration failed: {sol.message}")

    y_ref = sol.y.T
    metrics = trajectory_validation_metrics(t, y_ref, yc)
    return metrics, y_ref


def interpolate_states(
    t_source: np.ndarray,
    y_source: np.ndarray,
    t_target: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of each state component onto ``t_target`` (for mismatched sampling)."""
    t_source = np.asarray(t_source, dtype=float).ravel()
    y_source = np.asarray(y_source, dtype=float)
    if y_source.shape[0] == 3:
        y_source = y_source.T
    if y_source.ndim != 2 or y_source.shape[1] != 3:
        raise ValueError("y_source must be (n, 3)")

    out = np.zeros((t_target.size, 3), dtype=float)
    for k in range(3):
        out[:, k] = interp1d(
            t_source,
            y_source[:, k],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(t_target)
    return out
