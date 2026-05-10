"""
Forward integration of a discovered 2D SINDy model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from idtools.preprocess import affine_from_sklearn_scaler


def eval_sindy_rhs_2d(z_phys: np.ndarray, fit: Dict[str, Any]) -> np.ndarray:
    lib = fit["library"]
    scaler = fit["scaler"]
    Z = np.asarray(z_phys, dtype=float).reshape(1, -1)
    if fit.get("use_physical_library"):
        theta = lib.transform(Z_scaled=Z, Z_phys=Z)
    else:
        aff = affine_from_sklearn_scaler(scaler)
        zs = aff.transform(Z)
        theta = lib.transform(Z_scaled=zs, Z_phys=Z)
    ki = np.asarray(fit["kept_idx"], dtype=int)
    coef = np.asarray(fit["coef_kept"], dtype=float)
    return (theta[:, ki] @ coef).ravel()


def integrate_discovered_2d(
    fit: Dict[str, Any],
    z0: np.ndarray,
    t_eval: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Integrate a 2-state discovered ODE along t_eval from z0. Returns (len(t_eval), 2)."""
    t_eval = np.asarray(t_eval, dtype=float).ravel()
    z0 = np.asarray(z0, dtype=float).ravel()

    def rhs(_t, x):
        return eval_sindy_rhs_2d(x, fit)

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        z0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"SINDy forward integration failed: {sol.message}")
    return sol.y.T
