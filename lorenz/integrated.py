"""
Forward integration of a discovered SINDy model and 3D comparison to the true Lorenz trajectory.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from idtools.preprocess import affine_from_sklearn_scaler


def eval_sindy_rhs(z_phys: np.ndarray, fit: Dict[str, Any]) -> np.ndarray:
    """
    Time-autonomous SINDy RHS: derivatives at physical state ``z_phys`` (length 3).

    Matches :func:`sindy.pipeline.validate_sindy_general` (scaled library + ``coef_kept``).
    """
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


def integrate_discovered_sindy(
    fit: Dict[str, Any],
    z0: np.ndarray,
    t_eval: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Integrate discovered ODE along ``t_eval`` from initial condition ``z0``.

    Returns ``Z_sindy`` with shape ``(len(t_eval), 3)``.
    """
    t_eval = np.asarray(t_eval, dtype=float).ravel()
    z0 = np.asarray(z0, dtype=float).ravel()
    if z0.size != 3:
        raise ValueError(f"z0 must have length 3, got {z0.shape}")

    def rhs(_t: float, x: np.ndarray) -> np.ndarray:
        return eval_sindy_rhs(x, fit)

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


def integrate_true_lorenz(
    z0: np.ndarray,
    t_eval: np.ndarray,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> np.ndarray:
    """Integrate true Lorenz ODE (same defaults as :func:`lorenz.simulate.simulate_lorenz`)."""

    from lorenz.simulate import lorenz_rhs

    t_eval = np.asarray(t_eval, dtype=float).ravel()
    z0 = np.asarray(z0, dtype=float).ravel()

    sol = solve_ivp(
        lorenz_rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        z0,
        t_eval=t_eval,
        args=(sigma, rho, beta),
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"True Lorenz integration failed: {sol.message}")
    return sol.y.T


def plot_lorenz_3d_true_vs_sindy(
    Z_true: np.ndarray,
    Z_sindy: np.ndarray,
    *,
    Z_true_reint: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
    max_points: int = 12000,
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    3D line plot: true trajectory vs SINDy-forward trajectory (optional true re-integrated overlay).
    """
    import matplotlib.pyplot as plt

    n = min(len(Z_true), len(Z_sindy))
    Zt = np.asarray(Z_true[:n], dtype=float)
    Zs = np.asarray(Z_sindy[:n], dtype=float)
    idx = np.arange(n)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        Zt = Zt[idx]
        Zs = Zs[idx]

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Zt[:, 0], Zt[:, 1], Zt[:, 2], lw=0.5, alpha=0.9, color="#1f77b4", label="True (reference data)")
    ax.plot(Zs[:, 0], Zs[:, 1], Zs[:, 2], lw=0.5, alpha=0.9, color="#ff7f0e", label="SINDy (integrated)")
    if Z_true_reint is not None:
        Zr = np.asarray(Z_true_reint[:n], dtype=float)
        if n > max_points:
            Zr = Zr[idx]
        ax.plot(
            Zr[:, 0],
            Zr[:, 1],
            Zr[:, 2],
            lw=0.4,
            alpha=0.5,
            color="#2ca02c",
            linestyle="--",
            label="True ODE (re-integrated)",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title or "Lorenz: true vs SINDy forward integration")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig, ax


def integrated_3d_comparison(
    res: Dict[str, Any],
    t: np.ndarray,
    Z_true: np.ndarray,
    *,
    out_path: Optional[str] = None,
    z0: Optional[np.ndarray] = None,
    include_true_reintegrated: bool = True,
    plot_max_points: int = 12000,
    title: Optional[str] = None,
    t_subsample_step: int = 1,
) -> Dict[str, Any]:
    """
    Integrate discovered model from ``z0`` (default ``Z_true[0]``), optionally re-integrate true ODE,
    and build a 3D figure.

    Parameters
    ----------
    res :
        Pipeline result dict containing ``res["fit"]``.
    t_subsample_step :
        If > 1, use ``t[::step]`` and ``Z_true[::step]`` for integration (faster for long demos;
        RMSE is on this subsampled grid).
    """
    t = np.asarray(t, dtype=float).ravel()
    Z_true = np.asarray(Z_true, dtype=float)
    step = max(1, int(t_subsample_step))
    if step > 1:
        t = t[::step]
        Z_true = Z_true[::step]
    z0 = np.asarray(Z_true[0], dtype=float) if z0 is None else np.asarray(z0, dtype=float).ravel()

    fit = res["fit"]
    Z_sindy = integrate_discovered_sindy(fit, z0, t)
    Z_reint = None
    if include_true_reintegrated:
        Z_reint = integrate_true_lorenz(z0, t)

    fig, ax = plot_lorenz_3d_true_vs_sindy(
        Z_true,
        Z_sindy,
        Z_true_reint=Z_reint,
        out_path=out_path,
        max_points=plot_max_points,
        title=title,
    )
    rmse = float(np.sqrt(np.mean((Z_true - Z_sindy) ** 2)))
    return {
        "Z_sindy": Z_sindy,
        "Z_true_reintegrated": Z_reint,
        "rmse_state": rmse,
        "fig": fig,
        "ax": ax,
    }
