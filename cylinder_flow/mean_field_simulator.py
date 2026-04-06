"""
Mean-field cylinder wake model in POD + shift coordinates (Brunton et al., PNAS 2016, Eq. 8;
Noack et al., J. Fluid Mech. 2003).

State (x, y, z): first two POD coefficients and shift mode. Large lambda_z forces z ~ x^2 + y^2 on a
fast time scale, recovering Hopf normal-form structure on the slow manifold.

Reduced coordinates from DNS/POD are only consistent with this scaling after fixing POD L2 norms;
see ``estimate_a3_scale_lstsq`` / ``calibrate_a3_scale`` in ``snapshot_pod_shift``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class CylinderMeanFieldParams:
    """Parameters for Eq. 8 (mu, omega, A, lambda)."""

    mu: float
    omega: float
    A: float
    lambda_z: float


# Not from the paper's numeric tables (SI PDF); illustrative values giving a stable limit cycle
# for validation workflows. Replace with fitted values from DNS/POD data or SI Appendix.
DEFAULT_ILLUSTRATIVE_PARAMS = CylinderMeanFieldParams(
    mu=0.1,
    omega=1.0,
    A=-1.0,
    lambda_z=10.0,
)


def cylinder_mean_field_rhs(
    _t: float,
    state: ArrayLike,
    mu: float,
    omega: float,
    A: float,
    lambda_z: float,
) -> np.ndarray:
    """Right-hand side for Eq. 8; state order (x, y, z)."""
    x, y, z = np.asarray(state, dtype=float).ravel()[:3]
    x_dot = mu * x - omega * y + A * x * z
    y_dot = omega * x + mu * y + A * y * z
    z_dot = -lambda_z * (z - x * x - y * y)
    return np.array([x_dot, y_dot, z_dot], dtype=float)


def _pack_rhs(
    params: CylinderMeanFieldParams,
) -> Callable[[float, np.ndarray], np.ndarray]:
    def rhs(t: float, s: np.ndarray) -> np.ndarray:
        return cylinder_mean_field_rhs(
            t, s, params.mu, params.omega, params.A, params.lambda_z
        )

    return rhs


def simulate_cylinder_mean_field(
    t_span: Tuple[float, float],
    y0: ArrayLike,
    *,
    t_eval: Optional[np.ndarray] = None,
    params: CylinderMeanFieldParams = DEFAULT_ILLUSTRATIVE_PARAMS,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    **solve_ivp_kwargs: Any,
) -> Any:
    """
    Integrate Eq. 8 with scipy.integrate.solve_ivp.

    Parameters
    ----------
    t_span : (t0, t1)
    y0 : array-like, shape (3,)
    t_eval : optional dense output times (monotonic)
    params : CylinderMeanFieldParams
    rtol, atol : integrator tolerances
    **solve_ivp_kwargs : forwarded to solve_ivp (e.g. method='RK45')

    Returns
    -------
    OdeSolution
        SciPy solution object; sol.t and sol.y (shape (3, n_times)).
    """
    y0_arr = np.asarray(y0, dtype=float).ravel()
    if y0_arr.size != 3:
        raise ValueError("y0 must have length 3 (x, y, z)")

    rhs = _pack_rhs(params)
    return solve_ivp(
        rhs,
        t_span,
        y0_arr,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **solve_ivp_kwargs,
    )
