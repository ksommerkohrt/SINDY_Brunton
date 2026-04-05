"""
Lorenz system simulation for SINDY Engineering Toolkit demos.
dx/dt = sigma*(y - x),  dy/dt = x*(rho - z) - y,  dz/dt = x*y - beta*z
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def lorenz_rhs(t, x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz ODE RHS. x = (x, y, z)."""
    return [
        sigma * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - beta * x[2],
    ]


def simulate_lorenz(
    t_span: tuple = (0.0, 20.0),
    n_points: int | None = None,
    dt: float | None = None,
    x0: tuple = (1.0, 1.0, 1.0),
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate Lorenz and return (t, Z_phys) with Z_phys shape (n_samples, 3) columns [x, y, z].

    Either pass dt (time step) or n_points (number of samples). If dt is given, time points
    are t_span[0], t_span[0]+dt, ... up to t_span[1] (paper-style: dt=0.001, t_span=(0, 100)).
    If only n_points is given, times are linspace(t_span[0], t_span[1], n_points).
    """
    if dt is not None:
        dt = float(dt)
        t_eval = np.arange(t_span[0], t_span[1] + 0.5 * dt, dt)
        t_eval = t_eval[t_eval <= t_span[1]]
    else:
        n = int(n_points) if n_points is not None else 2000
        t_eval = np.linspace(t_span[0], t_span[1], n)
    if random_state is not None:
        rng = np.random.default_rng(random_state)
        x0 = (float(x0[0] + 0.1 * rng.standard_normal()),
              float(x0[1] + 0.1 * rng.standard_normal()),
              float(x0[2] + 0.1 * rng.standard_normal()))
    sol = solve_ivp(
        lorenz_rhs,
        t_span,
        x0,
        t_eval=t_eval,
        args=(sigma, rho, beta),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    t = sol.t
    Z_phys = sol.y.T  # (n_points, 3)
    return t, Z_phys
