"""
2D damped linear oscillator: dx/dt = mu*x + omega*y,  dy/dt = -omega*x + mu*y.

Exact solution from (x0, 0):  x(t) = x0*exp(mu*t)*cos(omega*t)
                               y(t) = -x0*exp(mu*t)*sin(omega*t)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def oscillator_rhs(t, state, mu: float = -0.1, omega: float = 2.0):
    x, y = state
    return [mu * x + omega * y, -omega * x + mu * y]


def simulate_damped_oscillator(
    t_span: tuple = (0.0, 30.0),
    n_points: int = 3000,
    x0: tuple = (2.0, 0.0),
    mu: float = -0.1,
    omega: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the 2D damped oscillator and return (t, Z) with Z shape (n_points, 2).
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        oscillator_rhs,
        t_span,
        list(x0),
        t_eval=t_eval,
        args=(mu, omega),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    return sol.t, sol.y.T


def exact_solution(
    t: np.ndarray,
    x0: float = 2.0,
    mu: float = -0.1,
    omega: float = 2.0,
) -> np.ndarray:
    """Analytical solution for initial condition (x0, 0) -> shape (len(t), 2)."""
    t = np.asarray(t)
    x = x0 * np.exp(mu * t) * np.cos(omega * t)
    y = -x0 * np.exp(mu * t) * np.sin(omega * t)
    return np.column_stack([x, y])
