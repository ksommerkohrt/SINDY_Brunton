"""
SINDy model and config for the 2D damped oscillator.
True equations:  dx/dt = mu*x + omega*y,  dy/dt = -omega*x + mu*y
"""
from __future__ import annotations

import sympy as sp

from sindy.pipeline import SINDySystemModel, SINDyRunConfig


def build_oscillator_model(mu: float = -0.1, omega: float = 2.0) -> SINDySystemModel:
    x, y = sp.symbols("x y", real=True)
    rhs_sym = [mu * x + omega * y, -omega * x + mu * y]
    rhs_lambdified = [sp.lambdify([x, y], e, modules="numpy") for e in rhs_sym]
    return SINDySystemModel(
        measured_names=["x", "y"],
        target_names=["x", "y"],
        target_indices=[0, 1],
        all_symbols=[x, y],
        rhs_lambdified=rhs_lambdified,
        rhs_symbolic=rhs_sym,
    )


def oscillator_sindy_config() -> SINDyRunConfig:
    """
    Pareto + BIC on raw states, degree-2 polynomial library.

    Using identity scaler and no column normalisation matches the Lorenz setup.
    Degree-2 library lets SINDy include quadratic terms; the sparsity penalty
    should zero all nonlinear features and keep only the four linear terms.
    """
    return SINDyRunConfig(
        sindy_mode="pareto",
        pareto_pick_mode="bic",
        scaler_kind="identity",
        max_degree=2,
        max_interaction=2,
        normalize_library_columns=False,
        use_physical_library=False,
        prefer_parsimony=False,
        alpha_ridge=1e-3,
        n_thresholds=50,
        threshold_min=1e-4,
        threshold_max=1.0,
        remove_double_trig_terms=False,
        library_include_constant=True,
        equal_weight_per_target=True,
    )
