"""Lorenz system demo for SINDY Engineering Toolkit."""
from .simulate import simulate_lorenz, lorenz_rhs
from .model import (
    build_lorenz_model,
    lorenz_sindy_config_bic_standard,
    lorenz_sindy_config_ensemble_bic_standard,
)
from .integrated import (
    eval_sindy_rhs,
    integrate_discovered_sindy,
    integrate_true_lorenz,
    integrated_3d_comparison,
    plot_lorenz_3d_true_vs_sindy,
)

__all__ = [
    "simulate_lorenz",
    "lorenz_rhs",
    "build_lorenz_model",
    "lorenz_sindy_config_bic_standard",
    "lorenz_sindy_config_ensemble_bic_standard",
    "eval_sindy_rhs",
    "integrate_discovered_sindy",
    "integrate_true_lorenz",
    "integrated_3d_comparison",
    "plot_lorenz_3d_true_vs_sindy",
]
