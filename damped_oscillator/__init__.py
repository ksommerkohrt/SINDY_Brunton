"""2D damped linear oscillator — SINDy toy/verification example."""
from .simulate import simulate_damped_oscillator, exact_solution, oscillator_rhs
from .model import build_oscillator_model, oscillator_sindy_config
from .plot import plot_phase_portrait, plot_time_series

__all__ = [
    "simulate_damped_oscillator",
    "exact_solution",
    "oscillator_rhs",
    "build_oscillator_model",
    "oscillator_sindy_config",
    "plot_phase_portrait",
    "plot_time_series",
]
