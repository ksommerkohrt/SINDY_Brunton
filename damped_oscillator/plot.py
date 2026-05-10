"""
Plots for the damped-oscillator toy example.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_phase_portrait(
    Z_true: np.ndarray,
    Z_sindy: np.ndarray,
    Z_exact: Optional[np.ndarray] = None,
    *,
    out_path: Optional[str] = None,
) -> tuple:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(Z_true[:, 0], Z_true[:, 1], lw=1.2, color="#1f77b4", label="Reference data")
    ax.plot(Z_sindy[:, 0], Z_sindy[:, 1], lw=1.0, ls="--", color="#ff7f0e", label="SINDy (integrated)")
    if Z_exact is not None:
        ax.plot(Z_exact[:, 0], Z_exact[:, 1], lw=0.6, ls=":", color="#2ca02c", label="Analytical solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Damped oscillator: phase portrait")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig, ax


def plot_time_series(
    t: np.ndarray,
    Z_true: np.ndarray,
    Z_sindy: np.ndarray,
    Z_exact: Optional[np.ndarray] = None,
    *,
    out_path: Optional[str] = None,
) -> tuple:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    labels = ["x(t)", "y(t)"]
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        n = min(len(t), Z_true.shape[0], Z_sindy.shape[0])
        ax.plot(t[:n], Z_true[:n, i], lw=1.2, color="#1f77b4", label="Reference data")
        ax.plot(t[:n], Z_sindy[:n, i], lw=1.0, ls="--", color="#ff7f0e", label="SINDy")
        if Z_exact is not None:
            ax.plot(t[:n], Z_exact[:n, i], lw=0.6, ls=":", color="#2ca02c", label="Analytical")
        ax.set_ylabel(lbl)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("t")
    fig.suptitle("Damped oscillator: time series")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig, axes
