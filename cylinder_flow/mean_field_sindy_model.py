"""
Brunton PNAS Eq. 8 mean-field cylinder wake as :class:`~sindy.pipeline.SINDySystemModel` + SINDy config.
"""
from __future__ import annotations

from typing import Optional

import sympy as sp

from sindy.pipeline import SINDyRunConfig, SINDySystemModel

from .mean_field_simulator import CylinderMeanFieldParams, DEFAULT_ILLUSTRATIVE_PARAMS


def build_cylinder_mean_field_model(
    params: Optional[CylinderMeanFieldParams] = None,
) -> SINDySystemModel:
    """
    Symbolic RHS for Eq. 8 in coordinates ``(x, y, z)`` (first two POD modes + shift mode).

    Parameters match :data:`~cylinder_flow.mean_field_simulator.DEFAULT_ILLUSTRATIVE_PARAMS` by default.
    """
    params = params or DEFAULT_ILLUSTRATIVE_PARAMS
    x, y, z = sp.symbols("x y z", real=True)
    state_syms = [x, y, z]
    mu, om, A, lz = params.mu, params.omega, params.A, params.lambda_z
    rhs_sym = [
        float(mu) * x - float(om) * y + float(A) * x * z,
        float(om) * x + float(mu) * y + float(A) * y * z,
        float(-lz) * (z - x**2 - y**2),
    ]
    rhs_lambdified = [sp.lambdify(state_syms, e, modules="numpy") for e in rhs_sym]
    return SINDySystemModel(
        measured_names=["x", "y", "z"],
        target_names=["x", "y", "z"],
        target_indices=[0, 1, 2],
        all_symbols=state_syms,
        rhs_lambdified=rhs_lambdified,
        rhs_symbolic=rhs_sym,
    )


def cylinder_mean_field_polynomial_budget() -> dict:
    """
    Polynomial atoms for PNAS Eq. 8 in ``(x, y, z)``.

    **Per-state ``lin`` counts are not per-target locks.** The pipeline builds one shared library
    ``Theta(Z)`` from *all* atoms; each equation (``dx/dt``, ``dy/dt``, ``dz/dt``) is fit with the
    same columns. So ``z: {"lin": 1}`` only adds a linear atom for the symbol *z* (enabling ``z``,
    ``x*z``, ``y*z``, …); it does **not** forbid ``x**2`` or ``y**2`` in the ``dz/dt`` row. Those
    monomials come from ``x``/``y`` having ``lin: 2`` (atoms ``x``, ``x*x``, ``y``, ``y*y``) plus
    ``max_degree`` / ``max_interaction``. Library column names use ``x*x``, ``y*y`` (not ``x**2``).

    With ``max_degree=3``, ``max_interaction=2`` (see :func:`cylinder_sindy_config_bic_standard`):

    - ``dx/dt``: ``x``, ``y``, ``x*z`` (and other allowed monomials).
    - ``dy/dt``: ``x``, ``y``, ``y*z``.
    - ``dz/dt``: ``z``, ``x*x``, ``y*y`` for ``z_dot = -lambda * (z - x^2 - y^2)``.
    """
    b = {"lin": 2, "trig": 0, "inv": 0, "sat": 0}
    return {"x": dict(b), "y": dict(b), "z": {"lin": 1, "trig": 0, "inv": 0, "sat": 0}}


def cylinder_sindy_config_bic_standard() -> SINDyRunConfig:
    """
    Cylinder-oriented defaults: **identity** state scaling (physical ``x,y,z``), **L2-normalized**
    library columns before STLSQ so small-magnitude terms (e.g. ``x*z`` vs ``x``) are not drowned
    out. Uses ``pareto_dial=1`` (error-reduction end of the Pareto front) because **BIC** with
    normalized Θ often over-prunes the ``dz/dt`` row (dropping ``x*x``, ``y*y``); dial ``1`` keeps
    the dense model needed for Eq.~8. For a sparser model, set ``pareto_dial=0`` or
    ``pareto_pick_mode='bic'`` with ``pareto_dial=None`` explicitly.
    """
    return SINDyRunConfig(
        sindy_mode="pareto",
        pareto_pick_mode="bic",
        pareto_dial=1.0,
        scaler_kind="identity",
        savgol_window=7,
        savgol_poly=5,
        normalize_library_columns=True,
        use_physical_library=False,
        prefer_parsimony=False,
        alpha_ridge=1e-3,
        n_thresholds=50,
        threshold_min=1e-3,
        threshold_max=1.0,
        max_degree=3,
        max_interaction=2,
        remove_double_trig_terms=False,
        library_include_constant=True,
        equal_weight_per_target=True,
    )
