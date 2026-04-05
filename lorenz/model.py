"""
Lorenz system model and bridge to SINDy pipeline (SymPy RHS + SINDySystemModel).
"""
from __future__ import annotations

import sympy as sp

from sindy.pipeline import SINDySystemModel


def build_lorenz_model(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> SINDySystemModel:
    """
    Build SINDySystemModel for Lorenz: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z.
    """
    x, y, z = sp.symbols("x y z", real=True)
    state_syms = [x, y, z]

    # Symbolic RHS (with symbolic params for clarity, then substitute)
    sig, r, b = sp.symbols("sigma rho beta", real=True, positive=True)
    rhs_sym = [
        sig * (y - x),
        x * (r - z) - y,
        x * y - b * z,
    ]
    rhs_num = [e.subs([(sig, sigma), (r, rho), (b, beta)]) for e in rhs_sym]

    rhs_lambdified = [
        sp.lambdify(state_syms, e, modules="numpy") for e in rhs_num
    ]

    return SINDySystemModel(
        measured_names=["x", "y", "z"],
        target_names=["x", "y", "z"],
        target_indices=[0, 1, 2],
        all_symbols=state_syms,
        rhs_lambdified=rhs_lambdified,
        rhs_symbolic=rhs_num,
    )


def lorenz_sindy_config_bic_standard():
    """
    Default SINDy settings for Lorenz: Pareto STLSQ sweep with **BIC** model pick.

    **No state/derivative scaling** before regression: ``scaler_kind="identity"`` (raw ``x,y,z`` and
    exact ``Z_dot_phys``). **No library column L2 normalization** (``normalize_library_columns=False``)
    so STLSQ runs on the same physical-scale Θ as the Brunton-style Lorenz demo.
    """
    from sindy.pipeline import SINDyRunConfig

    return SINDyRunConfig(
        sindy_mode="pareto",
        pareto_pick_mode="bic",
        scaler_kind="identity",
        savgol_window=7,
        savgol_poly=5,
        normalize_library_columns=False,
        use_physical_library=False,
        prefer_parsimony=False,
        alpha_ridge=1e-3,
        n_thresholds=50,
        threshold_min=1e-3,
        threshold_max=1.0,
        remove_double_trig_terms=False,
        library_include_constant=True,
        equal_weight_per_target=True,
    )


def lorenz_sindy_config_ensemble_bic_standard(*, ensemble_B: int = 50, ensemble_frac: float = 0.8):
    """
    Same preprocessing (no state scaling, no Θ column norm) and **BIC** Pareto pick as
    :func:`lorenz_sindy_config_bic_standard`, but **bootstrap Pareto ensemble**: each of ``ensemble_B``
    draws fits STLSQ on a ``ensemble_frac``
    row subsample, picks the BIC-best point on the front, then coefficients are **median-aggregated**
    across draws (robust to collinearity / conditioning).
    """
    from dataclasses import replace

    base = lorenz_sindy_config_bic_standard()
    return replace(
        base,
        sindy_mode="pareto_ensemble",
        ensemble_B=int(ensemble_B),
        ensemble_frac=float(ensemble_frac),
    )
