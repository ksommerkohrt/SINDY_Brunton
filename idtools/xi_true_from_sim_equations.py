from __future__ import annotations

from typing import Dict, Sequence, Tuple, Any

import numpy as np
import sympy as sp

from idtools.preprocess import affine_from_sklearn_scaler
from sindy.library import SINDyLibrary  # adjust import path if needed


def _projected_to_physical_units(
    Xi_proj: np.ndarray,
    lib: Any,
    scaler: Any,
    kept_idx: np.ndarray | None,
    use_physical_library: bool,
) -> np.ndarray:
    """
    Convert projected coefficients from regression/library space to physical units.

    Matches the conversion used in `sindy.pipeline.fit_sindy_main` for `coef_phys`.
    """
    Xi_proj = np.asarray(Xi_proj, dtype=float)
    if kept_idx is None:
        return Xi_proj

    kept_idx = np.asarray(kept_idx, dtype=int).reshape(-1)
    scale_arr = np.asarray(scaler.scale_, float).reshape(-1)
    offset_arr = np.asarray(getattr(scaler, "mean_", np.zeros_like(scale_arr)), float).reshape(-1)

    feature_scales = np.ones(len(kept_idx), dtype=float)
    if not use_physical_library:
        for k, lib_idx in enumerate(kept_idx):
            ci = lib.combo_index_for_library_column(int(lib_idx))
            if ci is None:
                continue
            combo = lib.valid_combos[ci]
            for ai in combo:
                atom = lib.active_atoms[ai]
                if atom.input_space == "scaled":
                    feature_scales[k] *= scale_arr[atom.idx]

    Xi_phys = Xi_proj / np.maximum(feature_scales[:, None], 1e-20)

    # StandardScaler mean offsets contribute to the constant term in physical coordinates.
    if np.any(offset_arr != 0):
        k_const_list = [
            k
            for k, j in enumerate(kept_idx)
            if str(lib.feature_names[int(j)]).strip() == "1"
        ]
        if len(k_const_list) > 0:
            k_const = int(k_const_list[0])
            for k, lib_idx in enumerate(kept_idx):
                ci = lib.combo_index_for_library_column(int(lib_idx))
                if ci is None or k == k_const:
                    continue
                combo = lib.valid_combos[ci]
                factor = 1.0
                has_scaled = False
                for ai in combo:
                    atom = lib.active_atoms[ai]
                    if atom.input_space == "scaled":
                        has_scaled = True
                        s, o = scale_arr[atom.idx], offset_arr[atom.idx]
                        factor *= (-o / s) if s != 0 else 0.0
                if has_scaled and factor != 0:
                    Xi_phys[k_const, :] += Xi_proj[k, :] * factor

    return Xi_phys


def xi_true_from_sim_equations(
    model: Any,
    lib: Any,
    scaler: Any,
    Z_phys: np.ndarray,
    kept_idx: np.ndarray = None,
    ridge: float = 1e-8,
    use_physical_library: bool = False,
    return_physical_units: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    True Numerical Projector: No SymPy parsing, just raw evaluation and Ridge.
    use_physical_library: if True, build Theta from Z_phys (matches fit when use_physical_library was used).
    return_physical_units: if True, return coefficients in the same physical units as fit["coef_phys"].
    """
    Z_phys = np.asarray(Z_phys, dtype=float)
    n_samples = Z_phys.shape[0]

    # 1. Map columns to match the exact order the lambdified functions expect
    symbol_to_idx = {name: i for i, name in enumerate(model.measured_names)}
    ordered_indices = [symbol_to_idx[s.name] for s in model.all_symbols]
    Z_ordered = Z_phys[:, ordered_indices]

    # 2. Evaluate the "Perfect" derivatives (F_true)
    F_true = np.zeros((n_samples, len(model.rhs_lambdified)))
    for i, f_num in enumerate(model.rhs_lambdified):
        # We wrap in np.nan_to_num to handle any potential math errors
        val = f_num(*Z_ordered.T)
        F_true[:, i] = np.nan_to_num(val, nan=0.0)

    # 3. Build the Library Matrix (Theta) using the same scaling as discovery
    from idtools.preprocess import affine_from_sklearn_scaler
    aff = affine_from_sklearn_scaler(scaler)
    if use_physical_library:
        Theta_full = lib.transform(Z_scaled=Z_phys, Z_phys=Z_phys)
    else:
        Z_scaled = aff.transform(Z_phys)
        Theta_full = lib.transform(Z_scaled=Z_scaled, Z_phys=Z_phys)
    
    # Filter to only the features used in the discovery run
    Theta = Theta_full[:, kept_idx] if kept_idx is not None else Theta_full

    # 4. The Numerical Projection (Ridge Regression)
    # This identifies the 'True' weights for the library features
    n_feat = Theta.shape[1]
    A = Theta.T @ Theta + ridge * np.eye(n_feat)
    B = Theta.T @ F_true
    Xi_true = np.linalg.solve(A, B)
    Xi_out = (
        _projected_to_physical_units(
            Xi_true,
            lib=lib,
            scaler=scaler,
            kept_idx=kept_idx,
            use_physical_library=use_physical_library,
        )
        if return_physical_units
        else Xi_true
    )

    # 5. Metrics for the dashboard
    resid = (Theta @ Xi_true) - F_true
    rmse = np.sqrt(np.mean(resid**2, axis=0))

    return Xi_out, {"rmse": rmse.tolist(), "max_err": float(np.max(np.abs(resid)))}