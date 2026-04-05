"""
Compare discovered SINDy model to true equations (from symbolic model).
Use this to distinguish structure recovery from mere curve fitting.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from idtools.xi_true_from_sim_equations import xi_true_from_sim_equations
from idtools.xi_true_from_sim_equations import _projected_to_physical_units


def compare_to_truth(
    res: Dict[str, Any],
    ridge: float = 1e-8,
    zero_tol: float = 1e-10,
    verbose: bool = True,
    truth_Y_phys_override: Optional[np.ndarray] = None,
    truth_label: str = "override targets",
) -> Dict[str, Any]:
    """
    Compare pipeline result to true equations: print true vs discovered RHS and coefficient-level metrics.

    Uses the model's symbolic RHS (truth) and projects it onto the same library to get Xi_true
    in physical units, then compares to fit["coef_phys"]. Use the returned metrics to judge whether SINDy recovered
    structure (high coefficient correlation, low error on true terms) vs curve fitting (good R² but
    poor coefficient match).

    Parameters
    ----------
    truth_Y_phys_override : np.ndarray, optional
        Optional physical target matrix with shape (n_samples, n_targets). When provided,
        this is used as the ground-truth target instead of evaluating `model.rhs_lambdified`.
        Useful when fitting transformed targets like [L/q_dyn, M/q_dyn, N/q_dyn] rather than
        the model's native derivative targets.
    truth_label : str
        Label shown in verbose output when `truth_Y_phys_override` is used.

    Returns
    -------
    dict with keys: xi_true, xi_disc, coef_correlation, rel_errors, max_rel_error_nz, summary_lines
    """
    fit = res["fit"]
    model = res["model"]
    Z = res["Z"]
    lib = fit["library"]
    scaler = fit["scaler"]
    kept_idx = np.asarray(fit["kept_idx"], dtype=int)
    coef_disc_kept = np.asarray(fit.get("coef_phys", fit.get("coef_kept")), dtype=float)
    kept_names = fit["kept_names"]
    target_names = fit["target_names"]

    if truth_Y_phys_override is None:
        try:
            xi_true_full, truth_metrics = xi_true_from_sim_equations(
                model=model,
                lib=lib,
                scaler=scaler,
                Z_phys=Z,
                kept_idx=None,
                ridge=ridge,
                use_physical_library=fit.get("use_physical_library", False),
            )
        except Exception as e:
            if verbose:
                print(f"Could not compute Xi_true: {e}")
            return {"xi_true": None, "error": str(e)}

        xi_true_full = np.asarray(xi_true_full, dtype=float)
        if xi_true_full.ndim != 2:
            return {"xi_true": None, "error": "xi_true_from_sim_equations returned non-2D xi_true."}
        n_full, n_targ = xi_true_full.shape

        # Expand discovered coefficients from kept rows to full library rows.
        coef_disc_full = np.zeros((n_full, n_targ), dtype=float)
        coef_disc_kept = np.asarray(coef_disc_kept, dtype=float)
        if coef_disc_kept.ndim == 1:
            coef_disc_kept = coef_disc_kept.reshape(-1, 1)
        for k, lib_idx in enumerate(kept_idx):
            if 0 <= int(lib_idx) < n_full and k < coef_disc_kept.shape[0]:
                coef_disc_full[int(lib_idx), :] = coef_disc_kept[k, :]

        # Union support for fair structure comparison.
        tiny = 1e-8
        row_mag = np.maximum(
            np.max(np.abs(xi_true_full), axis=1),
            np.max(np.abs(coef_disc_full), axis=1),
        )
        union_mask = row_mag > tiny
        if not np.any(union_mask):
            union_mask[:] = True
        xi_true = xi_true_full[union_mask, :]
        coef_disc = coef_disc_full[union_mask, :]
    else:
        Y_true = np.asarray(truth_Y_phys_override, dtype=float)
        Theta_clean = np.asarray(fit["Theta_clean"], dtype=float)
        if Y_true.ndim != 2:
            return {"xi_true": None, "error": "truth_Y_phys_override must be 2D (n_samples, n_targets)"}
        if Y_true.shape[0] != Theta_clean.shape[0]:
            return {
                "xi_true": None,
                "error": f"truth_Y_phys_override row mismatch: {Y_true.shape[0]} vs {Theta_clean.shape[0]}",
            }
        ck = np.asarray(coef_disc_kept, dtype=float)
        if ck.ndim == 1:
            ck = ck.reshape(-1, 1)
        if Y_true.shape[1] != ck.shape[1]:
            return {
                "xi_true": None,
                "error": f"truth_Y_phys_override target mismatch: {Y_true.shape[1]} vs {ck.shape[1]}",
            }
        n_feat = Theta_clean.shape[1]
        A = Theta_clean.T @ Theta_clean + ridge * np.eye(n_feat)
        B = Theta_clean.T @ Y_true
        xi_true_kept = np.linalg.solve(A, B)
        xi_true = _projected_to_physical_units(
            xi_true_kept,
            lib=lib,
            scaler=scaler,
            kept_idx=kept_idx,
            use_physical_library=fit.get("use_physical_library", False),
        )
        coef_disc = np.asarray(coef_disc_kept, dtype=float)
        resid = (Theta_clean @ xi_true_kept) - Y_true
        truth_metrics = {
            "rmse": np.sqrt(np.mean(resid**2, axis=0)).tolist(),
            "max_err": float(np.max(np.abs(resid))),
            "truth_source": truth_label,
        }

    xi_true = np.asarray(xi_true, dtype=float)
    coef_disc = np.asarray(coef_disc, dtype=float)
    n_feat, n_targ = coef_disc.shape

    # Coefficient correlation (flatten): are true and discovered coefficient vectors aligned?
    xt_flat = xi_true.ravel()
    xd_flat = coef_disc.ravel()
    if np.std(xt_flat) > 1e-14 and np.std(xd_flat) > 1e-14:
        coef_correlation = float(np.corrcoef(xt_flat, xd_flat)[0, 1])
    else:
        coef_correlation = float("nan")

    # Relative errors on terms that are nonzero in the true model (structure recovery)
    rel_errors = np.full_like(xi_true, np.nan)
    nz_true = np.abs(xi_true) > zero_tol
    rel_errors[nz_true] = np.abs(coef_disc[nz_true] - xi_true[nz_true]) / (
        np.abs(xi_true[nz_true]) + zero_tol
    )
    max_rel_error_nz = float(np.nanmax(rel_errors)) if np.any(nz_true) else 0.0
    mean_rel_error_nz = float(np.nanmean(rel_errors)) if np.any(nz_true) else 0.0

    summary_lines = []

    if verbose:
        print("\n" + "=" * 60)
        print("  TRUE vs DISCOVERED (structure recovery vs curve fitting)")
        print("=" * 60)

        if truth_Y_phys_override is None:
            print("\nTrue equations (from model):")
            for j, name in enumerate(target_names):
                expr = model.rhs_symbolic[j] if hasattr(model, "rhs_symbolic") and model.rhs_symbolic else f"target {j}"
                print(f"  d{name}/dt = {expr}")
        else:
            print(f"\nTrue targets source: {truth_label}")
            print("  (Projected from provided truth_Y_phys_override onto the same kept library features)")

        print("\nDiscovered equations:")
        for name, expr in fit["equations"].items():
            print(f"  d{name}/dt = {expr}")

        print("\nCoefficient comparison (physical units, same library terms):")
        print(f"  Correlation(true, discovered) = {coef_correlation:.4f}")
        print(f"  Mean relative error (on true nonzero terms) = {mean_rel_error_nz:.4f}")
        print(f"  Max relative error (on true nonzero terms)  = {max_rel_error_nz:.4f}")

        summary_lines.append(f"Correlation(true, discovered) = {coef_correlation:.4f}")
        summary_lines.append(f"Mean rel error (true nonzero terms) = {mean_rel_error_nz:.4f}")
        summary_lines.append(f"Max rel error (true nonzero terms)  = {max_rel_error_nz:.4f}")

        # Short interpretation
        if coef_correlation > 0.9 and max_rel_error_nz < 0.2:
            print("\n  -> Strong coefficient agreement: suggests structure recovery, not just curve fitting.")
        elif coef_correlation > 0.7:
            print("\n  -> Moderate agreement: some structure recovered; check terms with large rel error.")
        else:
            print("\n  -> Weak coefficient agreement: good fit may be curve fitting; inspect term list.")
        print("=" * 60 + "\n")

    # Single boolean: strong structure recovery (used by diagnostic_suite and sindy_success)
    structure_ok = bool(
        coef_correlation > 0.9 and (max_rel_error_nz < 0.2 if np.isfinite(max_rel_error_nz) else False)
    )

    return {
        "xi_true": xi_true,
        "xi_disc": coef_disc,
        "coef_correlation": coef_correlation,
        "rel_errors": rel_errors,
        "mean_rel_error_nz": mean_rel_error_nz,
        "max_rel_error_nz": max_rel_error_nz,
        "structure_ok": structure_ok,
        "truth_metrics": truth_metrics,
        "summary_lines": summary_lines,
    }
