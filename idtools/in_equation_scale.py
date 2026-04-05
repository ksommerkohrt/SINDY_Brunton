"""
In-equation (within-equation) scaling: normalize library columns per target so that
single-term LS coefficients are O(1), then fit. Use to test whether balancing scale
within one equation (e.g. dy/dt = 28·x − x·z − y) helps recover small coefficients
without running the full SINDy pipeline.

Usage (no SINDy run needed — just need Theta and Y, e.g. from a previous fit or minimal build):
  from idtools.in_equation_scale import fit_target_with_in_equation_scaling
  coef_j, scale_factors = fit_target_with_in_equation_scaling(Theta, Y, target_index=1)
  # coef_j are in original (unscaled) space; compare to SINDy's coef_kept[:, j].
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def fit_target_with_in_equation_scaling(
    Theta: np.ndarray,
    Y: np.ndarray,
    target_index: int = 0,
    threshold: float = 0.0,
    feature_names: Optional[List[str]] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    For one target column Y[:, j], scale Theta columns so single-term LS coefficients
    are ~O(1), then fit Y[:, j] = Theta_scaled @ xi, return coefficients in original space.

    Does not run SINDy; use this to see if in-equation scaling helps (e.g. for dy/dt).
    Theta: (n_samples, n_features), Y: (n_samples, n_targets).
    threshold: optional STLSQ-style threshold (zero out coefficients smaller than this in scaled space).
    Returns: coef_original (n_features,), scale_factors (n_features,), info dict.
    """
    Theta = np.asarray(Theta, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    j = int(target_index)
    y_j = Y[:, j]

    # Single-term LS coefficient for each column k: b_k = (Theta[:,k]' @ y) / (Theta[:,k]' @ Theta[:,k])
    K = Theta.shape[1]
    b_single = np.zeros(K)
    for k in range(K):
        col = Theta[:, k]
        dot_tt = np.dot(col, col) + eps
        b_single[k] = np.dot(col, y_j) / dot_tt

    # Scale factors: make |b_single| ~ 1 so thresholding doesn't kill small terms
    scale_factors = np.maximum(np.abs(b_single), eps)

    # Theta_scaled[:, k] = Theta[:, k] * scale_factors[k]  =>  y = Theta_scaled @ xi  with xi[k] = b_k / scale_factors[k]
    Theta_scaled = Theta * scale_factors[None, :]

    # Fit in scaled space (LS or thresholded LS)
    xi_scaled, *_ = np.linalg.lstsq(Theta_scaled, y_j, rcond=None)
    if threshold > 0:
        xi_scaled = np.where(np.abs(xi_scaled) >= threshold, xi_scaled, 0.0)

    # Back to original space: y = Theta @ coef_original  =>  coef_original[k] = xi_scaled[k] * scale_factors[k]
    coef_original = xi_scaled * scale_factors

    info = {
        "b_single_term": b_single.copy(),
        "scale_factors": scale_factors.copy(),
        "xi_scaled": xi_scaled.copy(),
    }
    if feature_names is not None and len(feature_names) == K:
        info["feature_names"] = feature_names
    return coef_original, scale_factors, info


def compare_raw_vs_in_equation(
    Theta: np.ndarray,
    Y: np.ndarray,
    target_index: int = 1,
    target_name: str = "y",
    feature_names: Optional[List[str]] = None,
    threshold: float = 0.0,
) -> None:
    """
    Print a quick comparison: raw LS vs in-equation-scaled LS for one target.
    No SINDy run; use Theta and Y (e.g. from fit['Theta_clean'], fit['Y_phys'] after one run).
    With threshold=0, both give the same LS solution. With threshold>0, raw applies threshold
    to coefficients; in-equation applies threshold in scaled space then converts back, so
    small true terms are less likely to be zeroed out.
    """
    y_j = Y[:, target_index]
    coef_raw, *_ = np.linalg.lstsq(Theta, y_j, rcond=None)
    if threshold > 0:
        coef_raw_thresh = np.where(np.abs(coef_raw) >= threshold, coef_raw, 0.0)
    else:
        coef_raw_thresh = coef_raw

    coef_scaled, scale_factors, info = fit_target_with_in_equation_scaling(
        Theta, Y, target_index=target_index, threshold=threshold, feature_names=feature_names
    )

    K = Theta.shape[1]
    names = feature_names if feature_names and len(feature_names) == K else [f"k{k}" for k in range(K)]

    print(f"In-equation scaling check for target {target_name} (index {target_index}), threshold={threshold}:")
    print("  Raw LS (thresholded) vs in-equation-scaled LS (threshold in scaled space), both in original space:")
    print("  " + "-" * 72)
    for k in range(K):
        if np.abs(coef_raw_thresh[k]) > 1e-10 or np.abs(coef_scaled[k]) > 1e-10:
            print(f"    {names[k]:20s}  raw={coef_raw_thresh[k]:+12.5f}  in_eq={coef_scaled[k]:+12.5f}  (scale_fac={scale_factors[k]:.4f})")
    print("  " + "-" * 72)
    print("  Scale factors = |single-term LS coef|; in scaled space coefs are ~O(1), so threshold hits terms more fairly.")
