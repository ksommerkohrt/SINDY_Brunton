# idtools/excitation_report.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.std(x))


def _safe_ptp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.ptp(x))


def _coef_of_variation(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    mu = float(np.mean(x))
    sig = float(np.std(x))
    return float(sig / (abs(mu) + eps))


def _window_slices(n: int, n_windows: int) -> List[slice]:
    n_windows = int(max(1, n_windows))
    edges = np.linspace(0, n, n_windows + 1).astype(int)
    out: List[slice] = []
    for i in range(n_windows):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            continue
        out.append(slice(a, b))
    return out


def _corr_matrix(X: np.ndarray) -> np.ndarray:
    """
    X: (n_samples, n_dims)
    Returns abs correlation matrix (n_dims, n_dims), with nan -> 0.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_dims).")
    C = np.corrcoef(X, rowvar=False)
    C = np.abs(C)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    return C


def _top_corr_pairs(C: np.ndarray, names: Sequence[str], k: int = 20, threshold: float = 0.98):
    n = C.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            cij = float(C[i, j])
            if cij >= float(threshold):
                pairs.append((cij, names[i], names[j], i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[: int(k)]


def _svd_effective_rank(s: np.ndarray, rel_tol: float = 1e-8) -> int:
    s = np.asarray(s, dtype=float)
    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0:
        return 0
    return int(np.sum(s >= float(rel_tol) * float(s[0])))


def _condition_number_from_svd(s: np.ndarray, eps: float = 1e-300) -> float:
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return float("inf")
    s0 = float(s[0])
    slast = float(s[-1])
    if not np.isfinite(s0) or not np.isfinite(slast):
        return float("inf")
    return float(s0 / max(slast, float(eps)))


@dataclass(frozen=True)
class ExcitationReportConfig:
    n_windows: int = 4

    # Data excitation checks (on Z)
    data_flat_cv_threshold: float = 0.02
    data_low_std_threshold: float = 1e-3
    data_corr_threshold: float = 0.995

    # Theta excitation checks (on Theta, preferably Theta_n)
    theta_corr_threshold: float = 0.995
    theta_flat_cv_threshold: float = 0.02
    theta_cond_threshold: float = 1e6
    theta_effective_rank_frac_min: float = 0.4

    strict_data_corr_fail: bool = False
    strict_theta_corr_fail: bool = False

    # Optional: do not fail data_ok just because these variables are flat/low-std
    # (useful when refs/actuators/gust are intentionally constant in a run)
    ignore_data_flat_names: frozenset[str] = frozenset()
    ignore_data_low_std_names: frozenset[str] = frozenset()

def excitation_report(
    t: np.ndarray,
    Z: np.ndarray,
    names: Sequence[str],
    Theta: np.ndarray,
    feature_names: Sequence[str],
    input_names: Sequence[str] = (),
    Y: Optional[np.ndarray] = None,
    config: Optional[ExcitationReportConfig] = None,
) -> Dict:
    """
    Validate data excitation and Theta excitation.
    """
    if config is None:
        config = ExcitationReportConfig()

    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    names = list(names)
    feature_names = list(feature_names)

    if Z.ndim != 2:
        raise ValueError("Z must be 2D (n_samples, n_vars).")
    if Theta.ndim != 2:
        raise ValueError("Theta must be 2D (n_samples, n_features).")
    if Z.shape[0] != t.size:
        raise ValueError("t and Z must have same n_samples.")
    if Theta.shape[0] != t.size:
        raise ValueError("t and Theta must have same n_samples.")
    if len(names) != Z.shape[1]:
        raise ValueError("names length must match Z.shape[1].")
    if len(feature_names) != Theta.shape[1]:
        raise ValueError("feature_names length must match Theta.shape[1].")

    if Y is not None:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or Y.shape[0] != t.size:
            raise ValueError("If provided, Y must be 2D with same n_samples as t.")

    windows = _window_slices(t.size, config.n_windows)

    # -------------------------
    # Data excitation (Z)
    # -------------------------
    data_stats: Dict[str, Dict[str, float]] = {}
    flat_vars: List[str] = []
    low_std_vars: List[str] = []
    window_std: Dict[str, List[float]] = {}

    for j, nm in enumerate(names):
        col = Z[:, j]
        st = _safe_std(col)
        ptp = _safe_ptp(col)
        cv = _coef_of_variation(col)
        data_stats[nm] = {"std": float(st), "ptp": float(ptp), "cv": float(cv)}

        if np.isfinite(cv) and cv < config.data_flat_cv_threshold:
            flat_vars.append(nm)
        if np.isfinite(st) and st < config.data_low_std_threshold:
            low_std_vars.append(nm)

        wstds: List[float] = []
        for sl in windows:
            wstds.append(_safe_std(col[sl]))
        window_std[nm] = wstds

    Cdata = _corr_matrix(Z)

    input_set = set(input_names)
    top_data_pairs = _top_corr_pairs(Cdata, names, k=30, threshold=config.data_corr_threshold)
    
    # Identify correlations involving inputs (which are often constant/highly correlated)
    input_corr_pairs = [
        (c, a, b) for (c, a, b, _, _) in top_data_pairs 
        if (a in input_set or b in input_set)
    ]

    # State/measurement scale spread: max(std)/median(std) across columns. High ratio → z-score scaling can help.
    stds = [data_stats[nm]["std"] for nm in names]
    stds_arr = np.asarray(stds, dtype=float)
    stds_arr = np.where(np.isfinite(stds_arr) & (stds_arr > 0), stds_arr, np.nan)
    valid = stds_arr[np.isfinite(stds_arr)]
    if valid.size > 0:
        state_std_ratio = float(np.nanmax(stds_arr) / max(float(np.median(valid)), 1e-300))
    else:
        state_std_ratio = 1.0
    suggest_standard_scaling = bool(np.isfinite(state_std_ratio) and state_std_ratio > 100)

    data_flags = {
        "flat_vars_cv": flat_vars,
        "low_std_vars": low_std_vars,
        "high_corr_pairs": [(c, a, b) for (c, a, b, _, _) in top_data_pairs],
        "input_corr_pairs": input_corr_pairs,
    }

    # -------------------------
    # Theta excitation (Theta)
    # -------------------------
    theta_flat: List[str] = []
    theta_col_cv: List[float] = []

    for k, fn in enumerate(feature_names):
        col = Theta[:, k]
        cv = _coef_of_variation(col)
        theta_col_cv.append(float(cv))
        if np.isfinite(cv) and cv < config.theta_flat_cv_threshold:
            theta_flat.append(fn)

    try:
        s = np.linalg.svd(Theta, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.array([], dtype=float)

    cond = _condition_number_from_svd(s)
    eff_rank = _svd_effective_rank(s, rel_tol=1e-8)
    nfeat = int(Theta.shape[1])
    eff_rank_frac = float(eff_rank / max(1, nfeat))

    theta_window_cond: List[float] = []
    for sl in windows:
        Thw = Theta[sl, :]
        try:
            sw = np.linalg.svd(Thw, full_matrices=False, compute_uv=False)
            theta_window_cond.append(_condition_number_from_svd(sw))
        except np.linalg.LinAlgError:
            theta_window_cond.append(float("inf"))

    if Theta.shape[1] > 400:
        Ctheta = np.zeros((0, 0), dtype=float)
        top_theta_pairs = []
    else:
        Ctheta = _corr_matrix(Theta)
        top_theta_pairs = _top_corr_pairs(Ctheta, feature_names, k=30, threshold=config.theta_corr_threshold)

    # Column norm ratio: max/median (or max/min with floor). High ratio → normalization helps thresholding.
    col_norms = np.linalg.norm(Theta, axis=0)
    col_norms = np.where(col_norms <= 0, 1.0, col_norms)
    med_norm = float(np.median(col_norms))
    max_norm = float(np.max(col_norms))
    col_norm_ratio = float(max_norm / max(med_norm, 1e-300)) if nfeat else 1.0
    # Suggest normalization when columns span a large range (default True); when ratio is low, raw coeffs are comparable.
    suggest_normalize_library = bool(np.isfinite(col_norm_ratio) and col_norm_ratio > 1e3)

    theta_stats = {
        "n_samples": int(t.size),
        "n_features": int(nfeat),
        "cond": float(cond),
        "singular_values_head": [float(x) for x in s[:10]] if s.size else [],
        "effective_rank": int(eff_rank),
        "effective_rank_frac": float(eff_rank_frac),
        "window_cond": [float(x) for x in theta_window_cond],
        "flat_features_cv": theta_flat[:50],
        "flat_features_cv_count": int(len(theta_flat)),
        "col_norm_ratio": float(col_norm_ratio),
        "suggest_normalize_library": suggest_normalize_library,
    }

    theta_flags = {"high_corr_pairs": [(c, a, b) for (c, a, b, _, _) in top_theta_pairs]}

    # -------------------------
    # Pass/fail + warnings
    # -------------------------
    warnings: List[str] = []

    if low_std_vars:
        warnings.append(f"Data excitation: low-std variables: {low_std_vars}")
    if flat_vars:
        warnings.append(f"Data excitation: flat (low-CV) variables: {flat_vars}")
    if top_data_pairs:
        show = [(round(c, 4), a, b) for (c, a, b, _, _) in top_data_pairs[:10]]
        warnings.append(f"Data excitation: very high correlations detected (top): {show}")
    if input_corr_pairs:
        show = [(round(c, 4), a, b) for (c, a, b) in input_corr_pairs[:10]]
        warnings.append(f"Data excitation: ref/controller variables highly correlated with others (top): {show}")

    if not np.isfinite(cond) or cond > config.theta_cond_threshold:
        warnings.append(f"Theta excitation: poor conditioning, cond(Theta)={cond:.3e}")
    if eff_rank_frac < config.theta_effective_rank_frac_min:
        warnings.append(
            f"Theta excitation: low effective rank frac={eff_rank_frac:.2f} "
            f"(eff_rank={eff_rank}, nfeat={nfeat})"
        )
    if top_theta_pairs:
        show = [(round(c, 4), a, b) for (c, a, b, _, _) in top_theta_pairs[:10]]
        warnings.append(f"Theta excitation: highly collinear features remain (top): {show}")
    if len(theta_flat) > 0:
        warnings.append(f"Theta excitation: {len(theta_flat)} pseudo-constant features (low CV).")

    # Data OK logic (ignore ref/actuator channels by default)
    flat_core = [nm for nm in flat_vars if nm not in config.ignore_data_flat_names]
    low_std_core = [nm for nm in low_std_vars if nm not in config.ignore_data_low_std_names]
    data_ok = (len(low_std_core) == 0) and (len(flat_core) == 0)

    theta_ok = (
        np.isfinite(cond)
        and cond <= config.theta_cond_threshold
        and eff_rank_frac >= config.theta_effective_rank_frac_min
    )

    if config.strict_data_corr_fail and (len(top_data_pairs) > 0):
        data_ok = False
    if config.strict_theta_corr_fail and (len(top_theta_pairs) > 0):
        theta_ok = False

    pass_fail = {"data_ok": bool(data_ok), "theta_ok": bool(theta_ok), "ok": bool(data_ok and theta_ok)}

    return {
        "config": {
            **config.__dict__,
            "ignore_data_flat_names": sorted(list(config.ignore_data_flat_names)),
            "ignore_data_low_std_names": sorted(list(config.ignore_data_low_std_names)),
        },
        "data": {
            "stats": data_stats,
            "window_std": window_std,
            "corr_threshold": float(config.data_corr_threshold),
            "corr_matrix_abs": Cdata,
            "top_corr_pairs": [(float(c), a, b) for (c, a, b, _, _) in top_data_pairs],
            "flags": data_flags,
            "state_std_ratio": float(state_std_ratio),
            "suggest_standard_scaling": suggest_standard_scaling,
        },
        "theta": {
            "stats": theta_stats,
            "corr_threshold": float(config.theta_corr_threshold),
            "corr_matrix_abs": Ctheta,
            "top_corr_pairs": [(float(c), a, b) for (c, a, b, _, _) in top_theta_pairs],
            "col_cv": {feature_names[i]: float(theta_col_cv[i]) for i in range(len(feature_names))},
            "flags": theta_flags,
        },
        "pass_fail": pass_fail,
        "warnings": warnings,
    }


def format_excitation_report(rep: Dict, max_lines: int = 80) -> str:
    lines: List[str] = []

    pf = rep.get("pass_fail", {})
    lines.append("=== Excitation Report ===")
    lines.append(f"OK: {pf.get('ok')} (data_ok={pf.get('data_ok')}, theta_ok={pf.get('theta_ok')})")

    data = rep.get("data", {})
    lines.append("")
    lines.append("Data excitation:")

    flags = data.get("flags", {})
    low_std = flags.get("low_std_vars", [])
    flat_vars = flags.get("flat_vars_cv", [])
    if low_std:
        lines.append(f"- Low std vars: {low_std}")
    if flat_vars:
        lines.append(f"- Flat (low CV) vars: {flat_vars}")

    top_pairs = data.get("top_corr_pairs", [])[:10]
    if top_pairs:
        lines.append("- Top abs correlations (Z):")
        for c, a, b in top_pairs:
            lines.append(f"  {c:.4f}  {a} vs {b}")
    state_ratio = data.get("state_std_ratio", None)
    suggest_scale = data.get("suggest_standard_scaling", False)
    if state_ratio is not None:
        lines.append(
            f"- State std ratio (max/median)={state_ratio:.1f} → "
            f"{'suggest scaler_kind=standard (z-score)' if suggest_scale else 'maxabs OK (default)'}"
        )

    theta = rep.get("theta", {})
    tstats = theta.get("stats", {})
    lines.append("")
    lines.append("Theta excitation:")
    cond = tstats.get("cond", float("nan"))
    lines.append(
        f"- Theta shape: {tstats.get('n_samples')} x {tstats.get('n_features')}, "
        f"cond={cond:.3e}, "
        f"eff_rank={tstats.get('effective_rank')} "
        f"({tstats.get('effective_rank_frac'):.2f} frac)"
    )
    col_ratio = tstats.get("col_norm_ratio", None)
    suggest_norm = tstats.get("suggest_normalize_library", True)
    if col_ratio is not None:
        lines.append(
            f"- Library col norm ratio (max/median)={col_ratio:.1f} → "
            f"{'suggest normalize_library_columns=True' if suggest_norm else 'raw OK (paper-like default)'}"
        )

    flat_count = tstats.get("flat_features_cv_count", 0)
    if flat_count:
        lines.append(f"- Pseudo-constant features (low CV): {flat_count} (showing up to 10)")
        flat_list = tstats.get("flat_features_cv", [])[:10]
        for fn in flat_list:
            lines.append(f"  {fn}")

    top_theta_pairs = theta.get("top_corr_pairs", [])[:10]
    if top_theta_pairs:
        lines.append("- Top abs correlations (Theta columns):")
        for c, a, b in top_theta_pairs:
            lines.append(f"  {c:.4f}  {a} vs {b}")

    warns = rep.get("warnings", [])
    if warns:
        lines.append("")
        lines.append("Warnings:")
        for w in warns:
            lines.append(f"- {w}")

    if len(lines) > int(max_lines):
        lines = lines[: int(max_lines) - 1] + ["(truncated)"]

    return "\n".join(lines)
