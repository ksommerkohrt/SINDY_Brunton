# sindy/pipeline.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict, Sequence, Tuple, Callable

import numpy as np
import sympy as sp

from idtools.preprocess import (
    affine_from_sklearn_scaler,
    preprocess_timeseries,
    normalize_columns,   # keep this
    estimate_dt,
)

from dataclasses import replace

from idtools.excitation_report import excitation_report, format_excitation_report
from idtools.auto_config import (
    recommend_config,
    flatten_scout_from_excitation,
    compute_derivative_agreement,
    infer_problem_class,
    recommend_consensus_shortlist,
    get_config_reason,
)
from sindy import SINDyLibrary

from sindy.fit import (
    drop_constant_like_columns,
    prefer_parsimony,
    AdaptiveSTLSQ,       # only if you still use it elsewhere; not needed in fit_sindy_first4 now
    SINDyFitConfig,
    fit_sindy,
)




# ----------------------------
# Helpers
# ----------------------------
def _getattr_any(obj: Any, names: Sequence[str]):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"Missing attrs {list(names)}; available={sorted(dir(obj))}")


def _debug_metrics_from_err(Y_true: np.ndarray, Y_hat: np.ndarray, eps: float = 1e-12) -> Dict:
    Y_true = np.asarray(Y_true, dtype=float)
    Y_hat = np.asarray(Y_hat, dtype=float)
    if Y_true.shape != Y_hat.shape:
        raise ValueError(f"Shape mismatch: Y_true {Y_true.shape} vs Y_hat {Y_hat.shape}")

    err = Y_hat - Y_true
    mse_by = np.mean(err**2, axis=0)
    rmse_by = np.sqrt(mse_by)

    var_by = np.var(Y_true, axis=0) + eps
    r2_by = 1.0 - mse_by / var_by

    rmse_all = float(np.sqrt(np.mean(err**2)))
    r2_mean = float(np.mean(r2_by))

    return {
        "err": err,
        "mse_by_state": mse_by,
        "rmse_by_state": rmse_by,
        "r2_by_state": r2_by,
        "rmse": rmse_all,
        "r2_mean": r2_mean,
    }

# ----------------------------
# 1. System Metadata Container
# ----------------------------
@dataclass
class SINDySystemModel:
    """
    Metadata container to bridge physical simulations to the SINDy pipeline.
    """
    measured_names: List[str]   # All state names in the simulation (e.g. 17 states)
    target_names: List[str]     # Names of states we want to discover (e.g. V, alpha, gamma, Q)
    target_indices: List[int]   # Indices of target states in the simulation vector
    all_symbols: List[Any]      # Plain SymPy symbols matching measured_names
    rhs_lambdified: List[Any] = None # Numeric functions for truth comparison
    rhs_symbolic: List[Any] = None   # Pure symbolic equations for truth comparison
# ----------------------------
# Config
# ----------------------------
# Full reference: docs/SOLVER_OPTIONS.md
#
# Solver modes (sindy_mode): "pareto" | "pareto_ensemble" | "bayes_map" | "bayes_map_ensemble"
#   pareto: single STLSQ threshold sweep → pick one model (BIC, knee, or last by config).
#   pareto_ensemble: same + bootstrap median over ensemble_B runs.
#   bayes_map: single MAP (L1) fit, no sweep.
#   bayes_map_ensemble: MAP on ensemble_B subsamples, median coefs.
#
# Main knobs affecting SINDy success rate (tune these when structure recovery is poor):
#   1. Sparsity tradeoff  -> pareto_dial [0,1], or pareto_pick_mode ("bic"/"knee"/"last"), n_thresholds, sparsity_bias.
#      pareto_dial: 0 = prioritize sparsity, 1 = prioritize error reduction; overrides pareto_pick_mode when set.
#      To help recover borderline/collinear true terms: dial=1 or "last", and/or larger n_thresholds; "knee"/dial=0 can drop weak terms.
#   2. Collinearity cutoff -> collinear_threshold; prefer_parsimony (if True, :func:`sindy.fit.prefer_parsimony`
#      resolves |corr|>threshold pairs before fit—true terms that are collinear on data can be lost; False keeps all Θ columns)
#   3. Constant/correlation cutoff -> constant_cv_thresh (drop near-constant columns; lower = stricter)
#   4. Library granularity -> budget (custom_budget), max_degree, max_interaction
#
# Tuning strategies (knobs are not independent; use one of these approaches):
#   - Sequential: fix library first, then tune cutoffs, then sparsity (or: coarse library -> add terms if
#     true terms missing; tighten collinear/constant cutoffs if wrong terms appear; then try knee vs last).
#   - Small grid: run a few (library, collinear_threshold, pareto_pick_mode) combinations and compare
#     structure metrics (e.g. coef_correlation, max_rel_error_nz from compare_to_truth). Use idtools.config_sweep.
#   - Consensus across configs: run multiple configs and keep terms that appear in most runs (or with
#     high inclusion probability); drop terms that are unstable across cutoffs/sparsity choices.
# ----------------------------
@dataclass
class SINDyRunConfig:
    seed: int = 0

    savgol_window: int = 7
    savgol_poly: int = 3
    scaler_kind: str = "maxabs"  # "maxabs" | "standard" | "identity" | "none" (no state scale)

    max_degree: int = 3
    max_interaction: int = 2
    collinear_threshold: float = 0.995  # (2) |corr| above this → prefer_parsimony resolves same-root pairs

    # pruning
    drop_constant_like: bool = True
    constant_cv_thresh: float = 1e-3   # (3) drop columns with CV below this (near-constant)
    # If True, resolve |corr| > collinear_threshold via :func:`sindy.fit.prefer_parsimony`
    # (complexity / parsimony; not first-column order). When False, alpha_ridge is auto-increased (min 1e-3).
    prefer_parsimony: bool = False

    # STLSQ / pareto. Ridge (L2) stabilizes least-squares when columns are collinear. When
    # prefer_parsimony=False we use at least 1e-3 so Lorenz-style dynamics can converge.
    alpha_ridge: float = 1e-6
    n_thresholds: int = 50   # More + finer range can help Pareto include denser models (recover weak/collinear terms)
    threshold_min: float = 1e-3
    threshold_max: float = 1.0
    
    sindy_mode: str = "pareto"          # "pareto", "pareto_ensemble", "bayes_map", "bayes_map_ensemble"
    ensemble_B: int = 100
    ensemble_frac: float = 0.8
    map_lam: float = 1e-3
    map_sigma2: float = 1.0

    pareto_pick_mode: str = "bic"  # "bic" | "per_target_bic" = min BIC over threshold sweep; "per_target_knee"/"knee" = Pareto knee; "last" = densest Pareto point
    # Continuous dial [0, 1]: 0 = prioritize sparsity, 1 = prioritize error reduction. If set, overrides pareto_pick_mode.
    pareto_dial: Optional[float] = None
    pareto_use_log: bool = True
    pareto_lam: float = 0.0
    # General best practice: weight each target equation equally in MSE so scale doesn't
    # let high-variance equations dominate and treat others like trash.
    equal_weight_per_target: bool = True
    # BIC MSE floor from target scale: max(fraction * var(Y), epsilon). See :func:`sindy.pareto.pick_by_bic`.
    bic_mse_variance_fraction: float = 1e-3
    bic_mse_floor_epsilon: float = 1e-12

    pareto_plot: bool = True
    pareto_plot_path: str = "outputs/pareto_frontiers.png"
    sparsity_bias: float = 0.0

    # Data-processing defaults = paper-like (no scaling/norm); diagnostics can suggest normalization or z-score.
    normalize_library_columns: bool = False  # Default False (paper-like). True = L2-normalize library columns before regression.
    use_physical_library: bool = False       # If True, build Theta from Z_phys for all terms (paper uses raw state).
    single_threshold: Optional[float] = None  # If set, use this threshold only (e.g. 0.05) instead of Pareto sweep.
    # Optional structural filter: drop features containing two or more trig atoms
    # (e.g. sin(alpha)*cos(gamma)) before fitting.
    remove_double_trig_terms: bool = True

    # Small-angle / identifiability: can drop linear θ when highly correlated with sin(θ) on data
    # (makes α ≈ sin α in practice and hides divergence at larger angles). Default off so collinear
    # libraries (α vs sin α) stay in Θ for Ridge/STLSQ to resolve; see _apply_small_angle_preference.
    small_angle_preference: bool = False
    small_angle_drop_linear_for_sin: bool = True  # drop raw θ if |corr(θ,sin θ)| is high
    small_angle_drop_cos_near_one: bool = True  # drop cos(θ) if ≈ constant / collinear with bias
    small_angle_drop_control_for_cos_product: bool = True  # drop C if C ≈ C*cos(θ) (keep product)
    small_angle_sin_corr_thresh: float = 0.99
    small_angle_cos_const_corr_thresh: float = 0.995  # |corr(cos θ, 1)|
    small_angle_cos_min_std: float = 1e-4  # drop cos θ if std below this
    small_angle_interaction_corr_thresh: float = 0.99
    # If None, use all measured_names (angles + controls filtered by presence in library).
    small_angle_angle_names: Optional[List[str]] = None
    small_angle_control_names: Optional[List[str]] = None  # e.g. ["T_act", "elv_act"]

    # Optional: curate polynomial/trig library by feature name before Θ is built (classic SINDy
    # only; optional ``library_keep_feature`` callback can prune Θ by feature name).
    library_keep_feature: Optional[Callable[[str], bool]] = None
    # If False, :class:`~sindy.library.SINDyLibrary` omits the leading ``"1"`` column (no intercept).
    # Option 2 NF sets False: Newton–Euler specific force should not include a levitation bias at rest.
    library_include_constant: bool = True


# ----------------------------
# Consensus dimensions (Solver × Sparsity × Data)
# ----------------------------
# Solver: 4 options. Sparsity: 6 options (target-based 3 + single-threshold 3); only for Pareto/Ensemble.
# Data: 5 options. Names are Solver_Sparsity_Data or Solver_Data (Bayes has no sparsity label).

SOLVERS = ("Pareto", "Ensemble", "BayesMAP", "BayesEnsemble")
SPARSITY_OPTS = (
    "target_sparsest",   # dial=0
    "target_bic",        # min BIC over threshold sweep
    "target_knee",       # per_target_knee
    "target_densest",    # last
    "single_sparsest",   # single thresh → fewer terms (0.5)
    "single_knee",       # single thresh 0.05
    "single_densest",    # single thresh 0.005 → more terms
)
DATA_OPTS = ("baseline", "collinear_keep", "scale_zscore", "lib_raw", "lib_physical")

# Single-threshold factors for single_* sparsity (multiplied by max_coef at fit time so they differ)
# High factor = sparser, low = denser. Same scale as full Pareto sweep (logspace * max_coef).
SINGLE_THRESH = {"single_sparsest": 0.5, "single_knee": 0.05, "single_densest": 0.005}


def _config_from_dims(
    solver: str,
    sparsity: Optional[str],
    data: str,
) -> SINDyRunConfig:
    """Build SINDyRunConfig from Solver × Sparsity × Data. Sparsity is None for Bayes solvers."""
    prefer_parsimony = data != "collinear_keep"
    scaler_kind = "standard" if data == "scale_zscore" else "maxabs"
    # Brunton/paper-like = no library column normalization. Only turn on for options that explicitly test it.
    normalize_library_columns = data in ("collinear_keep", "scale_zscore")
    use_physical_library = data == "lib_physical"
    data_kw = dict(prefer_parsimony=prefer_parsimony, scaler_kind=scaler_kind,
                  normalize_library_columns=normalize_library_columns, use_physical_library=use_physical_library)

    if solver == "BayesMAP":
        return SINDyRunConfig(sindy_mode="bayes_map", map_lam=1e-3, **data_kw)
    if solver == "BayesEnsemble":
        return SINDyRunConfig(sindy_mode="bayes_map_ensemble", ensemble_B=50, **data_kw)

    # Pareto or Ensemble: need sparsity (default target_densest if None)
    sp = sparsity or "target_densest"
    n_thr = 100
    if sp == "target_sparsest":
        base = SINDyRunConfig(sindy_mode="pareto", n_thresholds=n_thr, pareto_dial=0.0)
    elif sp == "target_bic":
        base = SINDyRunConfig(sindy_mode="pareto", n_thresholds=n_thr, pareto_pick_mode="bic")
    elif sp == "target_knee":
        base = SINDyRunConfig(sindy_mode="pareto", n_thresholds=n_thr, pareto_pick_mode="per_target_knee")
    elif sp == "target_densest":
        base = SINDyRunConfig(sindy_mode="pareto", n_thresholds=n_thr, pareto_pick_mode="last")
    elif sp in ("single_sparsest", "single_knee", "single_densest"):
        single_thr = SINGLE_THRESH[sp]
        base = SINDyRunConfig(sindy_mode="pareto", n_thresholds=1, single_threshold=single_thr, pareto_pick_mode="last")
    else:
        base = SINDyRunConfig(sindy_mode="pareto", n_thresholds=n_thr, pareto_pick_mode="last")

    cfg = SINDyRunConfig(
        sindy_mode="pareto_ensemble" if solver == "Ensemble" else "pareto",
        ensemble_B=50 if solver == "Ensemble" else 100,
        n_thresholds=base.n_thresholds, pareto_pick_mode=base.pareto_pick_mode,
        pareto_dial=base.pareto_dial, single_threshold=base.single_threshold,
        **data_kw,
    )
    return cfg


def _method_name(solver: str, sparsity: Optional[str], data: str) -> str:
    """Unique name: Solver_Sparsity_Data or Solver_Data for Bayes."""
    if sparsity is None:
        return f"{solver}_{data}"
    return f"{solver}_{sparsity}_{data}"


# ----------------------------
# Fit
# ----------------------------
def fit_sindy_main(
    Z_phys: np.ndarray,
    measured_names: Sequence[str],
    target_names: Sequence[str],
    target_indices: Sequence[int],
    dt: float,
    config: SINDyRunConfig,
    budget: Optional[Dict[str, Dict[str, int]]] = None,
    # New: Aligned feature space from runner
    pre_theta: Optional[np.ndarray] = None,
    pre_names: Optional[List[str]] = None,
    pre_sp: Optional[List[sp.Expr]] = None,
    pre_idx: Optional[np.ndarray] = None,
    Z_dot_phys: Optional[np.ndarray] = None,
    use_physical_library: bool = False,
) -> Dict:
    Z_phys = np.asarray(Z_phys, float)
    dt = float(dt)

    # 1. Differentiation and Scaling (use exact derivatives if provided, else SavGol)
    prep = preprocess_timeseries(
        Z_phys, dt=dt, scaler_kind=getattr(config, "scaler_kind", "maxabs"),
        savgol_window=config.savgol_window, savgol_poly=config.savgol_poly, deriv=1,
        X_dot_phys=Z_dot_phys,
    )
    scaler = prep.scaler
    Y_scaled = prep.X_dot_scaled[:, target_indices]
    
    # Target units: Physical derivative
    target_scales = np.asarray(scaler.scale_)[target_indices]
    Y_phys = Y_scaled * target_scales[None, :]

    # 2. Shared Ballot Logic
    _lkf = getattr(config, "library_keep_feature", None)
    if pre_theta is not None:
        Theta_clean, kept_names, kept_sp, kept_idx = pre_theta, pre_names, pre_sp, pre_idx
        collinear_dropped = []  # caller (pipeline) will attach its own list to fit
        lib = SINDyLibrary(
            measured_names=measured_names,
            max_degree=config.max_degree,
            max_interaction=config.max_interaction,
            custom_budget=budget,
            keep_feature=_lkf,
            include_constant_term=getattr(config, "library_include_constant", True),
        )
    else:
        # Internal fallback if called standalone
        lib = SINDyLibrary(
            measured_names=measured_names,
            max_degree=config.max_degree,
            max_interaction=config.max_interaction,
            custom_budget=budget,
            keep_feature=_lkf,
            include_constant_term=getattr(config, "library_include_constant", True),
        )
        Theta_raw = lib.transform(Z_scaled=prep.X_scaled, Z_phys=Z_phys)
        T_w, keep, n_w, sp_w = drop_constant_like_columns(
            Theta_raw, lib.feature_names, lib.sp_features,
            cv_thresh=config.constant_cv_thresh,
        )
        T_w, n_w, sp_w, keep, _ = _apply_small_angle_preference(
            T_w, n_w, sp_w, keep, config, lib.measured_names, Z_phys=Z_phys
        )
        if getattr(config, "remove_double_trig_terms", False):
            trig_filtered_cols: List[int] = []
            trig_filtered_sp: List[Any] = []
            trig_filtered_keep: List[int] = []
            for col_i, (nm, sp_expr, kidx) in enumerate(zip(n_w, sp_w, keep)):
                n_trig = str(nm).count("sin(") + str(nm).count("cos(")
                if n_trig >= 2:
                    continue
                trig_filtered_cols.append(col_i)
                trig_filtered_sp.append(sp_expr)
                trig_filtered_keep.append(int(kidx))
            if trig_filtered_cols:
                T_w = T_w[:, trig_filtered_cols]
                n_w = [n_w[i] for i in trig_filtered_cols]
                sp_w = trig_filtered_sp
                keep = np.asarray(trig_filtered_keep, dtype=int)
            elif getattr(config, "library_include_constant", True):
                T_w = np.ones((T_w.shape[0], 1), dtype=float)
                n_w = ["1"]
                sp_w = [sp.Integer(1)]
                keep = np.asarray([0], dtype=int)
            else:
                raise ValueError(
                    "remove_double_trig_terms removed every library column; with "
                    "library_include_constant=False (e.g. Option 2 NF) a synthetic constant Θ column "
                    "is not allowed. Relax the trig filter or the library."
                )
        if getattr(config, "prefer_parsimony", False):
            Theta_clean, rel, kept_names, kept_sp, collinear_dropped = prefer_parsimony(
                T_w, n_w, sp_w, threshold=config.collinear_threshold
            )
            kept_idx = np.asarray(keep)[rel]
        else:
            Theta_clean, kept_names, kept_sp = T_w, n_w, sp_w
            collinear_dropped = []
            kept_idx = np.asarray(keep)

    # 3. Fit (aligned with SINDY LA UQ: normalized Theta, then un-normalize coefs)
    # Paper-like: skip column normalization so threshold is in raw coefficient space.
    normalize_cols = getattr(config, "normalize_library_columns", False)
    if normalize_cols:
        Theta_n, col_norms = normalize_columns(Theta_clean)
    else:
        Theta_n = np.asarray(Theta_clean, float)
        col_norms = np.ones(Theta_clean.shape[1], dtype=float)
    _ppm = (getattr(config, "pareto_pick_mode", None) or "").lower()
    if _ppm == "last":
        pareto_pick = "last"
    elif _ppm in ("bic", "per_target_bic"):
        pareto_pick = "bic"
    else:
        pareto_pick = "knee"
    # When keeping collinear terms, use stronger ridge so regression stabilizes (Lorenz x,z,x*z etc.)
    # Paper-like with exact derivatives: use minimal ridge so fit matches LSTSQ — do NOT override.
    alpha_ridge = getattr(config, "alpha_ridge", 1e-6)
    in_paper_like_mode = (
        getattr(config, "single_threshold", None) is not None
        and getattr(config, "use_physical_library", False)
    )
    if not getattr(config, "prefer_parsimony", False) and not in_paper_like_mode:
        alpha_ridge = max(alpha_ridge, 1e-3)
    fit_cfg = SINDyFitConfig(
        mode=config.sindy_mode,
        ensemble_B=config.ensemble_B,
        ensemble_frac=config.ensemble_frac,
        map_lam=config.map_lam,
        map_sigma2=config.map_sigma2,
        sparsity_bias=config.sparsity_bias,
        alpha_ridge=alpha_ridge,
        pareto_pick=pareto_pick,
        pareto_dial=getattr(config, "pareto_dial", None),
        equal_weight_per_target=getattr(config, "equal_weight_per_target", True),
        bic_mse_variance_fraction=float(getattr(config, "bic_mse_variance_fraction", 1e-2)),
        bic_mse_floor_epsilon=float(getattr(config, "bic_mse_floor_epsilon", 1e-12)),
    )
    # Threshold scale: same max_coef as full Pareto sweep so single_* (sparsest/knee/densest) actually differ
    K = Theta_n.shape[1]
    if K > 0:
        I = np.eye(K, dtype=float)
        try:
            Xi_ls = np.linalg.solve(Theta_n.T @ Theta_n + alpha_ridge * I, Theta_n.T @ Y_phys)
        except np.linalg.LinAlgError:
            Xi_ls = np.linalg.lstsq(Theta_n.T @ Theta_n + alpha_ridge * I, Theta_n.T @ Y_phys, rcond=None)[0]
        max_coef = float(np.max(np.abs(Xi_ls))) if Xi_ls.size else 1.0
    else:
        max_coef = 1.0

    single_thr = getattr(config, "single_threshold", None)
    if single_thr is not None:
        # Scale single threshold by max_coef so 0.5/0.05/0.005 give sparsest/knee/densest vs coefficient magnitudes
        fit_cfg.thresholds = np.array([float(single_thr) * max_coef], dtype=float)
    elif fit_cfg.thresholds is None:
        fit_cfg.thresholds = np.logspace(-6, 2, config.n_thresholds) * max_coef
    fit_res = fit_sindy(Theta_n, Y_phys, cfg=fit_cfg, plot=config.pareto_plot)

    # coef_kept: coefficients in Theta_clean space (Y_phys = Theta_clean @ coef_kept)
    coef_kept = np.asarray(fit_res["coef"], float) / col_norms[:, None]

    # 4. Convert to physical units for equation display (Theta uses Z_scaled for linear terms)
    # So coef_kept is "per scaled unit"; for sympy display we need "per physical unit".
    # When use_physical_library=True, library was built from Z_phys so coef_kept is already physical.
    scale_arr = np.asarray(scaler.scale_, float).reshape(-1)
    offset_arr = np.asarray(getattr(scaler, "mean_", np.zeros_like(scale_arr)), float).reshape(-1)
    feature_scales = np.ones(len(kept_idx), dtype=float)
    if not use_physical_library:
        for k, lib_idx in enumerate(kept_idx):
            ci = lib.combo_index_for_library_column(int(lib_idx))
            if ci is None:
                continue  # constant "1" column (only when include_constant_term)
            combo = lib.valid_combos[ci]
            for ai in combo:
                atom = lib.active_atoms[ai]
                if atom.input_space == "scaled":
                    feature_scales[k] *= scale_arr[atom.idx]
    coef_phys = coef_kept / np.maximum(feature_scales[:, None], 1e-20)
    # For StandardScaler (nonzero mean): library column is product of (x_i - mean_i)/scale_i. Expanding
    # (x_i - mean_i)/scale_i = x_i/scale_i - mean_i/scale_i, so constant part is product(-mean_i/scale_i).
    # Add that to the constant coefficient so equations match (only if a literal "1" column exists).
    if np.any(offset_arr != 0):
        k_const_list = [k for k, nm in enumerate(kept_names) if str(nm).strip() == "1"]
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
                    coef_phys[k_const, :] += coef_kept[k, :] * factor

    # 5. Build Equations (physical coefficients × physical sympy terms)
    # Use evalf(5) so displayed equations are in decimal form, not fractions
    equations = {}
    for j, name in enumerate(target_names):
        expr = sp.Integer(0)
        for i, c in enumerate(coef_phys[:, j]):
            if abs(float(c)) > 1e-10:
                expr += float(c) * kept_sp[i]
        equations[name] = sp.nsimplify(expr, rational=False, tolerance=1e-3).evalf(5)

    # 6. No-truth metrics: AIC/BIC (lower = better parsimony vs fit; no ξ_true needed)
    n_obs = Theta_n.shape[0] * Y_phys.shape[1]
    k_params = int(np.count_nonzero(coef_kept))
    mse_fit = float(np.mean((Y_phys - Theta_n @ coef_kept) ** 2)) + 1e-30
    aic = n_obs * np.log(mse_fit) + 2.0 * k_params
    bic = n_obs * np.log(mse_fit) + np.log(max(n_obs, 1)) * k_params

    out = {
        "dt": dt,
        "scaler": scaler,
        "library": lib,
        "Theta_clean": Theta_clean,
        "kept_idx": kept_idx,
        "kept_names": kept_names,
        "kept_sp": kept_sp,
        "col_norms": col_norms,
        "coef_kept": coef_kept,
        "coef_phys": coef_phys,
        "equations": equations,
        "target_names": target_names,
        "Y_phys": Y_phys,
        "inclusion_probs": fit_res.get("inclusion_probs"),
        "collinear_dropped": collinear_dropped,
        "use_physical_library": use_physical_library,
        "aic": float(aic),
        "bic": float(bic),
        "n_obs": n_obs,
        "k_params": k_params,
    }
    # Pass through pareto/var_y etc. so diagnostic dashboard can plot Pareto frontier
    for key in (
        "pareto",
        "var_y",
        "best_mse",
        "best_complexity",
        "best_threshold",
        "bic_scores",
        "best_bic",
    ):
        if key in fit_res:
            out[key] = fit_res[key]
    return out
# ----------------------------
# Validation
# ----------------------------

def validate_sindy_general(fit: Dict, Z_phys: np.ndarray, target_indices: list[int]) -> Dict:
    lib = fit["library"]
    kept_idx = np.asarray(fit["kept_idx"], dtype=int)
    coef = np.asarray(fit["coef_kept"], dtype=float)

    # Use stored Y_phys from fit when available (same preprocessing as training)
    if "Y_phys" in fit and fit["Y_phys"] is not None:
        Y_true = np.asarray(fit["Y_phys"], dtype=float)
    else:
        prep = preprocess_timeseries(
            Z_phys, dt=fit["dt"],
            savgol_window=7, savgol_poly=3, deriv=1,
        )
        scale = np.asarray(fit["scaler"].scale_)
        Y_true = prep.X_dot_scaled[:, target_indices] * scale[target_indices][None, :]

    aff = affine_from_sklearn_scaler(fit["scaler"])
    Zs = aff.transform(Z_phys)
    if fit.get("use_physical_library"):
        Theta_full = lib.transform(Z_scaled=Z_phys, Z_phys=Z_phys)
    else:
        Theta_full = lib.transform(Z_scaled=Zs, Z_phys=Z_phys)
    Y_hat = Theta_full[:, kept_idx] @ coef

    return _debug_metrics_from_err(Y_true=Y_true, Y_hat=Y_hat)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r; returns nan if either series is (near) constant."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-15 or sb < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _apply_small_angle_preference(
    T_w: np.ndarray,
    n_w: List[str],
    sp_w: List[Any],
    keep: np.ndarray,
    config: SINDyRunConfig,
    measured_names: Sequence[str],
    Z_phys: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str], List[Any], np.ndarray, List[Tuple[str, str]]]:
    """
    Data-driven library pruning for small-angle regimes:
    - Prefer sin(theta) over linear theta when |corr(theta, sin(theta))| ~ 1.
    - Drop standalone cos(theta) when ~1 (constant / collinear with bias).
    - Prefer C*cos(theta) over standalone C when C ~ C*cos(theta) on data.

    Linear library atoms use **scaled** state; trig atoms use **physical** state (see
    ``SINDyLibrary``). Rule (1) must correlate **physical** theta with ``sin(theta)``
    (same space as the trig column); otherwise |corr| is artificially low when
    ``normalize_library_columns`` / maxabs scaling is on.

    Returns updated (T_w, n_w, sp_w, keep) and a list of (dropped_name, reason).
    """
    if not getattr(config, "small_angle_preference", False):
        return T_w, n_w, sp_w, keep, []

    dropped: List[Tuple[str, str]] = []
    drop_names: set = set()

    sin_thresh = float(getattr(config, "small_angle_sin_corr_thresh", 0.99))
    cos_corr = float(getattr(config, "small_angle_cos_const_corr_thresh", 0.995))
    cos_std = float(getattr(config, "small_angle_cos_min_std", 1e-4))
    int_thresh = float(getattr(config, "small_angle_interaction_corr_thresh", 0.99))

    name_to_i = {nm: i for i, nm in enumerate(n_w)}
    phys_col = {nm: j for j, nm in enumerate(measured_names)}

    angle_names = getattr(config, "small_angle_angle_names", None)
    if angle_names is None:
        # Only states that actually have both linear and sin(·) atoms in this library.
        angle_names = [
            nm
            for nm in measured_names
            if nm in name_to_i and f"sin({nm})" in name_to_i
        ]
    control_names = getattr(config, "small_angle_control_names", None) or []
    ones = np.ones(T_w.shape[0], dtype=float)

    # 1) Linear θ vs sin(θ): drop raw θ only (single-factor linear atom).
    if getattr(config, "small_angle_drop_linear_for_sin", True):
        for theta in angle_names:
            lin = str(theta)
            sfeat = f"sin({theta})"
            if lin not in name_to_i or sfeat not in name_to_i:
                continue
            i_lin = name_to_i[lin]
            i_sin = name_to_i[sfeat]
            # Trig column = sin(Z_phys); linear column = scaled θ — compare θ_phys to sin.
            if (
                Z_phys is not None
                and Z_phys.ndim == 2
                and lin in phys_col
                and Z_phys.shape[1] > phys_col[lin]
            ):
                lin_series = np.asarray(Z_phys[:, phys_col[lin]], dtype=float).ravel()
            else:
                lin_series = T_w[:, i_lin]
            r = _safe_pearson(lin_series, T_w[:, i_sin])
            if not np.isnan(r) and abs(r) >= sin_thresh:
                drop_names.add(lin)
                dropped.append(
                    (
                        lin,
                        f"small-angle|prefer sin over linear: |corr({lin}_phys,{sfeat})|={abs(r):.4f}≥{sin_thresh}",
                    )
                )

    # 2) Standalone cos(θ) ≈ 1: drop cos(θ) (bias term remains).
    if getattr(config, "small_angle_drop_cos_near_one", True):
        for theta in angle_names:
            cfeat = f"cos({theta})"
            if cfeat not in name_to_i:
                continue
            j = name_to_i[cfeat]
            colc = T_w[:, j]
            std_c = float(np.std(colc))
            r1 = _safe_pearson(colc, ones)
            if std_c < cos_std or (not np.isnan(r1) and abs(r1) >= cos_corr):
                drop_names.add(cfeat)
                r1_abs = abs(r1) if not np.isnan(r1) else float("nan")
                dropped.append(
                    (cfeat, f"small-angle|cos≈1: std={std_c:.2e}, |corr(·,1)|={r1_abs}"),
                )

    # 3) C vs C*cos(θ): drop standalone C when collinear with interacting cosine term.
    if getattr(config, "small_angle_drop_control_for_cos_product", True):
        for C in control_names:
            if C not in name_to_i:
                continue
            i_c = name_to_i[C]
            col_c = T_w[:, i_c]
            for theta in angle_names:
                cos_t = f"cos({theta})"
                prod_a = f"{C}*{cos_t}"
                prod_b = f"{cos_t}*{C}"
                prod_i = None
                pn = None
                if prod_a in name_to_i:
                    prod_i, pn = name_to_i[prod_a], prod_a
                elif prod_b in name_to_i:
                    prod_i, pn = name_to_i[prod_b], prod_b
                if prod_i is None:
                    continue
                r = _safe_pearson(col_c, T_w[:, prod_i])
                if not np.isnan(r) and abs(r) >= int_thresh:
                    if C not in drop_names:
                        drop_names.add(C)
                        dropped.append(
                            (
                                C,
                                f"small-angle|prefer {pn} over {C}: |corr|={abs(r):.4f}≥{int_thresh}",
                            )
                        )
                    break  # one reason is enough for C

    if not drop_names:
        return T_w, n_w, sp_w, keep, []

    keep_cols: List[int] = []
    new_n: List[str] = []
    new_sp: List[Any] = []
    new_keep: List[int] = []
    for col_i, (nm, sp_expr, kidx) in enumerate(zip(n_w, sp_w, keep)):
        if nm in drop_names:
            continue
        keep_cols.append(col_i)
        new_n.append(nm)
        new_sp.append(sp_expr)
        new_keep.append(int(kidx))

    if not keep_cols:
        return (
            np.ones((T_w.shape[0], 1), dtype=float),
            ["1"],
            [sp.Integer(1)],
            np.asarray([0], dtype=int),
            dropped,
        )

    T_out = T_w[:, keep_cols]
    return T_out, new_n, new_sp, np.asarray(new_keep, dtype=int), dropped


# ----------------------------
# Theta build (shared by runner and scout)
# ----------------------------
def _build_theta_and_prep(t, Z_phys, model, config, budget, Z_dot_phys=None):
    """
    Build preprocessed state, library matrix Theta (after constant/collinear pruning).
    Returns (Theta_clean, kept_names, kept_sp, kept_idx, prep, lib, collinear_dropped,
    constant_dropped_names, trig_dropped_names, small_angle_dropped).
    """
    from sindy.library import SINDyLibrary

    t = np.asarray(t).flatten()
    Z_phys = np.asarray(Z_phys)
    Z_dot_phys = np.asarray(Z_dot_phys) if Z_dot_phys is not None else None
    dt = estimate_dt(t)

    prep = preprocess_timeseries(
        Z_phys,
        dt=dt,
        scaler_kind=getattr(config, "scaler_kind", "maxabs"),
        savgol_window=config.savgol_window,
        savgol_poly=config.savgol_poly,
        deriv=1,
        X_dot_phys=Z_dot_phys,
    )
    Z_scaled = prep.X_scaled

    lib = SINDyLibrary(
        model.measured_names,
        max_degree=config.max_degree,
        max_interaction=config.max_interaction,
        custom_budget=budget,
        keep_feature=getattr(config, "library_keep_feature", None),
        include_constant_term=getattr(config, "library_include_constant", True),
    )
    # Sanity: with max_degree=1 the library must not contain product terms (e.g. "ail_act*elv_act")
    if config.max_degree == 1:
        combo_names = [f for f in lib.feature_names if f != "1"]
        product_terms = [f for f in combo_names if "*" in f]
        if product_terms:
            raise ValueError(
                f"Library has product terms with max_degree=1: {product_terms[:5]}. "
                "Ensure config has max_degree=1 and max_interaction=1, or pass force_linear_library=True."
            )
    if getattr(config, "use_physical_library", False):
        Theta_raw = lib.transform(Z_scaled=Z_phys, Z_phys=Z_phys)
    else:
        Theta_raw = lib.transform(Z_scaled=Z_scaled, Z_phys=Z_phys)

    T_w, keep, n_w, sp_w = drop_constant_like_columns(
        Theta_raw, lib.feature_names, lib.sp_features, cv_thresh=config.constant_cv_thresh
    )
    keep_after_const = {int(i) for i in np.asarray(keep).ravel().tolist()}
    constant_dropped_names = [
        str(nm) for i, nm in enumerate(lib.feature_names) if i not in keep_after_const
    ]

    T_w, n_w, sp_w, keep, small_angle_dropped = _apply_small_angle_preference(
        T_w, n_w, sp_w, keep, config, lib.measured_names, Z_phys=Z_phys
    )

    # Optional: remove features containing multiple trig factors.
    trig_dropped_names = []
    if getattr(config, "remove_double_trig_terms", False):
        trig_filtered_cols = []
        trig_filtered_names = []
        trig_filtered_sp = []
        trig_filtered_keep = []
        for col_i, (nm, sp_expr, kidx) in enumerate(zip(n_w, sp_w, keep)):
            n_trig = str(nm).count("sin(") + str(nm).count("cos(")
            if n_trig >= 2:
                trig_filtered_names.append(str(nm))
                continue
            trig_filtered_cols.append(col_i)
            trig_filtered_sp.append(sp_expr)
            trig_filtered_keep.append(kidx)
        if len(trig_filtered_cols) > 0:
            T_w = T_w[:, trig_filtered_cols]
            n_w = [n_w[i] for i in trig_filtered_cols]
            sp_w = trig_filtered_sp
            keep = np.asarray(trig_filtered_keep, dtype=int)
        elif getattr(config, "library_include_constant", True):
            # Keep at least constant if all got filtered unexpectedly.
            T_w = np.ones((T_w.shape[0], 1), dtype=float)
            n_w = ["1"]
            sp_w = [sp.Integer(1)]
            keep = np.asarray([0], dtype=int)
        else:
            raise ValueError(
                "remove_double_trig_terms removed every library column; with "
                "library_include_constant=False (e.g. Option 2 NF) a synthetic constant Θ column "
                "is not allowed. Relax the trig filter or the library."
            )
    else:
        trig_filtered_names = []
    trig_dropped_names = trig_filtered_names
    if getattr(config, "prefer_parsimony", False):
        Theta_clean, rel, kept_names, kept_sp, collinear_dropped = prefer_parsimony(
            T_w, n_w, sp_w, threshold=config.collinear_threshold
        )
        kept_idx = np.asarray(keep)[rel]
    else:
        Theta_clean, kept_names, kept_sp = T_w, n_w, sp_w
        collinear_dropped = []
        kept_idx = np.asarray(keep)

    return (
        Theta_clean,
        kept_names,
        kept_sp,
        kept_idx,
        prep,
        lib,
        collinear_dropped,
        constant_dropped_names,
        trig_dropped_names,
        small_angle_dropped,
    )


def run_scout(t, Z_phys, model, config, budget, Z_dot_phys=None) -> Dict:
    """
    Lightweight scout: build Theta with the given config and run excitation only (no fit).
    Returns a flat scout dict for recommend_config().
    """
    Theta_clean, kept_names, kept_sp, kept_idx, prep, lib, _, _, _, _ = _build_theta_and_prep(
        t, Z_phys, model, config, budget, Z_dot_phys
    )
    target_indices = model.target_indices
    scale = np.asarray(prep.scaler.scale_).reshape(-1)
    Y_phys = prep.X_dot_scaled[:, target_indices] * scale[target_indices][None, :]

    ex = excitation_report(
        t=t,
        Z=Z_phys,
        names=model.measured_names,
        Theta=Theta_clean,
        feature_names=kept_names,
        Y=Y_phys,
    )
    return flatten_scout_from_excitation(ex)


# ----------------------------
# Runner
# ----------------------------
def run_sindy_pipeline_general(
    *, t, Z_phys, model, config=None, budget=None,
    run_diagnostics=True, output_dir="outputs/",
    Z_dot_phys=None,
    force_linear_library: bool = False,
) -> Dict:
    """
    Run the SINDy pipeline. Optional Z_dot_phys: (n_samples, n_states) exact derivatives
    (e.g. from known ODE). When provided, used instead of Savitzky–Golay; replicates
    Brunton paper setup for validation (e.g. Lorenz).

    Set config="auto" to run a scout and apply recommended overrides before fitting.

    If force_linear_library=True, overrides config.max_degree=1 and config.max_interaction=1
    so the library has only single-variable terms (no cross-terms like P*elv_act).
    Use this for aerodynamic coefficient fits where the true model is linear in state/controls.
    """
    if config == "auto":
        return run_sindy_pipeline_auto(
            t=t, Z_phys=Z_phys, model=model, budget=budget,
            Z_dot_phys=Z_dot_phys, run_diagnostics=run_diagnostics, output_dir=output_dir,
        )
    if config is None:
        config = SINDyRunConfig()
    if force_linear_library:
        config = replace(config, max_degree=1, max_interaction=1)
    t = np.asarray(t).flatten()
    Z_phys = np.asarray(Z_phys)
    Z_dot_phys = np.asarray(Z_dot_phys) if Z_dot_phys is not None else None
    dt = estimate_dt(t)

    (
        Theta_clean,
        kept_names,
        kept_sp,
        kept_idx,
        prep,
        lib,
        collinear_dropped,
        constant_dropped_names,
        trig_dropped_names,
        small_angle_dropped,
    ) = _build_theta_and_prep(t, Z_phys, model, config, budget, Z_dot_phys)
    Z_scaled = prep.X_scaled
    dt = prep.dt

    # ALIGNED FIT: Hand off pre-aligned data (and optional exact derivatives)
    fit = fit_sindy_main(
        Z_phys=Z_phys, measured_names=model.measured_names,
        target_names=model.target_names, target_indices=model.target_indices,
        dt=dt, config=config, budget=budget,
        pre_theta=Theta_clean, pre_names=kept_names, pre_sp=kept_sp, pre_idx=kept_idx,
        Z_dot_phys=Z_dot_phys,
        use_physical_library=getattr(config, "use_physical_library", False),
    )
    fit["collinear_dropped"] = collinear_dropped  # (dropped_name, kept_name, r, reason) from prefer_parsimony
    fit["constant_dropped_names"] = constant_dropped_names
    fit["double_trig_dropped_names"] = trig_dropped_names
    fit["small_angle_dropped"] = small_angle_dropped
    if constant_dropped_names:
        print(
            f"Warning: dropped {len(constant_dropped_names)} near-constant feature(s) "
            f"(CV < {getattr(config, 'constant_cv_thresh', 1e-3)}):"
        )
        for nm in constant_dropped_names:
            print(f"  dropped {nm!r} (reason: near-constant feature)")
    if small_angle_dropped:
        print(
            f"Small-angle preference: dropped {len(small_angle_dropped)} feature(s) "
            "(prefer trig over local linear / cos≈1 / proxy controls on training data):"
        )
        for nm, reason in small_angle_dropped:
            print(f"  dropped {nm!r} (reason: {reason})")
    if collinear_dropped:
        print(
            f"Correlated library columns resolved ({len(collinear_dropped)} event(s)), "
            f"threshold={getattr(config, 'collinear_threshold', 0.995)}, prefer_parsimony:"
        )
        for row in collinear_dropped:
            dropped_name, kept_name, r = row[0], row[1], row[2]
            tag = row[3] if len(row) > 3 else "legacy"
            print(f"  dropped {dropped_name!r} → kept {kept_name!r}  |r|={r:.3f}  ({tag})")
    if trig_dropped_names:
        print(f"Warning: dropped {len(trig_dropped_names)} double-trig feature(s) (remove_double_trig_terms=True):")
        for nm in trig_dropped_names:
            print(f"  dropped {nm!r} (reason: double trig feature)")

    val = validate_sindy_general(fit, Z_phys, model.target_indices)
    res = {"t": t, "Z": Z_phys, "config": config, "fit": fit, "validation": val, "model": model}

    if run_diagnostics:
        ex = excitation_report(
            t=t,
            Z=Z_phys,
            names=model.measured_names,
            Theta=Theta_clean,
            feature_names=kept_names,
            Y=fit.get("Y_phys"),
        )
        print(format_excitation_report(ex))

    if run_diagnostics:
        from sindy.diagnostics_plots import run_diagnostic_dashboard
        run_diagnostic_dashboard(res, output_dir=output_dir)
        # Unified numeric report (truth, excitation, scale, collinearity) for sensitivity analysis
        try:
            from idtools.diagnostic_suite import run_diagnostic_suite, DiagnosticSuiteConfig
            res["diagnostic_suite"] = run_diagnostic_suite(
                res,
                config=DiagnosticSuiteConfig(run_plots=False),  # dashboard already run above
                verbose=False,  # excitation already printed above
            )
            # Idiot-proof: attach success assessment when diagnostics are available
            try:
                from idtools.sindy_success import assess_sindy_success
                sa = assess_sindy_success(res, diagnostic_report=res.get("diagnostic_suite"))
                res["success_assessment"] = {
                    "success_level": sa.success_level,
                    "confidence_score": sa.confidence_score,
                    "message": sa.message,
                    "has_truth": sa.has_truth,
                    "metrics": sa.metrics,
                }
            except Exception:
                pass
        except Exception as e:
            res["diagnostic_suite"] = {"error": str(e)}

    return res


def run_sindy_pipeline_auto(
    *,
    t,
    Z_phys,
    model,
    budget=None,
    base_config=None,
    Z_dot_phys=None,
    run_diagnostics=True,
    output_dir="outputs/",
    verbose_auto=True,
    problem_class=None,
) -> Dict:
    """
    Run the pipeline with automated config: run a lightweight scout (build Theta + excitation),
    recommend overrides from diagnostics, then run once with the recommended config.

    problem_class : "toy" | "real" | None
        If "toy", force paper-like settings when exact derivatives + small system + truth.
        If "real", use robust rules only (no paper-like). If None, infer from data via infer_problem_class.

    Returns the same structure as run_sindy_pipeline_general, plus:
      res["auto_config"] = {"recommended_overrides": dict, "reasons": list of str, "scout": dict}
    """
    config = base_config or SINDyRunConfig()
    scout = run_scout(t, Z_phys, model, config, budget, Z_dot_phys)
    derivative_agreement = None
    if Z_dot_phys is not None:
        try:
            derivative_agreement = compute_derivative_agreement(
                t, Z_phys, np.asarray(Z_dot_phys),
                scaler_kind=getattr(config, "scaler_kind", "maxabs"),
                savgol_window=getattr(config, "savgol_window", 7),
                savgol_poly=getattr(config, "savgol_poly", 3),
            )
        except Exception:
            pass
    overrides, reasons = recommend_config(
        scout,
        Z_dot_phys=Z_dot_phys,
        model=model,
        problem_class_override=problem_class,
        derivative_agreement=derivative_agreement,
    )

    if overrides:
        config = replace(config, **overrides)
        if verbose_auto:
            print("[Auto config] Applying recommended overrides:")
            for r in reasons:
                print(f"  - {r}")
            print(f"  Overrides: {overrides}")
    elif verbose_auto:
        print("[Auto config] No overrides recommended; using base config.")

    res = run_sindy_pipeline_general(
        t=t,
        Z_phys=Z_phys,
        model=model,
        config=config,
        budget=budget,
        Z_dot_phys=Z_dot_phys,
        run_diagnostics=run_diagnostics,
        output_dir=output_dir,
    )
    inferred_class, _ = infer_problem_class(
        scout, Z_dot_phys=Z_dot_phys, model=model, derivative_agreement=derivative_agreement
    )
    effective_class = (
        "toy_like" if problem_class == "toy" else
        "real_world" if problem_class == "real" else
        inferred_class
    )
    res["auto_config"] = {
        "recommended_overrides": overrides,
        "reasons": reasons,
        "scout": scout,
        "problem_class": effective_class,
        "derivative_agreement": derivative_agreement,
    }
    return res


def run_consensus_suite(
    t: np.ndarray,
    Z_phys: np.ndarray,
    model: Any,
    budget: Dict,
    output_dir: str = "outputs/consensus/",
) -> Dict:
    """
    Orchestrates a multi-method discovery suite.
    Generates 1 Diagnostic Report and 1 Consensus Summary.
    """
    from sindy.fit import summarize_consensus
    
    # Use pareto_pick_mode="last" (densest model) to match SINDY LA UQ and reduce ambiguous terms in consensus
    methodologies = {
        "pareto":     SINDyRunConfig(sindy_mode="pareto", n_thresholds=100, pareto_pick_mode="last"),
        "ensemble":   SINDyRunConfig(sindy_mode="pareto", ensemble_B=50, pareto_pick_mode="last"),
        "bayes_map":  SINDyRunConfig(sindy_mode="bayes_map", map_lam=1e-3),
        "bayes_ens":  SINDyRunConfig(sindy_mode="bayes_map", ensemble_B=50)
    }

    results_by_mode = {}
    
    # --- Step 1: The Anchor Run (Diagnostics ON) ---
    print(f"\n[1/4] Running PARETO (Anchor + Diagnostics)...")
    anchor = run_sindy_pipeline_general(
        t=t, Z_phys=Z_phys, model=model, budget=budget,
        config=methodologies["pareto"],
        output_dir=output_dir,
        run_diagnostics=True 
    )
    results_by_mode["pareto"] = anchor["fit"]

    # --- Step 2: The Silent Runs (Diagnostics OFF) ---
    for name in ["ensemble", "bayes_map", "bayes_ens"]:
        print(f"[{len(results_by_mode)+1}/4] Running {name.upper()}...")
        # FIX: Ensure Z_phys matches the function argument exactly
        res = run_sindy_pipeline_general(
            t=t, Z_phys=Z_phys, model=model, budget=budget,
            config=methodologies[name],
            run_diagnostics=False 
        )
        results_by_mode[name] = res["fit"]

    # --- Step 3: Automated Consensus (include true coefficients when available) ---
    Xi_true = None
    try:
        from idtools.xi_true_from_sim_equations import xi_true_from_sim_equations
        xi_true, _ = xi_true_from_sim_equations(
            model=anchor["model"],
            lib=anchor["fit"]["library"],
            scaler=anchor["fit"]["scaler"],
            Z_phys=anchor["Z"],
            kept_idx=anchor["fit"]["kept_idx"],
        )
        Xi_true = np.asarray(xi_true, float)
    except Exception:
        pass
    report = summarize_consensus(
        results_by_mode=results_by_mode,
        Xi_true=Xi_true,
        target_names=anchor["fit"].get("target_names"),
        true_decimal_fmt=".5f",
    )

    # --- Step 4: Comparison plot (Truth vs SINDy for all 4 methods) ---
    suite = {"modes": results_by_mode, "report": report, "anchor_meta": anchor}
    try:
        from sindy.diagnostics_plots import plot_consensus_comparison
        plot_consensus_comparison(suite, output_dir=output_dir)
    except Exception as e:
        print(f"Warning: Could not generate consensus comparison plot: {e}")

    return suite


def _data_processing_reasons_from_scout(scout: Dict) -> List[tuple]:
    """Build (option, reason) list for each data-processing setting from scout diagnostics."""
    col_ratio = scout.get("col_norm_ratio", 1.0)
    suggest_norm = scout.get("suggest_normalize_library", False)
    state_ratio = scout.get("state_std_ratio", 1.0)
    suggest_std = scout.get("suggest_standard_scaling", False)
    reasons = [
        ("baseline", "Brunton/paper-like: no lib column norm, maxabs scaling (default)."),
        (
            "normalize_library_columns",
            f"True recommended (col_norm_ratio={col_ratio:.1f} > 1000)" if suggest_norm
            else f"False — paper-like default (col_norm_ratio={col_ratio:.1f}).",
        ),
        (
            "scaler_kind",
            f"standard recommended (state_std_ratio={state_ratio:.1f} > 100)" if suggest_std
            else f"maxabs — paper-like default (state_std_ratio={state_ratio:.1f}).",
        ),
        ("collinear_keep", "prefer_parsimony=False + normalize_library_columns=True; use when terms dropped by correlation."),
        ("lib_raw", "Same as baseline: no column normalization (explicit raw library)."),
        ("lib_physical", "Optional (use_physical_library=True); use for raw-state library (e.g. paper match)."),
    ]
    return reasons


def run_extended_consensus_suite(
    t: np.ndarray,
    Z_phys: np.ndarray,
    model: Any,
    budget: Dict,
    output_dir: str = "outputs/consensus/",
    include_data_processing: bool = True,
    run_all_combos: bool = False,
    Z_dot_phys: Optional[np.ndarray] = None,
    *,
    custom_to_run: Optional[List[str]] = None,
    custom_methodologies: Optional[Dict[str, SINDyRunConfig]] = None,
    custom_data_opts: Optional[List[str]] = None,
    run_surrogate_battery: bool = False,
) -> Dict:
    """
    Extended consensus with non-overlapping dimensions: Solver × Sparsity × Data.

    Dimensions:
      - Solver: Pareto, Ensemble, BayesMAP, BayesEnsemble (4).
      - Sparsity: target_sparsest, target_knee, target_densest, single_sparsest, single_knee, single_densest (6; only for Pareto/Ensemble).
      - Data: baseline, collinear_keep, scale_zscore, lib_raw, lib_physical (5).

    By default (include_data_processing=True): dimension sweep (13 runs). Set run_all_combos=True for full grid (70 runs).

    If custom_to_run and custom_methodologies are provided, only those configs are run (targeted consensus);
    custom_data_opts can be set for report grouping (default baseline + scale_zscore when custom).
    """
    from sindy.fit import summarize_consensus

    data_opts = list(DATA_OPTS) if include_data_processing else ["baseline"]
    baseline_config = _config_from_dims("Pareto", "target_densest", "baseline")

    # Scout once for data-processing reasons (why/why not each option)
    data_processing_reasons: List[tuple] = []
    try:
        scout = run_scout(t, Z_phys, model, baseline_config, budget, Z_dot_phys=Z_dot_phys)
        data_processing_reasons = _data_processing_reasons_from_scout(scout)
    except Exception as e:
        data_processing_reasons = [("(scout failed)", str(e))]
    methodologies: Dict[str, SINDyRunConfig] = {}
    to_run: List[str] = []

    if custom_to_run is not None and custom_methodologies is not None:
        to_run = list(custom_to_run)
        methodologies = dict(custom_methodologies)
        if custom_data_opts is not None:
            data_opts = list(custom_data_opts)
    elif run_all_combos:
        # Full grid: Pareto × 6 × 5, Ensemble × 6 × 5, BayesMAP × 5, BayesEnsemble × 5
        for solver in ("Pareto", "Ensemble"):
            for sp in SPARSITY_OPTS:
                for data in data_opts:
                    name = _method_name(solver, sp, data)
                    methodologies[name] = _config_from_dims(solver, sp, data)
                    to_run.append(name)
        for solver in ("BayesMAP", "BayesEnsemble"):
            for data in data_opts:
                name = _method_name(solver, None, data)
                methodologies[name] = _config_from_dims(solver, None, data)
                to_run.append(name)
    else:
        # Dimension sweep: unique set of runs for Solver | Sparsity | Data
        seen: set = set()
        def add(name: str, s: str, sp: Optional[str], d: str):
            if name not in seen:
                seen.add(name)
                methodologies[name] = _config_from_dims(s, sp, d)
                to_run.append(name)
        # Solver (4): densest + baseline
        for solver in SOLVERS:
            sp = None if solver in ("BayesMAP", "BayesEnsemble") else "target_densest"
            add(_method_name(solver, sp, "baseline"), solver, sp, "baseline")
        # Sparsity (6): Pareto + baseline (one duplicate already added)
        for sp in SPARSITY_OPTS:
            add(_method_name("Pareto", sp, "baseline"), "Pareto", sp, "baseline")
        # Data (5): Pareto target_densest (one duplicate already added)
        for data in data_opts:
            add(_method_name("Pareto", "target_densest", data), "Pareto", "target_densest", data)

    results_by_mode: Dict[str, Dict] = {}
    full_results: Dict[str, Dict] = {}  # full res per method for validation

    for i, name in enumerate(to_run):
        print(f"\n[{i + 1}/{len(to_run)}] Running {name}...")
        # No per-run diagnostics or plots; consensus dir gets plots only after ALL runs finish.
        res = run_sindy_pipeline_general(
            t=t, Z_phys=Z_phys, model=model, budget=budget,
            config=methodologies[name],
            run_diagnostics=False,
            output_dir=output_dir,
            Z_dot_phys=Z_dot_phys,
        )
        results_by_mode[name] = res["fit"]
        full_results[name] = res

    anchor_name = _method_name("Pareto", "target_densest", "baseline")
    anchor = full_results.get(anchor_name) or full_results[to_run[0]]

    # Xi_true for consensus (when model has truth)
    Xi_true = None
    try:
        from idtools.xi_true_from_sim_equations import xi_true_from_sim_equations
        xi_true, _ = xi_true_from_sim_equations(
            model=anchor["model"],
            lib=anchor["fit"]["library"],
            scaler=anchor["fit"]["scaler"],
            Z_phys=anchor["Z"],
            kept_idx=anchor["fit"]["kept_idx"],
            use_physical_library=anchor["fit"].get("use_physical_library", False),
        )
        Xi_true = np.asarray(xi_true, float)
    except Exception:
        pass

    # --- Consensus reports by group (names from dimension grid) ---
    solver_methods = [_method_name(s, None if s in ("BayesMAP", "BayesEnsemble") else "target_densest", "baseline") for s in SOLVERS]
    solver_methods = [m for m in solver_methods if m in results_by_mode]
    sparsity_methods = [_method_name("Pareto", sp, "baseline") for sp in SPARSITY_OPTS]
    sparsity_methods = [m for m in sparsity_methods if m in results_by_mode]
    dp_methods = [_method_name("Pareto", "target_densest", d) for d in data_opts]
    dp_methods = [m for m in dp_methods if m in results_by_mode]

    def _safe_summarize(methods: list) -> Dict:
        if len(methods) < 2:
            return {}
        subset = {k: results_by_mode[k] for k in methods}
        n_features = [len(f.get("kept_names", [])) for f in subset.values()]
        if len(set(n_features)) > 1:
            return {"notes": ["Library sizes differ across methods; consensus by coefficient not computed."]}
        try:
            return summarize_consensus(
                results_by_mode=subset,
                Xi_true=Xi_true,
                target_names=anchor["fit"].get("target_names"),
                true_decimal_fmt=".5f",
            )
        except Exception as e:
            return {"notes": [f"Consensus summary failed: {e}"]}

    report_solver = _safe_summarize(solver_methods) if solver_methods else {}
    report_sparsity = _safe_summarize(sparsity_methods) if len(sparsity_methods) >= 2 else {}
    report_dp = _safe_summarize(dp_methods) if len(dp_methods) >= 2 else {}

    # --- Validation matrix and best method per state ---
    target_indices = model.target_indices
    target_names = list(anchor["fit"].get("target_names", []))
    n_states = len(target_names)
    method_names = list(results_by_mode.keys())
    r2_matrix = np.full((len(method_names), n_states), np.nan)
    for row, name in enumerate(method_names):
        val = full_results[name].get("validation", {})
        r2_by = val.get("r2_by_state", [])
        for j in range(min(len(r2_by), n_states)):
            r2_matrix[row, j] = float(r2_by[j])
    best_per_state = []
    for j in range(n_states):
        col = r2_matrix[:, j]
        valid = np.isfinite(col)
        if np.any(valid):
            best_per_state.append(method_names[int(np.nanargmax(col))])
        else:
            best_per_state.append(method_names[0])
    best_overall = method_names[int(np.nanargmax(np.nanmean(r2_matrix, axis=1)))] if np.any(np.isfinite(r2_matrix)) else method_names[0]

    # --- No-truth metrics: AIC/BIC per method (lower = better parsimony vs fit) ---
    aic_by_method = []
    bic_by_method = []
    n_terms_by_method: List[int] = []
    for name in method_names:
        fit = results_by_mode.get(name, {})
        aic_by_method.append(fit.get("aic", np.nan))
        bic_by_method.append(fit.get("bic", np.nan))
        coef = fit.get("coef_kept", fit.get("coef"))
        if coef is not None:
            n_terms_by_method.append(int(np.sum(np.abs(np.asarray(coef, dtype=float)) > 1e-12)))
        else:
            n_terms_by_method.append(0)
    best_by_aic = method_names[int(np.nanargmin(aic_by_method))] if np.any(np.isfinite(aic_by_method)) else None
    best_by_bic = method_names[int(np.nanargmin(bic_by_method))] if np.any(np.isfinite(bic_by_method)) else None

    # --- Per-method: stability (term agreement across configs), blind metrics (n_harmful, blind_score) ---
    stability_by_method: List[float] = []
    n_harmful_by_method: List[int] = []
    blind_score_by_method: List[float] = []
    mean_term_importance_by_method: List[float] = []
    adjusted_r2_by_method: List[float] = []
    try:
        from sindy.diagnostics_plots import _build_term_stability_matrix
        from idtools.config_sweep import _compute_blind_metrics
        _term_labels, _mnames, term_matrix, n_per_term = _build_term_stability_matrix(results_by_mode)
        M = len(method_names)
        for j, name in enumerate(method_names):
            if j < term_matrix.shape[1] and term_matrix.size > 0:
                # Stability: mean (n_methods_with_term / M) over this method's non-zero terms
                col = term_matrix[:, j]
                nonzero = np.abs(col) > 1e-10
                if np.any(nonzero):
                    fracs = np.asarray(n_per_term, dtype=float)[nonzero] / max(M, 1)
                    stability_by_method.append(float(np.mean(fracs)))
                else:
                    stability_by_method.append(0.0)
            else:
                stability_by_method.append(np.nan)
            # Blind metrics (harmful terms, blind_score, term importance)
            res = full_results.get(name, {})
            fit = res.get("fit", {})
            r2_mean = float(res.get("validation", {}).get("r2_mean", np.nan))
            n_terms = n_terms_by_method[j] if j < len(n_terms_by_method) else 1
            blind = _compute_blind_metrics(fit, r2_mean, max(n_terms, 1))
            n_harmful_by_method.append(blind.get("n_harmful_terms", 0))
            blind_score_by_method.append(blind.get("blind_score", np.nan))
            mean_term_importance_by_method.append(blind.get("mean_term_importance", np.nan))
            adjusted_r2_by_method.append(blind.get("adjusted_r2", r2_mean))
    except Exception:
        stability_by_method = [np.nan] * len(method_names)
        n_harmful_by_method = [0] * len(method_names)
        blind_score_by_method = [np.nan] * len(method_names)
        mean_term_importance_by_method = [np.nan] * len(method_names)
        adjusted_r2_by_method = [np.nan] * len(method_names)

    # --- ξ vs ξ_true: compare in a common basis (union of method features) ---
    # Methods can have different kept_idx subsets. Build one reference basis as the UNION of kept
    # library indices across all methods, project truth into that union basis, and expand each
    # method's physical coefficients into the same union order (0 where a method did not select a term).
    # Then ‖expanded − Xi_true_phys‖_F / ‖Xi_true_phys‖_F is comparable across methods.
    xi_agreement_matrix = None
    best_per_state_xi = None
    if anchor.get("model") is not None and Xi_true is not None:
        try:
            from idtools.preprocess import affine_from_sklearn_scaler
            Z_anchor = np.asarray(anchor["Z"])
            model = anchor["model"]
            fit_anchor = anchor["fit"]
            lib_anchor = fit_anchor["library"]
            n_t = Xi_true.shape[1]
            aff_anchor = affine_from_sklearn_scaler(fit_anchor["scaler"])
            Z_anchor_scaled = aff_anchor.transform(Z_anchor)
            Theta_full_anchor = lib_anchor.transform(Z_scaled=Z_anchor_scaled, Z_phys=Z_anchor)
            n_full_cols = int(Theta_full_anchor.shape[1])

            # True coefficients on full library basis (physical units).
            from idtools.xi_true_from_sim_equations import xi_true_from_sim_equations
            Xi_true_full = np.asarray(
                xi_true_from_sim_equations(
                    model=model,
                    lib=lib_anchor,
                    scaler=fit_anchor["scaler"],
                    Z_phys=Z_anchor,
                    kept_idx=None,
                    use_physical_library=fit_anchor.get("use_physical_library", False),
                )[0],
                dtype=float,
            )
            if Xi_true_full.ndim != 2:
                raise ValueError("Xi_true_full must be 2D.")
            n_ref = int(Xi_true_full.shape[0])
            if n_ref != n_full_cols:
                raise ValueError(f"Xi_true rows ({n_ref}) != library terms ({n_full_cols}).")

            # Expand every discovered method to full basis, then build union support = true ∪ discovered.
            expanded_by_method = []
            for name in method_names:
                fit_m = full_results[name]["fit"]
                coef_phys = fit_m.get("coef_phys")
                if coef_phys is None:
                    expanded_by_method.append(np.full((n_ref, n_t), np.nan))
                    continue
                coef_phys = np.asarray(coef_phys, float)
                if coef_phys.ndim == 1:
                    coef_phys = coef_phys.reshape(-1, 1)
                if coef_phys.shape[1] != n_t:
                    expanded_by_method.append(np.full((n_ref, n_t), np.nan))
                    continue
                kidx_m = np.asarray(fit_m.get("kept_idx", []), dtype=int).ravel()
                expanded_full = np.zeros((n_ref, n_t), dtype=float)
                for k, lib_idx in enumerate(kidx_m):
                    if 0 <= int(lib_idx) < n_ref and k < coef_phys.shape[0]:
                        expanded_full[int(lib_idx), :] = coef_phys[k, :]
                expanded_by_method.append(expanded_full)

            tiny = 1e-8
            discovered_stack = np.array(
                [
                    np.nan_to_num(np.max(np.abs(E), axis=1), nan=0.0)
                    for E in expanded_by_method
                ],
                dtype=float,
            ) if expanded_by_method else np.zeros((0, n_ref), dtype=float)
            discovered_row_mag = np.max(discovered_stack, axis=0) if discovered_stack.size else np.zeros(n_ref, dtype=float)
            true_row_mag = np.max(np.abs(Xi_true_full), axis=1)
            union_mask = np.maximum(true_row_mag, discovered_row_mag) > tiny
            if not np.any(union_mask):
                union_mask[:] = True

            Xi_true_phys = Xi_true_full[union_mask, :]
            n_ref_u = int(Xi_true_phys.shape[0])
            frob_true = float(np.sqrt(np.sum(Xi_true_phys ** 2)))
            xi_agreement_list = []
            xi_rel_err_list = []  # (method_name, rel_err) for "how far is ξ from ξ_true"
            xi_frob_err_list = []  # absolute Frobenius norm ||Xi - Xi_true||_F
            xi_frob_true_list = []  # denominator ||Xi_true||_F on union basis
            for name, expanded_full in zip(method_names, expanded_by_method):
                try:
                    if expanded_full.ndim != 2 or expanded_full.shape[1] != n_t:
                        xi_rel_err_list.append((name, np.nan))
                        xi_frob_err_list.append((name, np.nan))
                        xi_frob_true_list.append((name, frob_true))
                        xi_agreement_list.append(np.full(n_states, np.nan))
                        continue
                    expanded = np.asarray(expanded_full, dtype=float)[union_mask, :]
                    # Primary metric: relative Frobenius error (one number per method)
                    # ||ξ_proposed - ξ_true||_F / ||ξ_true||_F — low = recovered equations, high = curve-fitting
                    diff = expanded - Xi_true_phys
                    frob_err = float(np.sqrt(np.sum(diff ** 2)))
                    xi_rel_err = frob_err / (frob_true + 1e-30)
                    xi_rel_err_list.append((name, xi_rel_err))
                    xi_frob_err_list.append((name, frob_err))
                    xi_frob_true_list.append((name, frob_true))
                    # Per-state agreement for heatmap (optional detail)
                    frob_sq = float(np.sum(Xi_true_phys ** 2)) + 1e-30
                    denom_floor = 1e-4 * frob_sq
                    agree = np.zeros(n_t)
                    for j in range(n_t):
                        xt = Xi_true_phys[:, j]
                        xd = expanded[:, j]
                        denom = max(float(np.sum(xt ** 2)) + 1e-20, denom_floor)
                        ss_err = float(np.sum((xd - xt) ** 2))
                        agree[j] = denom / (ss_err + denom)
                    xi_agreement_list.append(agree)
                except Exception:
                    xi_rel_err_list.append((name, np.nan))
                    xi_frob_err_list.append((name, np.nan))
                    xi_frob_true_list.append((name, frob_true))
                    xi_agreement_list.append(np.full(n_states, np.nan))
            if xi_agreement_list:
                xi_agreement_matrix = np.array(xi_agreement_list)
                best_per_state_xi = []
                for j in range(n_states):
                    col = xi_agreement_matrix[:, j]
                    valid = np.isfinite(col)
                    if np.any(valid):
                        best_per_state_xi.append(method_names[int(np.nanargmax(col))])
                    else:
                        best_per_state_xi.append(method_names[0])
                # One number per method: ||ξ - ξ_true||_F / ||ξ_true||_F (lower = less curve-fitting)
                xi_rel_err_by_method = [err for _, err in xi_rel_err_list]
                xi_frob_err_by_method = [err for _, err in xi_frob_err_list]
                xi_frob_true_by_method = [den for _, den in xi_frob_true_list]
                best_by_xi_err = method_names[int(np.nanargmin(xi_rel_err_by_method))] if np.any(np.isfinite(xi_rel_err_by_method)) else None
            else:
                xi_rel_err_by_method = []
                xi_frob_err_by_method = []
                xi_frob_true_by_method = []
                best_by_xi_err = None
        except Exception:
            xi_rel_err_by_method = []
            xi_frob_err_by_method = []
            xi_frob_true_by_method = []
            best_by_xi_err = None
    else:
        xi_rel_err_by_method = []
        xi_frob_err_by_method = []
        xi_frob_true_by_method = []
        best_by_xi_err = None

    validation_dict = {
        "r2_matrix": r2_matrix,
        "method_names": method_names,
        "target_names": target_names,
        "best_per_state": best_per_state,
        "best_overall": best_overall,
        "aic_by_method": aic_by_method,
        "bic_by_method": bic_by_method,
        "best_by_aic": best_by_aic,
        "best_by_bic": best_by_bic,
        "n_terms_by_method": n_terms_by_method,
        "stability_by_method": stability_by_method,
        "n_harmful_by_method": n_harmful_by_method,
        "blind_score_by_method": blind_score_by_method,
        "mean_term_importance_by_method": mean_term_importance_by_method,
        "adjusted_r2_by_method": adjusted_r2_by_method,
    }
    if xi_agreement_matrix is not None and best_per_state_xi is not None:
        validation_dict["xi_agreement_matrix"] = xi_agreement_matrix
        validation_dict["best_per_state_xi"] = best_per_state_xi
    if xi_rel_err_by_method:
        validation_dict["xi_rel_err_by_method"] = xi_rel_err_by_method
        validation_dict["xi_frob_err_by_method"] = xi_frob_err_by_method
        validation_dict["xi_frob_true_by_method"] = xi_frob_true_by_method
        validation_dict["best_by_xi_err"] = best_by_xi_err
    # Best for discovery: never use truth (ξ). Use AIC/BIC then R². Truth is for post-hoc only.
    best_for_discovery = best_by_aic or best_by_bic or best_overall
    validation_dict["best_for_discovery"] = best_for_discovery
    best_for_display = best_for_discovery
    validation_dict["best_for_display"] = best_for_display
    if best_by_aic and best_for_display == best_by_aic:
        validation_dict["best_for_display_criterion"] = "AIC (discovery; truth not used)"
    elif best_by_bic and best_for_display == best_by_bic:
        validation_dict["best_for_display_criterion"] = "BIC (discovery; truth not used)"
    else:
        validation_dict["best_for_display_criterion"] = "R² (fallback)"
    # Overfitting warning: highest R² often has worse equation accuracy
    if best_overall and best_by_aic and best_overall != best_by_aic:
        validation_dict["overfit_warning"] = (
            f"Highest R² config ({best_overall}) may be overfitted; "
            f"AIC recommends {best_by_aic} for dynamics discovery."
        )

    suite = {
        "modes": results_by_mode,
        "full_results": full_results,
        "report": report_solver,
        "anchor_meta": anchor,
        "consensus": {
            "solver": {"report": report_solver, "methods": solver_methods},
            "sparsity": {"report": report_sparsity, "methods": sparsity_methods},
            "data_processing": {"report": report_dp, "methods": dp_methods},
        },
        "validation": validation_dict,
        "include_data_processing": include_data_processing,
        "data_processing_reasons": data_processing_reasons,
    }

    # Plots only after ALL methods have been run and suite is complete (physical units throughout).
    all_done = (
        len(results_by_mode) == len(to_run)
        and all(name in results_by_mode for name in to_run)
    )
    if not all_done:
        import warnings
        warnings.warn(
            f"Consensus suite incomplete: {len(results_by_mode)} results vs {len(to_run)} requested; skipping heat map and time series plots.",
            UserWarning,
        )
    else:
        # Optional: four-test surrogate for Xi-error (E_short, E_long, C=AIC, S=stability → J(m))
        if run_surrogate_battery:
            try:
                from idtools.surrogate_xi import four_test_battery_suite, surrogate_best_method
                components, weights, J_list = four_test_battery_suite(suite, horizon=None, n_ics=3)
                validation_dict["e_short_by_method"] = components["E_short"]
                validation_dict["e_long_by_method"] = components["E_long"]
                validation_dict["c_by_method"] = components["C"]
                validation_dict["s_by_method"] = components["S"]
                validation_dict["surrogate_j_by_method"] = J_list
                validation_dict["best_by_surrogate_j"] = surrogate_best_method(suite, J_list=J_list)
            except Exception as e:
                import warnings
                warnings.warn(f"Surrogate battery failed: {e}", UserWarning)
        # Plots highlight discovery recommendation (AIC/BIC), never truth.
        best_for_display = validation_dict.get("best_for_display", "")
        criterion = validation_dict.get("best_for_display_criterion", "AIC (discovery)")
        print(f"\n  Consensus plots: highlighting best method = {best_for_display} ({criterion}; truth not used for selection)")
        try:
            from sindy.diagnostics_plots import (
                plot_best_method_heatmap,
                plot_consensus_comparison_with_best,
                plot_xi_agreement_heatmap,
                plot_xi_rel_err_bars,
                plot_aic_bic_bars,
                plot_term_stability_heatmap,
                run_posthoc_equation_heatmap,
                print_posthoc_table,
            )
            plot_best_method_heatmap(suite, output_dir=output_dir)
            plot_consensus_comparison_with_best(suite, output_dir=output_dir)
            plot_term_stability_heatmap(suite, output_dir=output_dir)
            if suite.get("validation", {}).get("xi_agreement_matrix") is not None:
                plot_xi_agreement_heatmap(suite, output_dir=output_dir)
            if suite.get("validation", {}).get("xi_rel_err_by_method"):
                plot_xi_rel_err_bars(suite, output_dir=output_dir)
            if suite.get("validation", {}).get("aic_by_method"):
                plot_aic_bic_bars(suite, output_dir=output_dir)
            # Post-hoc: coefficient (equation) error heatmap for best-by-ξ and table of hypotheses tried
            run_posthoc_equation_heatmap(suite, output_dir=output_dir)
            print_posthoc_table(suite)
        except Exception as e:
            print(f"Warning: Could not generate extended consensus plots: {e}")

    return suite


def run_targeted_consensus_suite(
    t: np.ndarray,
    Z_phys: np.ndarray,
    model: Any,
    budget: Dict,
    output_dir: str = "outputs/consensus/",
    Z_dot_phys: Optional[np.ndarray] = None,
    *,
    problem_class_override: Optional[str] = None,
    max_configs: int = 8,
    run_surrogate_battery: bool = False,
) -> Dict:
    """
    Run a scout-driven shortlist of SINDy configs (in-between auto and full consensus).
    Uses scout + problem_class to pick a small set of (solver, sparsity, data) configs
    (typically 4–8 runs). For real-world problems this usually includes Pareto + Ensemble
    with target_densest and baseline + scale_zscore, giving better results than a single
    auto run without the cost of run_extended_consensus_suite (13+ runs).
    Returns the same suite structure as run_extended_consensus_suite (modes, validation,
    consensus reports, term-stability heatmap, etc.).
    """
    baseline_config = _config_from_dims("Pareto", "target_densest", "baseline")
    scout = run_scout(t, Z_phys, model, baseline_config, budget, Z_dot_phys=Z_dot_phys)
    derivative_agreement = None
    if Z_dot_phys is not None:
        derivative_agreement = compute_derivative_agreement(
            t, Z_phys, Z_dot_phys,
            scaler_kind=baseline_config.scaler_kind,
            savgol_window=baseline_config.savgol_window,
            savgol_poly=baseline_config.savgol_poly,
        )
    shortlist, reason, problem_class = recommend_consensus_shortlist(
        scout,
        Z_dot_phys=Z_dot_phys,
        model=model,
        derivative_agreement=derivative_agreement,
        problem_class_override=problem_class_override,
        max_configs=max_configs,
    )
    methodologies = {}
    data_opts_seen: List[str] = []
    print(f"Targeted consensus: {len(shortlist)} configs ({problem_class}) — {reason}")
    for solver, sparsity, data in shortlist:
        name = _method_name(solver, sparsity, data)
        methodologies[name] = _config_from_dims(solver, sparsity, data)
        if data not in data_opts_seen:
            data_opts_seen.append(data)
        cfg_reason = get_config_reason(solver, sparsity, data, problem_class)
        print(f"  • {name}: {cfg_reason}")
    to_run = list(methodologies.keys())
    return run_extended_consensus_suite(
        t=t,
        Z_phys=Z_phys,
        model=model,
        budget=budget,
        output_dir=output_dir,
        include_data_processing=True,
        run_all_combos=False,
        Z_dot_phys=Z_dot_phys,
        custom_to_run=to_run,
        custom_methodologies=methodologies,
        custom_data_opts=data_opts_seen if data_opts_seen else None,
        run_surrogate_battery=run_surrogate_battery,
    )


def print_consensus_report(report: Dict) -> None:
    """
    Print the consensus report (true reference, core terms, unstable/ambiguous, missing true, notes).
    Use this with suite["report"] after run_consensus_suite.
    """
    print("\n" + "=" * 80)
    print("              SINDy MULTI-MODE CONSENSUS REPORT                ")
    print("=" * 80)
    sections = [
        ("true_coefficients", "TRUE COEFFICIENTS (reference, decimal)"),
        ("core_terms", "CORE TERMS (Agreed & Stable)"),
        ("unstable_terms", "UNSTABLE / AMBIGUOUS"),
        ("missing_true_terms", "MISSING TRUE TERMS"),
        ("notes", "NOTES"),
    ]
    for key, title in sections:
        print(f"\n{title}:")
        items = report.get(key) or []
        if not items:
            print("   None.")
        else:
            for line in items:
                print(f"   * {line}")
    print("\n" + "=" * 80)


def print_extended_consensus_reports(suite: Dict) -> None:
    """
    Print solver, pareto-method, and data-processing consensus reports from
    run_extended_consensus_suite().
    """
    consensus = suite.get("consensus", {})
    val = suite.get("validation", {})
    best_overall = val.get("best_overall", "—")
    best_per_state = val.get("best_per_state", [])
    target_names = val.get("target_names", [])

    print("\n" + "=" * 80)
    print("  EXTENDED CONSENSUS: best method and reports by group")
    print("=" * 80)

    # Summary: discovery uses AIC (never truth); ξ is post-hoc only
    best_by_xi = val.get("best_by_xi_err")
    best_by_aic = val.get("best_by_aic")
    best_by_bic = val.get("best_by_bic")
    best_discovery = val.get("best_for_discovery", best_by_aic or best_overall)
    print("\n  Summary — best method by criterion:")
    print(f"    Discovery recommendation (no truth): {best_discovery} (AIC/BIC)")
    print(f"    By R² (curve fit):           {best_overall}")
    if best_by_xi:
        print(f"    By ‖ξ−ξ_true‖ (post-hoc only): {best_by_xi}")
    if best_overall != best_discovery and val.get("overfit_warning"):
        print(f"    ⚠ {val['overfit_warning']}")
    best_for_display = val.get("best_for_display", best_discovery)
    criterion = val.get("best_for_display_criterion", "AIC (discovery)")
    print(f"    Saved images highlight: {best_for_display} ({criterion})")
    print()

    print("Best method per state (by R²):")
    if best_per_state and target_names:
        for name, best in zip(target_names, best_per_state):
            print(f"   {name}: {best}")
    else:
        print("   (n/a)")

    # ξ vs ξ_true: one number per method — low = recovered equations, high = curve-fitting
    xi_rel_err = val.get("xi_rel_err_by_method", [])
    method_names = val.get("method_names", [])
    best_by_xi = val.get("best_by_xi_err")
    if xi_rel_err and len(xi_rel_err) == len(method_names):
        print("\n  ξ vs ξ_true — relative Frobenius error (lower = equations recovered, not just curve-fitting):")
        print("  ‖ξ_proposed − ξ_true‖_F / ‖ξ_true‖_F")
        pairs = list(zip(method_names, xi_rel_err))
        pairs.sort(key=lambda p: (float("inf") if not np.isfinite(p[1]) else p[1], p[0]))
        for name, err in pairs:
            e_str = f"{float(err):.4f}" if np.isfinite(err) else "—"
            mark = "  ← best" if name == best_by_xi else ""
            print(f"    {name}: {e_str}{mark}")
        if best_by_xi:
            print(f"  Best (closest to true ξ): {best_by_xi}")

    # No-truth metrics: AIC/BIC (lower = better; no ground truth needed)
    aic_list = val.get("aic_by_method", [])
    bic_list = val.get("bic_by_method", [])
    if aic_list and len(aic_list) == len(method_names):
        print("\n  No-truth: AIC / BIC (lower = better parsimony vs fit):")
        best_aic = val.get("best_by_aic")
        best_bic = val.get("best_by_bic")
        for i, name in enumerate(method_names):
            a, b = (aic_list[i], bic_list[i]) if i < len(aic_list) else (np.nan, np.nan)
            a_str = f"{float(a):.1f}" if np.isfinite(a) else "—"
            b_str = f"{float(b):.1f}" if np.isfinite(b) else "—"
            marks = []
            if name == best_aic:
                marks.append("best AIC")
            if name == best_bic:
                marks.append("best BIC")
            mark = "  ← " + ", ".join(marks) if marks else ""
            print(f"    {name}: AIC={a_str}  BIC={b_str}{mark}")
        if best_aic or best_bic:
            print(f"  (Use these when ξ_true is unknown to compare methods.)")

    best_per_state_xi = val.get("best_per_state_xi")
    if best_per_state_xi and target_names:
        print("\nBest method per state (by per-equation agreement):")
        for name, best in zip(target_names, best_per_state_xi):
            print(f"   {name}: {best}")
    print()

    sections = [
        ("true_coefficients", "True coefficients (reference)"),
        ("core_terms", "Core terms (agreed)"),
        ("unstable_terms", "Unstable / ambiguous"),
        ("missing_true_terms", "Missing true terms"),
        ("notes", "Notes"),
    ]
    for group_name, key in [("Solver", "solver"), ("Sparsity", "sparsity"), ("Data processing", "data_processing")]:
        group = consensus.get(key, {})
        methods = group.get("methods", [])
        report = group.get("report", {})
        if not methods:
            continue
        print("\n" + "-" * 60)
        print(f"  {group_name.upper()} CONSENSUS (methods: {methods})")
        print("-" * 60)
        if report:
            for skey, stitle in sections:
                items = report.get(skey) or []
                if not items:
                    continue
                print(f"\n  {stitle}:")
                for line in items[:15]:
                    print(f"    * {line}")
                if len(items) > 15:
                    print(f"    ... and {len(items) - 15} more")
        print()

    # Data-processing reasons (from scout): why/why not each option when not sweeping data
    dp_reasons = suite.get("data_processing_reasons", [])
    include_dp = suite.get("include_data_processing", True)
    if dp_reasons:
        print("\n" + "-" * 60)
        if include_dp:
            print("  DATA PROCESSING — diagnostic reasons (for reference)")
        else:
            print("  DATA PROCESSING — not run (solver+sparsity only); reasons below")
        print("-" * 60)
        for opt, reason in dp_reasons:
            print(f"    {opt}: {reason}")
        print()
    print("=" * 80)