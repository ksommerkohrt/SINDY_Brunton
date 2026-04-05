"""
Automated pipeline config: use diagnostics (scout or post-fit) to recommend
SINDyRunConfig overrides more likely to work for the current problem.

Usage:
  - Scout: build Theta once with a base config, run excitation_report, then
    recommend_config(scout_result, base_config, Z_dot_phys, model) → (overrides, reasons).
  - Pipeline applies overrides via replace(base_config, **overrides) and re-runs or
    runs once with the recommended config.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Thresholds used in decision rules (tunable)
COND_WORSE_THAN = 1e5
EFF_RANK_FRAC_LOW = 0.5
N_HIGH_CORR_WORSE_THAN = 5
SMALL_N_TARGETS = 6

# Problem-class inference: when to treat as "toy-like" (paper-like settings) vs "real_world"
STATE_STD_RATIO_TOY = 20.0   # below this → scales are similar (toy-like)
COND_GOOD = 1e4             # below this → well-conditioned Θ (toy-like)
DERIV_AGREEMENT_TOY = 0.95  # min correlation(SavGol ẋ, Z_dot_phys) to consider derivatives "clean"
REAL_WORLD_N_TARGETS_MIN = 4  # no exact derivatives + n_targets >= this → real_world (e.g. aircraft)


def infer_problem_class(
    scout: Dict[str, Any],
    Z_dot_phys: Optional[Any] = None,
    model: Optional[Any] = None,
    derivative_agreement: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Infer problem_class from data/scout so auto-selector can branch preprocessing and solver.

    Returns
    -------
    problem_class : "toy_like" | "real_world"
    reason : short human-readable reason (for logging).
    """
    cond = scout.get("cond", 0.0)
    eff_rank_frac = scout.get("effective_rank_frac", 1.0)
    n_high_corr_theta = scout.get("n_high_corr_pairs_theta", 0)
    state_std_ratio = scout.get("state_std_ratio", 1.0)
    has_exact_derivatives = Z_dot_phys is not None
    n_targets = len(getattr(model, "target_names", [])) if model else 0

    # Strong toy signal: exact derivatives that agree well with estimated (clean data)
    if has_exact_derivatives and derivative_agreement is not None:
        if derivative_agreement >= DERIV_AGREEMENT_TOY and n_targets <= SMALL_N_TARGETS:
            return (
                "toy_like",
                f"Exact derivatives with high agreement ({derivative_agreement:.2f}), n_targets={n_targets}.",
            )
        if derivative_agreement < DERIV_AGREEMENT_TOY:
            return (
                "real_world",
                f"Exact derivatives but low agreement with estimates ({derivative_agreement:.2f}) → treat as real-world.",
            )

    # No exact derivatives: multi-target systems (e.g. aircraft) → real_world
    if n_targets >= REAL_WORLD_N_TARGETS_MIN:
        return (
            "real_world",
            f"No exact derivatives and n_targets={n_targets} ≥ {REAL_WORLD_N_TARGETS_MIN} → real-world (e.g. aircraft).",
        )
    # No exact derivatives, small system: use Θ and state diagnostics
    cond_ok = cond == cond and 0 < cond <= COND_GOOD  # finite and good
    rank_ok = eff_rank_frac >= EFF_RANK_FRAC_LOW
    few_collinear = n_high_corr_theta < N_HIGH_CORR_WORSE_THAN
    scales_similar = state_std_ratio <= STATE_STD_RATIO_TOY

    toy_score = sum([cond_ok, rank_ok, few_collinear, scales_similar])
    if toy_score >= 3 and n_targets <= SMALL_N_TARGETS:
        return (
            "toy_like",
            f"Data look clean: cond_ok={cond_ok}, rank_ok={rank_ok}, few_collinear={few_collinear}, "
            f"scales_similar={scales_similar}, n_targets={n_targets}.",
        )
    return (
        "real_world",
        f"Data suggest real-world: cond_ok={cond_ok}, rank_ok={rank_ok}, few_collinear={few_collinear}, "
        f"scales_similar={scales_similar} (state_std_ratio={state_std_ratio:.1f}).",
    )


def compute_derivative_agreement(
    t: np.ndarray,
    Z_phys: np.ndarray,
    Z_dot_phys: np.ndarray,
    *,
    scaler_kind: str = "maxabs",
    savgol_window: int = 7,
    savgol_poly: int = 3,
) -> float:
    """
    Compare SavGol-estimated derivatives to exact Z_dot_phys (e.g. from known ODE).
    Returns the minimum correlation across state dimensions (worst agreement).
    Use in infer_problem_class when Z_dot_phys is provided to decide toy vs real.
    """
    from idtools.preprocess import preprocess_timeseries

    Z_phys = np.asarray(Z_phys, dtype=float)
    Z_dot_phys = np.asarray(Z_dot_phys, dtype=float)
    t = np.asarray(t).flatten()
    if Z_phys.shape[0] != t.size or Z_dot_phys.shape != Z_phys.shape:
        return 0.0
    prep = preprocess_timeseries(
        Z_phys,
        t=t,
        scaler_kind=scaler_kind,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly,
        deriv=1,
        compute_derivatives=True,
        X_dot_phys=None,
    )
    scale = np.asarray(prep.scaler.scale_, dtype=float).reshape(1, -1)
    scale = np.where(scale == 0, 1.0, scale)
    Z_dot_exact_scaled = Z_dot_phys / scale
    X_dot_sav = prep.X_dot_scaled
    if X_dot_sav is None:
        return 0.0
    n_vars = Z_phys.shape[1]
    corrs = []
    for j in range(n_vars):
        a, b = X_dot_sav[:, j], Z_dot_exact_scaled[:, j]
        if np.std(a) < 1e-14 or np.std(b) < 1e-14:
            corrs.append(0.0)
            continue
        c = np.corrcoef(a, b)[0, 1]
        corrs.append(float(c) if np.isfinite(c) else 0.0)
    return float(np.min(corrs)) if corrs else 0.0


def get_solver_and_preprocessing_by_type(problem_class: str) -> Dict[str, Any]:
    """
    Concise hypothesis: which solver and preprocessing to use per problem type (no exhaustive search).
    Use this to interpret or document the single auto run.

    problem_class : "toy_like" | "real_world"

    Returns a dict with keys: solver, sparsity, scaler_kind, normalize_library_columns, description.
    """
    if problem_class == "toy_like":
        return {
            "solver": "pareto",
            "sparsity": "single_knee or last (paper-like when exact derivs + truth)",
            "scaler_kind": "maxabs",
            "normalize_library_columns": False,
            "description": "Paper-like: raw library, single threshold, no collinear drop, minimal ridge.",
        }
    # real_world (z-score only when state_std_ratio > 100, else maxabs)
    return {
        "solver": "pareto",
        "sparsity": "last (densest)",
        "scaler_kind": "standard if state_std_ratio>100 else maxabs",
        "normalize_library_columns": True,
        "description": "Pareto, densest pick; scaling = z-score only when state scales differ a lot, else maxabs; library normalize.",
    }


def recommend_config(
    scout: Dict[str, Any],
    Z_dot_phys: Optional[Any] = None,
    model: Optional[Any] = None,
    *,
    problem_class_override: Optional[str] = None,
    derivative_agreement: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Recommend SINDyRunConfig overrides from scout/diagnostic dict.

    Scout dict should contain (from excitation_report or diagnostic_suite):
      - data_ok, theta_ok (bool)
      - cond, effective_rank_frac (float)
      - n_high_corr_pairs_theta (int, optional)
      - Optional: scale_discrepancy_max, structure_ok (from truth), in_eq_would_differ

    problem_class_override : "toy" | "real" | None
        If "toy", allow paper-like branch when exact derivatives + small system + truth.
        If "real", skip paper-like and use robust rules only. If None, use infer_problem_class().
    derivative_agreement : float | None
        If provided (and Z_dot_phys exists), used by infer_problem_class to decide toy vs real.

    Returns
    -------
    overrides : dict
        Key-value pairs to apply to base config: replace(base_config, **overrides).
        Can include sindy_mode ("pareto" | "pareto_ensemble" | "bayes_map" | "bayes_map_ensemble")
        when conditioning is poor or structure recovery failed (ensemble modes recommended).
    reasons : list of str
        Human-readable reasons for each override (for logging/UI).
    """
    overrides: Dict[str, Any] = {}
    reasons: List[str] = []

    data_ok = scout.get("data_ok", True)
    theta_ok = scout.get("theta_ok", True)
    cond = scout.get("cond", 0.0)
    eff_rank_frac = scout.get("effective_rank_frac", 1.0)
    n_high_corr_theta = scout.get("n_high_corr_pairs_theta", 0)
    col_norm_ratio = scout.get("col_norm_ratio", 1.0)
    suggest_normalize_library = scout.get("suggest_normalize_library", True)
    has_exact_derivatives = Z_dot_phys is not None
    n_targets = len(getattr(model, "target_names", [])) if model else 0
    has_truth = bool(getattr(model, "rhs_symbolic", None))

    # Data-driven problem class (unless user overrides)
    if problem_class_override is not None:
        problem_class = "toy_like" if problem_class_override == "toy" else "real_world"
        problem_reason = f"User override: problem_class={problem_class_override}"
    else:
        problem_class, problem_reason = infer_problem_class(
            scout, Z_dot_phys=Z_dot_phys, model=model, derivative_agreement=derivative_agreement
        )
    reasons.append(f"Problem class: {problem_class} ({problem_reason})")

    # --- Paper-like: only when problem is toy-like AND we have exact derivatives + small system + truth ---
    allow_paper_like = (problem_class == "toy_like") and (
        has_exact_derivatives
        and n_targets > 0
        and n_targets <= SMALL_N_TARGETS
        and has_truth
    )
    if problem_class_override == "real":
        allow_paper_like = False
    if allow_paper_like:
        overrides["use_physical_library"] = True
        overrides["normalize_library_columns"] = False
        overrides["single_threshold"] = 0.05
        overrides["prefer_parsimony"] = False
        overrides["alpha_ridge"] = 1e-8
        overrides["pareto_pick_mode"] = "last"
        reasons.append(
            "Exact derivatives + small system with known truth + toy_like data → paper-like mode "
            "(raw library, single threshold 0.05, prefer_parsimony off, minimal ridge)."
        )
        return overrides, reasons

    # --- Real-world: densest, normalize library; z-score only when state scales differ a lot ---
    state_std_ratio = scout.get("state_std_ratio", 1.0)
    suggest_standard_scaling = scout.get("suggest_standard_scaling", False)
    if problem_class == "real_world":
        overrides["normalize_library_columns"] = True
        overrides["pareto_pick_mode"] = "last"
        if "n_thresholds" not in overrides:
            overrides["n_thresholds"] = 80
        # Z-score only when diagnostic suggests (state_std_ratio > 100); else maxabs (z-score often hurts when scales are similar)
        if suggest_standard_scaling:
            overrides["scaler_kind"] = "standard"
            reasons.append(
                f"real_world → Pareto, densest, standard scaling (state_std_ratio={state_std_ratio:.0f} > 100) + library normalize."
            )
        else:
            reasons.append(
                f"real_world → Pareto, densest, maxabs (state_std_ratio={state_std_ratio:.1f}; z-score not suggested) + library normalize."
            )

    # --- Theta normalization (toy-like or when not already set by real_world) ---
    if suggest_normalize_library and "normalize_library_columns" not in overrides:
        overrides["normalize_library_columns"] = True
        reasons.append(
            f"Library column norm ratio high ({col_norm_ratio:.1f}) → normalize_library_columns=True for fair thresholding."
        )

    # --- State/derivative scaling for toy-like: z-score only when scout suggests ---
    if problem_class != "real_world" and suggest_standard_scaling and "scaler_kind" not in overrides:
        overrides["scaler_kind"] = "standard"
        reasons.append(
            f"State std ratio high ({state_std_ratio:.1f}) → scaler_kind=standard (z-score) for comparable measurement/derivative scales."
        )

    # --- Theta ill-conditioned or many collinear pairs: keep terms, stronger ridge, denser pick, consider ensemble ---
    cond_bad = (cond != cond) or (cond > COND_WORSE_THAN)  # NaN or high
    if not theta_ok or cond_bad or eff_rank_frac < EFF_RANK_FRAC_LOW or n_high_corr_theta >= N_HIGH_CORR_WORSE_THAN:
        overrides["prefer_parsimony"] = False
        overrides["alpha_ridge"] = 1e-3
        overrides["pareto_pick_mode"] = "last"
        if "n_thresholds" not in overrides:
            overrides["n_thresholds"] = 80
        # Ensemble solver is more stable when Theta is ill-conditioned or highly collinear
        overrides["sindy_mode"] = "pareto_ensemble"
        reasons.append(
            "Poor Theta conditioning or many collinear features → keep all terms, "
            "stronger ridge (1e-3), denser model (last), more thresholds, solver=sindy_mode=pareto_ensemble."
        )

    # --- Theta OK but in-equation scale issues (if present in scout) ---
    scale_high = scout.get("scale_discrepancy_max", 0) > 0.1
    in_eq_differ = scout.get("in_eq_would_differ", False)
    if (scale_high or in_eq_differ) and "pareto_pick_mode" not in overrides:
        overrides["pareto_pick_mode"] = "last"
        reasons.append(
            "In-equation scale discrepancy or different nonzero pattern → "
            "prefer denser model (last) to retain weak terms."
        )

    # --- Truth known and structure failed: suggest denser / no collinear drop / ensemble solver ---
    structure_ok = scout.get("structure_ok", None)
    if structure_ok is False and has_truth:
        if "prefer_parsimony" not in overrides:
            overrides["prefer_parsimony"] = False
            overrides["alpha_ridge"] = max(overrides.get("alpha_ridge", 1e-6), 1e-3)
            reasons.append("Structure recovery failed (truth known) → prefer_parsimony off, stronger ridge.")
        if "pareto_pick_mode" not in overrides:
            overrides["pareto_pick_mode"] = "last"
            overrides["n_thresholds"] = overrides.get("n_thresholds", 80)
            reasons.append("Structure recovery failed → denser model (last) and more thresholds.")
        # Try ensemble for more stable coefficient estimates when single-run structure failed
        if "sindy_mode" not in overrides:
            overrides["sindy_mode"] = "pareto_ensemble"
            reasons.append("Structure recovery failed → try sindy_mode=pareto_ensemble for more stable terms.")

    # --- Data excitation poor: cannot fix by config; add reason only ---
    if not data_ok:
        reasons.append(
            "Data excitation poor (flat/low-std or highly correlated states). "
            "Consider more/better data or check state selection; config overrides may not fix it."
        )

    return overrides, reasons


def flatten_scout_from_excitation(excitation_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a flat scout dict from excitation_report() output for use with recommend_config().
    """
    pf = excitation_result.get("pass_fail", {})
    theta_stats = excitation_result.get("theta", {}).get("stats", {})
    theta_data = excitation_result.get("theta", {})
    top_theta = theta_data.get("high_corr_pairs", []) or theta_data.get("top_corr_pairs", [])
    data_section = excitation_result.get("data", {})

    return {
        "data_ok": bool(pf.get("data_ok", True)),
        "theta_ok": bool(pf.get("theta_ok", True)),
        "cond": float(theta_stats.get("cond", 0.0)),
        "effective_rank": int(theta_stats.get("effective_rank", 0)),
        "effective_rank_frac": float(theta_stats.get("effective_rank_frac", 1.0)),
        "n_high_corr_pairs_theta": len(top_theta),
        "col_norm_ratio": float(theta_stats.get("col_norm_ratio", 1.0)),
        "suggest_normalize_library": bool(theta_stats.get("suggest_normalize_library", True)),
        "state_std_ratio": float(data_section.get("state_std_ratio", 1.0)),
        "suggest_standard_scaling": bool(data_section.get("suggest_standard_scaling", False)),
    }


def flatten_scout_from_diagnostic_suite(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a flat scout dict from run_diagnostic_suite() report for use with recommend_config().
    """
    ex = report.get("excitation", {})
    truth = report.get("truth", {})
    scale = report.get("scale", {})

    scout = {
        "data_ok": ex.get("data_ok", True),
        "theta_ok": ex.get("theta_ok", True),
        "cond": ex.get("cond", 0.0),
        "effective_rank_frac": ex.get("effective_rank_frac", 1.0),
        "n_high_corr_pairs_theta": ex.get("n_high_corr_pairs_theta", 0),
        "col_norm_ratio": ex.get("col_norm_ratio", 1.0),
        "suggest_normalize_library": ex.get("suggest_normalize_library", True),
        "state_std_ratio": ex.get("state_std_ratio", 1.0),
        "suggest_standard_scaling": ex.get("suggest_standard_scaling", False),
    }
    if truth:
        scout["structure_ok"] = truth.get("structure_ok")
    if scale and "error" not in scale:
        scout["scale_discrepancy_max"] = scale.get("scale_discrepancy_max", 0)
        scout["in_eq_would_differ"] = scale.get("in_eq_would_differ", False)
    return scout


def get_config_reason(solver: str, sparsity: Optional[str], data: str, problem_class: str) -> str:
    """
    Human-readable reason this config is in the shortlist (for dynamics discovery).
    Explains why each candidate is tested—not arbitrary.
    """
    parts: List[str] = []
    # Solver
    if solver == "Pareto":
        parts.append("Pareto (STLSQ threshold sweep): standard discovery")
    elif solver == "Ensemble":
        parts.append("Ensemble (bootstrap median): stable coefficients when Θ ill-conditioned or collinear")
    elif solver == "BayesMAP":
        parts.append("Bayes MAP: regularization when data are noisy")
    elif solver == "BayesEnsemble":
        parts.append("Bayes ensemble: robust when Θ has many collinear columns")
    # Sparsity
    if sparsity == "target_densest":
        parts.append("densest pick to avoid over-sparsifying (retains weak terms)")
    elif sparsity == "single_knee":
        parts.append("single-threshold knee (paper-like for toy systems)")
    elif sparsity == "target_knee":
        parts.append("per-target knee")
    elif sparsity == "target_sparsest":
        parts.append("sparsest pick (aggressive sparsity)")
    # Data
    if data == "baseline":
        parts.append("baseline scaling (maxabs) as reference")
    elif data == "scale_zscore":
        parts.append("z-score when state scales differ (often improves equation recovery in real-world)")
    return " | ".join(parts) if parts else f"{solver} {sparsity or ''} {data}"


def recommend_consensus_shortlist(
    scout: Dict[str, Any],
    Z_dot_phys: Optional[Any] = None,
    model: Optional[Any] = None,
    *,
    derivative_agreement: Optional[float] = None,
    problem_class_override: Optional[str] = None,
    max_configs: int = 8,
) -> Tuple[List[Tuple[str, Optional[str], str]], str, str]:
    """
    Recommend a short list of (solver, sparsity, data) configs to run instead of full consensus.
    Uses problem_class and scout diagnostics. Each config has a clear rationale (see get_config_reason).

    Returns
    -------
    shortlist : list of (solver, sparsity, data)
        Triples to run. sparsity is None for Bayes solvers.
    reason : str
        Short overall explanation (e.g. "real_world: Pareto+Ensemble densest, scale_zscore+baseline").
    problem_class : str
        "toy_like" or "real_world" for use with get_config_reason.
    """
    problem_class, reason = (
        ("toy_like" if problem_class_override == "toy" else "real_world", f"override={problem_class_override}")
        if problem_class_override
        else infer_problem_class(scout, Z_dot_phys=Z_dot_phys, model=model, derivative_agreement=derivative_agreement)
    )
    suggest_scale = scout.get("suggest_standard_scaling", False)
    suggest_norm = scout.get("suggest_normalize_library", False)
    cond = scout.get("cond", 0.0)
    n_high_corr = scout.get("n_high_corr_pairs_theta", 0)
    theta_bad = cond != cond or cond > COND_WORSE_THAN or n_high_corr >= N_HIGH_CORR_WORSE_THAN

    shortlist: List[Tuple[str, Optional[str], str]] = []
    # Data options to include: baseline always; add scale_zscore for real_world or when scout suggests
    data_opts_short = ["baseline"]
    if problem_class == "real_world" or suggest_scale or suggest_norm:
        data_opts_short.append("scale_zscore")

    if problem_class == "toy_like":
        # Toy: Pareto with paper-like (single or densest), baseline; optionally one ensemble
        shortlist.append(("Pareto", "single_knee", "baseline"))
        shortlist.append(("Pareto", "target_densest", "baseline"))
        if len(shortlist) < max_configs:
            shortlist.append(("Ensemble", "target_densest", "baseline"))
        reason = f"toy_like: Pareto single_knee + densest baseline; {reason}"
        return shortlist[:max_configs], reason, problem_class
    else:
        # Real-world: Pareto + Ensemble with target_densest (denser pick), baseline + scale_zscore
        for data in data_opts_short:
            shortlist.append(("Pareto", "target_densest", data))
            shortlist.append(("Ensemble", "target_densest", data))
        # If room and Θ is bad, add one Bayes variant for stability
        if len(shortlist) < max_configs and theta_bad:
            shortlist.append(("BayesEnsemble", None, data_opts_short[-1]))
        reason = f"real_world: Pareto+Ensemble target_densest, data={data_opts_short}; {reason}"

    return shortlist[:max_configs], reason, problem_class
