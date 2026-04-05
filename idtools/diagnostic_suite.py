"""
Unified diagnostic suite for SINDy identification: combines truth comparison, excitation,
in-equation scale checks, validation, and collinearity into one cohesive report.

Use for:
  - Single-run diagnostics: run_diagnostic_suite(res) → one dict of all metrics.
  - Sensitivity analysis: run_sensitivity_analysis(t, Z_phys, model, param_grid) → table of
    metrics vs independent variables (collinear_threshold, pareto_pick_mode, etc.) to see
    how correlation/scale affect Lorenz (or any system) identification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from idtools.compare_to_truth import compare_to_truth
from idtools.excitation_report import (
    ExcitationReportConfig,
    excitation_report,
    format_excitation_report,
)
from idtools.in_equation_scale import fit_target_with_in_equation_scaling


@dataclass
class DiagnosticSuiteConfig:
    """Options for the unified diagnostic suite."""
    # Excitation report
    excitation_config: Optional[ExcitationReportConfig] = None
    run_excitation: bool = True
    # In-equation scale (per-target): compare raw SINDy coefs vs in-equation-scaled LS
    run_scale_check: bool = True
    scale_threshold: float = 0.01  # threshold in scaled space for in_eq comparison
    # Truth comparison (when model has rhs_symbolic)
    run_truth: bool = True
    # Collinearity summary from fit
    include_collinear_summary: bool = True
    # Optional: run diagnostic plots (dashboard, correlation heatmap, etc.)
    run_plots: bool = False
    plots_output_dir: str = "outputs/"


def run_diagnostic_suite(
    res: Dict[str, Any],
    config: Optional[DiagnosticSuiteConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all diagnostics on a single pipeline result and return one cohesive report dict.

    Aggregates:
      - compare_to_truth: coef_correlation, max_rel_error_nz, mean_rel_error_nz
      - excitation_report: data_ok, theta_ok, cond(Theta), effective_rank_frac, top corr pairs
      - Per-target in-equation scale: scale_discrepancy (would in_eq change nonzero terms?)
      - validation: r2_mean, r2_by_state
      - collinear_dropped count and sample

    Returns
    -------
    report : dict
        Flat dict of metrics suitable for sensitivity tables:
        - truth: coef_correlation, max_rel_error_nz, mean_rel_error_nz, structure_ok
        - excitation: data_ok, theta_ok, cond, effective_rank_frac, n_high_corr_pairs_data, n_high_corr_pairs_theta
        - scale: scale_discrepancy_max, scale_discrepancy_by_target, in_eq_would_differ
        - validation: r2_mean, r2_by_state (list)
        - collinearity: n_collinear_dropped, collinear_dropped_sample
        - config_label: optional, set by caller for sensitivity runs
    """
    cfg = config or DiagnosticSuiteConfig()
    fit = res["fit"]
    model = res["model"]
    t = np.asarray(res["t"]).ravel()
    Z = np.asarray(res["Z"])
    val = res.get("validation", {})

    report: Dict[str, Any] = {
        "validation": {
            "r2_mean": float(val.get("r2_mean", np.nan)),
            "r2_by_state": list(val.get("r2_by_state", [])),
        },
        "collinearity": _collinear_summary(fit) if cfg.include_collinear_summary else {},
    }

    # --- Truth comparison ---
    if cfg.run_truth and getattr(model, "rhs_symbolic", None):
        try:
            truth = compare_to_truth(res, verbose=verbose)
            report["truth"] = {
                "coef_correlation": float(truth.get("coef_correlation", np.nan)),
                "max_rel_error_nz": float(truth.get("max_rel_error_nz", np.nan)),
                "mean_rel_error_nz": float(truth.get("mean_rel_error_nz", np.nan)),
                "structure_ok": truth.get("structure_ok", False),
            }
        except Exception as e:
            report["truth"] = {"error": str(e)}
    else:
        report["truth"] = {}

    # --- Excitation (data + Theta) ---
    if cfg.run_excitation:
        try:
            Theta = fit.get("Theta_clean")
            kept_names = fit.get("kept_names", [])
            Y_phys = fit.get("Y_phys")
            if Theta is not None and kept_names is not None:
                ex_config = cfg.excitation_config or ExcitationReportConfig()
                ex = excitation_report(
                    t=t,
                    Z=Z,
                    names=model.measured_names,
                    Theta=Theta,
                    feature_names=kept_names,
                    Y=Y_phys,
                    config=ex_config,
                )
                pf = ex.get("pass_fail", {})
                theta_stats = ex.get("theta", {}).get("stats", {})
                data_section = ex.get("data", {})
                report["excitation"] = {
                    "data_ok": bool(pf.get("data_ok", False)),
                    "theta_ok": bool(pf.get("theta_ok", False)),
                    "ok": bool(pf.get("ok", False)),
                    "cond": float(theta_stats.get("cond", np.nan)),
                    "effective_rank": int(theta_stats.get("effective_rank", 0)),
                    "effective_rank_frac": float(theta_stats.get("effective_rank_frac", np.nan)),
                    "n_high_corr_pairs_data": len(data_section.get("top_corr_pairs", [])),
                    "n_high_corr_pairs_theta": len(
                        ex.get("theta", {}).get("top_corr_pairs", [])
                    ),
                    "col_norm_ratio": float(theta_stats.get("col_norm_ratio", 1.0)),
                    "suggest_normalize_library": bool(theta_stats.get("suggest_normalize_library", True)),
                    "state_std_ratio": float(data_section.get("state_std_ratio", 1.0)),
                    "suggest_standard_scaling": bool(data_section.get("suggest_standard_scaling", False)),
                    "warnings": list(ex.get("warnings", [])),
                }
                if verbose:
                    print(format_excitation_report(ex, max_lines=60))
            else:
                report["excitation"] = {"error": "Theta_clean or kept_names missing"}
        except Exception as e:
            report["excitation"] = {"error": str(e)}
    else:
        report["excitation"] = {}

    # --- In-equation scale check (per target) ---
    if cfg.run_scale_check:
        scale_metrics = _scale_check(
            fit,
            threshold=cfg.scale_threshold,
        )
        report["scale"] = scale_metrics
    else:
        report["scale"] = {}

    # Flatten for sensitivity table (optional): top-level keys for easy column access
    flat = _flatten_report(report)
    report["_flat"] = flat

    if verbose and report.get("truth"):
        t = report["truth"]
        if "error" not in t:
            print(
                f"  Truth: coef_r={t.get('coef_correlation', np.nan):.4f} "
                f"max_rel_err_nz={t.get('max_rel_error_nz', np.nan):.4f} "
                f"structure_ok={t.get('structure_ok', False)}"
            )
    if verbose and report.get("scale"):
        s = report["scale"]
        if "error" not in s:
            print(
                f"  Scale: discrepancy_max={s.get('scale_discrepancy_max', np.nan):.4f} "
                f"in_eq_would_differ={s.get('in_eq_would_differ', False)}"
            )

    if cfg.run_plots:
        try:
            from sindy.diagnostics_plots import run_diagnostic_dashboard
            run_diagnostic_dashboard(res, output_dir=cfg.plots_output_dir)
        except Exception as e:
            report["plots_error"] = str(e)

    return report


def _collinear_summary(fit: Dict) -> Dict[str, Any]:
    dropped = fit.get("collinear_dropped", [])
    if not dropped:
        return {"n_collinear_dropped": 0, "collinear_dropped_sample": []}
    # dropped is list of (dropped_name, kept_name, r) or similar
    sample = dropped[:10] if len(dropped) > 10 else dropped
    return {
        "n_collinear_dropped": len(dropped),
        "collinear_dropped_sample": [str(x) for x in sample],
    }


def _scale_check(
    fit: Dict,
    threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    For each target, compare SINDy coef_kept to in-equation-scaled LS.
    Returns scale_discrepancy_max, per-target discrepancies, and whether in_eq would change nonzero pattern.
    """
    Theta = fit.get("Theta_clean")
    Y = fit.get("Y_phys")
    coef_kept = np.asarray(fit.get("coef_kept", np.zeros((0, 0))))
    names = fit.get("kept_names", [])

    if Theta is None or Y is None or coef_kept.size == 0:
        return {"error": "Theta_clean, Y_phys, or coef_kept missing"}

    Theta = np.asarray(Theta, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_targets = Y.shape[1]

    discrepancy_by_target: List[float] = []
    in_eq_diffs: List[bool] = []

    for j in range(n_targets):
        try:
            coef_sindy = coef_kept[:, j]
            coef_in_eq, scale_fac, info = fit_target_with_in_equation_scaling(
                Theta, Y, target_index=j, threshold=threshold, feature_names=names
            )
            # Relative difference (avoid div by zero)
            denom = np.maximum(np.abs(coef_sindy), 1e-12)
            rel_diff = np.abs(coef_sindy - coef_in_eq) / (denom + 1e-20)
            disc = float(np.max(rel_diff))
            discrepancy_by_target.append(disc)

            # Would in_eq change which terms are nonzero?
            nz_sindy = np.abs(coef_sindy) > 1e-10
            nz_in_eq = np.abs(coef_in_eq) > 1e-10
            in_eq_diffs.append(bool(np.any(nz_sindy != nz_in_eq)))
        except Exception:
            discrepancy_by_target.append(float("nan"))
            in_eq_diffs.append(False)

    return {
        "scale_discrepancy_max": float(np.nanmax(discrepancy_by_target))
        if discrepancy_by_target
        else np.nan,
        "scale_discrepancy_by_target": discrepancy_by_target,
        "in_eq_would_differ": any(in_eq_diffs),
    }


def _flatten_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """One-level dict for table columns: truth.coef_correlation -> coef_correlation, etc."""
    flat: Dict[str, Any] = {}
    for section, data in report.items():
        if section.startswith("_") or not isinstance(data, dict):
            continue
        for k, v in data.items():
            if k == "r2_by_state":
                flat["r2_by_state"] = v
                continue
            if k == "warnings" or k == "collinear_dropped_sample":
                continue
            flat[k] = v
    return flat


# ---------------------------------------------------------------------------
# Sensitivity analysis: vary independent variables and collect diagnostic_suite per run
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    t: np.ndarray,
    Z_phys: np.ndarray,
    model: Any,
    param_grid: Dict[str, Sequence[Any]],
    base_config: Optional[Any] = None,
    budget: Optional[Dict[str, Dict[str, int]]] = None,
    suite_config: Optional[DiagnosticSuiteConfig] = None,
    run_diagnostics_plots: bool = False,
    compare_truth: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the pipeline over a grid of independent variables and collect the full
    diagnostic suite for each run. Use this to see how coef_correlation,
    scale_discrepancy, excitation pass/fail, etc. respond to e.g. collinear_threshold,
    pareto_pick_mode, constant_cv_thresh (especially for Lorenz / high correlation).

    Parameters
    ----------
    t, Z_phys, model : as for run_sindy_pipeline_general
    param_grid : dict of list of values, e.g.
        {"collinear_threshold": [0.90, 0.95, 0.99],
         "pareto_pick_mode": ["per_target_knee", "last"]}
        Keys must be attributes of SINDyRunConfig (e.g. prefer_parsimony, collinear_threshold).
    base_config : optional SINDyRunConfig; defaults to SINDyRunConfig()
    budget : optional library budget
    suite_config : options for run_diagnostic_suite
    run_diagnostics_plots : if True, run full diagnostic dashboard for each run (slow)
    compare_truth : passed to diagnostic suite (run_truth)

    Returns
    -------
    results_list : list of dicts, each with keys: label, config, res, report
    summary_table : list of dicts, each row = one run with param columns + all _flat metrics
    """
    from dataclasses import replace
    from itertools import product

    from sindy.pipeline import SINDyRunConfig, run_sindy_pipeline_general

    base_config = base_config or SINDyRunConfig()
    suite_config = suite_config or DiagnosticSuiteConfig()
    suite_config = DiagnosticSuiteConfig(
        run_plots=run_diagnostics_plots,
        run_truth=compare_truth,
        run_excitation=True,
        run_scale_check=True,
    )

    # Build list of (label, config, combo) from param_grid
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    configs: List[Tuple[str, Any, Tuple[Any, ...]]] = []
    for combo in product(*values):
        cfg = replace(base_config, **dict(zip(keys, combo)))
        label = "_".join(f"{k}={v}" for k, v in zip(keys, combo))
        configs.append((label, cfg, combo))

    results_list: List[Dict[str, Any]] = []
    summary_table: List[Dict[str, Any]] = []

    for label, config, combo in configs:
        res = run_sindy_pipeline_general(
            t=t,
            Z_phys=Z_phys,
            model=model,
            config=config,
            budget=budget,
            run_diagnostics=False,
        )
        report = run_diagnostic_suite(
            res,
            config=suite_config,
            verbose=False,
        )

        results_list.append({
            "label": label,
            "config": config,
            "res": res,
            "report": report,
        })

        row: Dict[str, Any] = {"label": label, "config": config}
        for k, v in zip(keys, combo):
            row[k] = v
        # All flat metrics from diagnostic suite
        flat = report.get("_flat", _flatten_report(report))
        for k, v in flat.items():
            row[k] = v
        summary_table.append(row)

    return results_list, summary_table


def print_sensitivity_summary(
    summary_table: List[Dict[str, Any]],
    sort_by: str = "coef_correlation",
    descending: bool = True,
    max_rows: int = 30,
) -> None:
    """Print sensitivity table with one row per run; sort by metric (e.g. coef_correlation)."""
    if not summary_table:
        print("No sensitivity results.")
        return
    key = sort_by
    # default so missing values sort last: use -inf when descending (high first), +inf when ascending (low first)
    default = -np.inf if descending else np.inf
    sorted_table = sorted(
        summary_table,
        key=lambda r: r.get(key, default),
        reverse=descending,
    )
    sorted_table = sorted_table[:max_rows]
    print(f"Sensitivity summary (sorted by {sort_by}, descending={descending}):")
    print("-" * 100)
    for r in sorted_table:
        parts = [f"  {r.get('label', '')}"]
        for k in ["coef_correlation", "max_rel_error_nz", "r2_mean", "theta_ok", "scale_discrepancy_max"]:
            if k in r:
                v = r[k]
                if isinstance(v, float):
                    parts.append(f" {k}={v:.4f}")
                else:
                    parts.append(f" {k}={v}")
        print("".join(parts))
    print("-" * 100)
