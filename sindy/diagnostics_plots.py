# sindy/diagnostics_plots.py

from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from idtools.preprocess import affine_from_sklearn_scaler
from idtools.xi_true_from_sim_equations import xi_true_from_sim_equations
from sindy.pareto import plot_pareto_frontier

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# -------------------------------------------------------------------
# Xi heatmap utility
# -------------------------------------------------------------------

def _build_row_mask_for_heatmap(
    kept_names: Sequence[str],
    Xi_true: np.ndarray,
    Xi_disc: np.ndarray,
    tiny_coeff_thresh: float = 1e-4,
) -> np.ndarray:
    """
    General mask logic: keeps '1' and any row where either the 
    true or discovered coefficient is significant.
    """
    kept_names = list(kept_names)
    Xt = np.asarray(Xi_true, float)
    Xd = np.asarray(Xi_disc, float)
    keep = np.ones(len(kept_names), dtype=bool)

    for i, nm in enumerate(kept_names):
        nm_str = nm.strip()
        if nm_str == "1":
            continue
        
        # Only drop if BOTH true and discovered coeffs are effectively zero.
        max_coeff = max(np.max(np.abs(Xt[i, :])), np.max(np.abs(Xd[i, :])))
        if max_coeff < tiny_coeff_thresh:
            keep[i] = False

    return keep

def plot_xi_heatmaps(
    kept_names: Sequence[str],
    Xi_true: np.ndarray,
    Xi_disc: np.ndarray,
    target_names: Sequence[str],  # REVISED: No longer hardcoded defaults
    output_dir: str = "outputs",
    prefix: str = "xi",
    max_rows: int = 40,
    drop_constant_like: bool = False,
    tiny_coeff_thresh: float = 1e-4,
) -> Dict[str, str]:
    """
    Plot Xi TRUE, Xi DISC, and error as heatmaps for ANY system in physical coefficient units.
    """
    _ensure_dir(output_dir)
    kept_names = list(kept_names)
    target_names = list(target_names)
    Xt = np.asarray(Xi_true, dtype=float)
    Xd = np.asarray(Xi_disc, dtype=float)

    if Xt.shape != Xd.shape:
        raise ValueError(f"shape mismatch: true {Xt.shape} vs disc {Xd.shape}")

    if drop_constant_like:
        row_mask = _build_row_mask_for_heatmap(kept_names, Xt, Xd, tiny_coeff_thresh)
        Xt = Xt[row_mask, :]
        Xd = Xd[row_mask, :]
        kept_names = [nm for nm, k in zip(kept_names, row_mask) if k]

    # Order by magnitude of discovered coefficients to show important physics at the top
    mag = np.max(np.abs(Xd), axis=1)
    order = np.argsort(-mag)
    if order.size > int(max_rows):
        order = order[: int(max_rows)]

    names = [kept_names[i] for i in order]
    Xt, Xd = Xt[order, :], Xd[order, :]
    E = Xd - Xt

    def heat(M, title, fname, cmap="coolwarm", symmetric=True):
        vmax_local = max(float(np.max(np.abs(M))), 1e-12)
        vmin_local = -vmax_local if symmetric else 0.0
        
        fig_h = max(3.5, 0.25 * len(names) + 1.5)
        fig, ax = plt.subplots(1, 1, figsize=(10, fig_h))
        im = ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap,
                       vmin=vmin_local, vmax=vmax_local)
        
        ax.set_title(title)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_xticklabels([f"$\dot{{{n}}}$" for n in target_names], rotation=45)
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    return {
        "true": heat(Xt, "Xi TRUE (physical units)", f"{prefix}_Xi_true.png"),
        "disc": heat(Xd, "Xi DISCOVERED (physical units)", f"{prefix}_Xi_disc.png"),
        "err":  heat(E, "Xi Error (Disc - True, physical units)", f"{prefix}_Xi_err.png")
    }


def _build_union_xi_matrices_for_fit(res: Dict[str, Any]) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Build Xi_true and Xi_disc in a common union support for one fit result.
    Union support = rows where either true or discovered coefficients are non-negligible.
    """
    fit = res["fit"]
    model = res["model"]
    lib = fit["library"]
    n_full = int(getattr(lib, "num_terms", 0))
    if n_full <= 0:
        raise ValueError("Library has no terms.")

    xi_true_full, _ = xi_true_from_sim_equations(
        model=model,
        lib=lib,
        scaler=fit["scaler"],
        Z_phys=res["Z"],
        kept_idx=None,
        use_physical_library=fit.get("use_physical_library", False),
    )

    xi_disc_full = np.zeros((n_full, len(fit["target_names"])), dtype=float)
    coef_phys = np.asarray(fit.get("coef_phys", fit["coef_kept"]), dtype=float)
    if coef_phys.ndim == 1:
        coef_phys = coef_phys.reshape(-1, 1)
    kept_idx = np.asarray(fit.get("kept_idx", []), dtype=int).ravel()
    for k, lib_idx in enumerate(kept_idx):
        if 0 <= int(lib_idx) < n_full and k < coef_phys.shape[0]:
            xi_disc_full[int(lib_idx), :] = coef_phys[k, :]

    tiny = 1e-8
    row_mag = np.maximum(
        np.max(np.abs(xi_true_full), axis=1),
        np.max(np.abs(xi_disc_full), axis=1),
    )
    union_mask = row_mag > tiny
    if not np.any(union_mask):
        union_mask[:] = True

    kept_names_full = list(getattr(lib, "feature_names", [f"f{i}" for i in range(n_full)]))
    kept_names_union = [nm for nm, keep in zip(kept_names_full, union_mask) if keep]
    xi_true_union = xi_true_full[union_mask, :]
    xi_disc_union = xi_disc_full[union_mask, :]
    return kept_names_union, xi_true_union, xi_disc_union


# -------------------------------------------------------------------
# Main diagnostics function
# -------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import os

def plot_honest_diagnostics(
    res: dict, 
    model: Any, # Pass the SINDySystemModel instance
    save_path: str = "outputs/diagnostics.png"
):
    """
    Generalized diagnostic plotter that respects symbolic argument ordering.
    Z must have columns in model.measured_names order; all_symbols order must
    match the order used when building rhs_lambdified (e.g. bridge uses state_syms order).
    """
    t = res["t"]
    Z = res["Z"]
    fit = res["fit"]
    val = res["validation"]
    
    target_names = fit["target_names"]
    n_targets = len(target_names)

    if Z.shape[1] != len(model.measured_names):
        raise ValueError(
            f"Data columns ({Z.shape[1]}) do not match model.measured_names ({len(model.measured_names)}). "
            "Z_phys must have one column per measured_names in the same order (e.g. full state from simulation)."
        )

    # --- Align Z columns with model.all_symbols order (same order as rhs_lambdified) ---
    symbol_to_idx = {name: i for i, name in enumerate(model.measured_names)}
    try:
        ordered_indices = [symbol_to_idx[s.name] for s in model.all_symbols]
    except KeyError as e:
        raise ValueError(
            f"Model all_symbols include a name not in measured_names: {e}. "
            "Ensure model.all_symbols order matches the order used to build rhs_lambdified."
        ) from e
    Z_ordered = Z[:, ordered_indices]

    # 1. Calculate Truth (same argument order as when rhs_lambdified was built)
    F_true = np.zeros((len(t), n_targets))
    for i, f_num in enumerate(model.rhs_lambdified):
        F_true[:, i] = np.asarray(f_num(*Z_ordered.T), dtype=float).ravel()
        
    # 2. SINDy predictions (must match how validation/fit built Theta: physical vs scaled)
    lib = fit["library"]
    aff = affine_from_sklearn_scaler(fit["scaler"])
    if fit.get("use_physical_library"):
        Theta_full = lib.transform(Z_scaled=Z, Z_phys=Z)
    else:
        Theta_full = lib.transform(Z_scaled=aff.transform(Z), Z_phys=Z)
    F_sindy = Theta_full[:, fit["kept_idx"]] @ fit["coef_kept"]

    # 3. Dynamic Plotting
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 2.5 * n_targets), sharex=True)
    if n_targets == 1: axes = [axes]

    for i in range(n_targets):
        ax = axes[i]
        ax.plot(t, F_true[:, i], 'k-', alpha=0.5, label="Ground Truth")
        ax.plot(t, F_sindy[:, i], 'r--', lw=1.5, label="SINDy Model")
        
        ax.set_ylabel(f"$\dot{{{target_names[i]}}}$")
        ax.grid(True, alpha=0.2)
        
        r2 = val["r2_by_state"][i]
        ax.set_title(f"Target: {target_names[i]} ($R^2 = {r2:.4f}$)", loc='right', fontsize=9)
        if i == 0: ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_feature_correlation(fit: dict, save_path: str = "outputs/feature_correlation.png"):
    """
    Plots a heatmap of the correlations between the features kept in the library.
    Helps identify if multi-collinearity is affecting the sparse selection.
    """
    # Theta_clean should be passed in the fit dict from pipeline.fit_sindy_main
    Theta = fit.get("Theta_clean")
    names = fit.get("kept_names")
    
    if Theta is None or names is None:
        print("Warning: Theta_clean or kept_names not found in fit dict. Skipping correlation plot.")
        return

    # Calculate Pearson correlation matrix
    corr = np.corrcoef(Theta, rowvar=False)
    
    plt.figure(figsize=(max(8, 0.4*len(names)), max(6, 0.3*len(names))))
    sns.heatmap(
        corr, 
        xticklabels=names, 
        yticklabels=names, 
        cmap="RdBu_r", 
        vmin=-1, vmax=1, center=0,
        annot=len(names) < 15, # Only annotate if the library is small
        fmt=".2f"
    )
    plt.title("Feature Correlation Matrix (Kept Library)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_diagnostic_dashboard(res: Dict, output_dir: str = "outputs/"):
    """
    Executes all diagnostic plots in one go.
    """
    _ensure_dir(output_dir)
    fit = res["fit"]
    model = res["model"]
    library = fit["library"]
    
    try:
        kept_names_union, xi_true_union, xi_disc_union = _build_union_xi_matrices_for_fit(res)
    except Exception as e:
        print(f"Warning: Could not generate Xi union matrices for heatmap. Error: {e}")
        kept_names_union, xi_true_union, xi_disc_union = None, None, None

    # 1. Plot Truth vs. SINDy (Time series trajectories)
    plot_honest_diagnostics(res, model, save_path=os.path.join(output_dir, "time_series_match.png"))
    
    # 2. Structural Verification (The Heatmaps)
    if xi_true_union is not None:
        plot_xi_heatmaps(
            kept_names=kept_names_union,
            Xi_true=xi_true_union,
            Xi_disc=xi_disc_union,
            target_names=fit["target_names"],
            output_dir=output_dir,
            prefix="structure_verification",
            drop_constant_like=True
        )
        
    # 3. Plot Library Correlation (Collinearity check)
    plot_feature_correlation(fit, save_path=os.path.join(output_dir, "library_correlation.png"))
    
    # 4. Plot Pareto Frontier (only when fit came from pareto mode and has these keys)
    if "pareto" in fit and "var_y" in fit and "best_complexity" in fit and "best_mse" in fit:
        var_y = float(np.asarray(fit["var_y"]).flat[0])
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_pareto_frontier(
            pareto_list=fit["pareto"],
            var_y=var_y,
            pick={"complexity": fit["best_complexity"], "mse": fit["best_mse"]},
            title="Model Selection: Complexity vs Error",
            ax=ax
        )
        fig.savefig(os.path.join(output_dir, "pareto_frontier.png"), dpi=150)
        plt.close(fig)


def plot_consensus_comparison(suite: Dict, output_dir: str = "outputs/consensus/") -> str:
    """
    Plot Truth vs SINDy for all methods in the consensus suite (one figure, one subplot per target).
    Each subplot shows ground truth (black) and predicted derivatives for each method (e.g. Pareto_densest, Ensemble_densest, BayesMAP, BayesEnsemble).
    """
    _ensure_dir(output_dir)
    anchor = suite.get("anchor_meta") or suite.get("anchor")
    if anchor is None:
        raise ValueError("suite must contain 'anchor_meta' with t, Z, model from the consensus run.")
    modes = suite.get("modes", {})
    if not modes:
        raise ValueError("suite must contain 'modes' (dict of method name -> fit dict).")

    t = np.asarray(anchor["t"]).ravel()
    Z = np.asarray(anchor["Z"])
    model = anchor["model"]
    target_names = list(list(modes.values())[0]["target_names"])
    n_targets = len(target_names)

    if Z.shape[1] != len(model.measured_names):
        raise ValueError(
            f"Z columns ({Z.shape[1]}) != model.measured_names ({len(model.measured_names)}). "
            "Use full state matrix from simulation."
        )
    symbol_to_idx = {name: i for i, name in enumerate(model.measured_names)}
    ordered_indices = [symbol_to_idx[s.name] for s in model.all_symbols]
    Z_ordered = Z[:, ordered_indices]

    F_true = np.zeros((len(t), n_targets))
    for i, f_num in enumerate(model.rhs_lambdified):
        F_true[:, i] = np.asarray(f_num(*Z_ordered.T), dtype=float).ravel()

    method_list = list(modes.keys())
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 2.8 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]

    for i in range(n_targets):
        ax = axes[i]
        ax.plot(t, F_true[:, i], "k-", alpha=0.7, lw=2, label="Ground Truth")
        for idx, (name, fit) in enumerate(modes.items()):
            lib = fit["library"]
            aff = affine_from_sklearn_scaler(fit["scaler"])
            if fit.get("use_physical_library"):
                Theta_full = lib.transform(Z_scaled=Z, Z_phys=Z)
            else:
                Theta_full = lib.transform(Z_scaled=aff.transform(Z), Z_phys=Z)
            F_sindy = Theta_full[:, np.asarray(fit["kept_idx"])] @ np.asarray(fit["coef_kept"])
            c = plt.cm.tab20(idx % 20) if method_list else "gray"
            ax.plot(t, F_sindy[:, i], "--", color=c, lw=1.2, label=name)
        ax.set_ylabel(f"$\\dot{{{target_names[i]}}}$")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("Consensus: Truth vs SINDy by method", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_best_method_heatmap(suite: Dict, output_dir: str = "outputs/consensus/") -> str:
    """
    Heat map: rows = methods, columns = states (targets). Cell = R² for that method/state.
    Best method per state is highlighted (e.g. bold or marker).
    """
    _ensure_dir(output_dir)
    val = suite.get("validation", {})
    r2_matrix = np.asarray(val.get("r2_matrix"), dtype=float)
    method_names = val.get("method_names", [])
    target_names = val.get("target_names", [])
    best_per_state = val.get("best_per_state", [])

    if r2_matrix.size == 0 or len(method_names) == 0 or len(target_names) == 0:
        return ""

    n_methods, n_states = r2_matrix.shape
    # Clip for display (e.g. negative R² -> 0 for colormap)
    r2_display = np.clip(r2_matrix, -0.05, 1.0)
    vmin, vmax = float(np.nanmin(r2_display)) if np.any(np.isfinite(r2_display)) else 0, 1.0

    fig, ax = plt.subplots(1, 1, figsize=(max(6, n_states * 1.2), max(4, n_methods * 0.5)))
    im = ax.imshow(r2_display, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels([f"$\\dot{{{n}}}$" for n in target_names], rotation=45, ha="right")
    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(method_names, fontsize=9)
    best_display = val.get("best_for_display") or val.get("best_overall", "")
    criterion = val.get("best_for_display_criterion", "")
    ax.set_title(f"R² by method and state (primary: {best_display}" + (f" — {criterion})" if criterion else "; per-column best by R² highlighted)"))

    for j in range(n_states):
        for i in range(n_methods):
            v = r2_matrix[i, j]
            text = f"{v:.2f}" if np.isfinite(v) else "—"
            is_best = (i < len(method_names) and j < len(best_per_state) and
                       method_names[i] == best_per_state[j])
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_best else "normal", color="black")

    plt.colorbar(im, ax=ax, label="R²", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_best_method_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_xi_rel_err_bars(suite: Dict, output_dir: str = "outputs/consensus/") -> str:
    """
    Bar chart: ‖ξ_proposed − ξ_true‖_F / ‖ξ_true‖_F per method (lower = recovered equations, not curve-fitting).
    """
    _ensure_dir(output_dir)
    val = suite.get("validation", {})
    errs = val.get("xi_rel_err_by_method", [])
    method_names = val.get("method_names", [])
    best = val.get("best_by_xi_err")
    if not errs or len(errs) != len(method_names):
        return ""
    pairs = [(n, float(e)) for n, e in zip(method_names, errs) if np.isfinite(e)]
    if not pairs:
        return ""
    pairs.sort(key=lambda p: p[1])
    names, vals = [p[0] for p in pairs], [p[1] for p in pairs]
    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(names) * 0.6), 4))
    colors = ["C2" if n == best else "C0" for n in names]
    ax.bar(range(len(names)), vals, color=colors, edgecolor="gray")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("‖ξ − ξ_true‖_F / ‖ξ_true‖_F")
    ax.set_title("ξ distance from true (lower = equations recovered, not just curve-fitting)")
    ax.axhline(y=0, color="k", linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_xi_rel_err_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_aic_bic_bars(suite: Dict, output_dir: str = "outputs/consensus/") -> str:
    """
    Bar charts: AIC and BIC per method (lower = better parsimony vs fit; no ground truth needed).
    """
    _ensure_dir(output_dir)
    val = suite.get("validation", {})
    aic_list = val.get("aic_by_method", [])
    bic_list = val.get("bic_by_method", [])
    method_names = val.get("method_names", [])
    if not aic_list or len(aic_list) != len(method_names):
        return ""
    if not bic_list or len(bic_list) != len(method_names):
        bic_list = aic_list  # fallback
    aic_vals = [float(a) if np.isfinite(a) else np.nan for a in aic_list]
    bic_vals = [float(b) if np.isfinite(b) else np.nan for b in bic_list]
    best_aic = val.get("best_by_aic")
    best_bic = val.get("best_by_bic")
    n = len(method_names)
    fig, (ax_aic, ax_bic) = plt.subplots(2, 1, figsize=(max(6, n * 0.6), 5), sharex=True)
    x = np.arange(n)
    colors_aic = ["C2" if method_names[i] == best_aic else "C0" for i in range(n)]
    colors_bic = ["C2" if method_names[i] == best_bic else "C0" for i in range(n)]
    ax_aic.bar(x, aic_vals, color=colors_aic, edgecolor="gray")
    ax_aic.set_ylabel("AIC")
    ax_aic.set_title("AIC (lower = better; no truth needed)")
    ax_aic.tick_params(axis="x", labelbottom=False)
    ax_bic.bar(x, bic_vals, color=colors_bic, edgecolor="gray")
    ax_bic.set_ylabel("BIC")
    ax_bic.set_title("BIC (lower = better; no truth needed)")
    ax_bic.set_xticks(x)
    ax_bic.set_xticklabels(method_names, rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_aic_bic_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_xi_agreement_heatmap(suite: Dict, output_dir: str = "outputs/consensus/") -> str:
    """
    Heat map of equation agreement (ξ vs ξ_true): rows = methods, columns = states.
    Cell = agreement score (1 - relative squared error). The single best method (best_by_xi_err,
    i.e. lowest ‖ξ − ξ_true‖_F) is highlighted so the heatmap matches the time series / report.
    Only used when xi_true is provided (e.g. benchmark systems like Lorenz).
    """
    _ensure_dir(output_dir)
    val = suite.get("validation", {})
    xi_matrix = val.get("xi_agreement_matrix")
    method_names = val.get("method_names", [])
    target_names = val.get("target_names", [])
    best_for_display = val.get("best_for_display") or val.get("best_by_xi_err")

    if xi_matrix is None or xi_matrix.size == 0 or len(method_names) == 0 or len(target_names) == 0:
        return ""

    xi_matrix = np.asarray(xi_matrix, dtype=float)
    n_methods, n_states = xi_matrix.shape
    xi_display = np.clip(xi_matrix, -0.05, 1.0)
    vmin = float(np.nanmin(xi_display)) if np.any(np.isfinite(xi_display)) else 0.0
    vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(max(6, n_states * 1.2), max(4, n_methods * 0.5)))
    im = ax.imshow(xi_display, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels([f"$\\dot{{{n}}}$" for n in target_names], rotation=45, ha="right")
    ax.set_yticks(np.arange(n_methods))
    ax.set_yticklabels(method_names, fontsize=9)
    criterion = val.get("best_for_display_criterion", "by ξ if truth, else AIC")
    ax.set_title(f"Equation agreement (ξ vs ξ_true) — best row: {best_for_display or '—'} ({criterion})")

    for j in range(n_states):
        for i in range(n_methods):
            v = xi_matrix[i, j]
            text = f"{v:.2f}" if np.isfinite(v) else "—"
            is_best_row = best_for_display and i < len(method_names) and method_names[i] == best_for_display
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    fontweight="bold" if is_best_row else "normal", color="black")

    plt.colorbar(im, ax=ax, label="ξ agreement", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_best_method_by_xi_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _build_term_stability_matrix(
    results_by_mode: Dict[str, Dict],
    zero_tol: float = 1e-10,
) -> Tuple[list, list, np.ndarray, np.ndarray]:
    """
    Build term × method coefficient matrix from consensus results in physical units.
    Returns (term_labels, method_names, coef_matrix, n_methods_per_term).
    term_labels[i] = "fname → tname"; rows sorted by stability (most methods agreeing first).
    """
    method_names = list(results_by_mode.keys())
    if not method_names:
        return [], [], np.array([]), np.array([])

    # Collect all (fname, tname) that appear in any method
    term_set = set()
    for name, fit in results_by_mode.items():
        kept = fit.get("kept_names") or fit.get("feature_names") or []
        coef_src = fit.get("coef_phys", fit.get("coef_kept", fit.get("coef", np.zeros((0, 0)))))
        targets = fit.get("target_names") or [f"target_{j}" for j in range(np.asarray(coef_src).shape[1] if np.asarray(coef_src).ndim == 2 else 0)]
        coef = np.asarray(coef_src, dtype=float)
        if coef.ndim != 2 or len(kept) != coef.shape[0]:
            continue
        for k in range(coef.shape[0]):
            for j in range(min(coef.shape[1], len(targets))):
                term_set.add((kept[k], targets[j]))

    if not term_set:
        return [], method_names, np.zeros((0, len(method_names))), np.array([])

    # Per-method lookup: (fname, tname) -> coefficient
    def get_coef(method: str, fname: str, tname: str) -> float:
        fit = results_by_mode.get(method, {})
        kept = fit.get("kept_names") or fit.get("feature_names") or []
        targets = fit.get("target_names") or []
        coef = np.asarray(fit.get("coef_phys", fit.get("coef_kept", fit.get("coef"))), dtype=float)
        if coef.size == 0 or fname not in kept:
            return 0.0
        k = kept.index(fname) if fname in kept else -1
        if k < 0:
            return 0.0
        j = next((i for i, t in enumerate(targets) if t == tname), -1)
        if j < 0 or j >= coef.shape[1]:
            return 0.0
        return float(coef[k, j])

    # Build matrix and count presence per term
    term_list = sorted(term_set)
    n_terms = len(term_list)
    M = len(method_names)
    matrix = np.zeros((n_terms, M))
    n_per_term = np.zeros(n_terms, dtype=int)
    for i, (fname, tname) in enumerate(term_list):
        for j, method in enumerate(method_names):
            c = get_coef(method, fname, tname)
            matrix[i, j] = c
            if abs(c) > zero_tol:
                n_per_term[i] += 1

    # Sort by stability (most methods agreeing first), then by fname, then tname
    term_labels = [f"{f} → {t}" for f, t in term_list]
    order = np.lexsort(([term_list[i][1] for i in range(n_terms)],
                        [term_list[i][0] for i in range(n_terms)],
                        -n_per_term))
    matrix = matrix[order, :]
    n_per_term = n_per_term[order]
    term_labels = [term_labels[o] for o in order]

    return term_labels, method_names, matrix, n_per_term


def plot_term_stability_heatmap(
    suite: Dict,
    output_dir: str = "outputs/consensus/",
    zero_tol: float = 1e-10,
    max_terms: int = 80,
    figsize: Optional[Tuple[float, float]] = None,
) -> str:
    """
    Term stability heatmap: rows = terms, columns = methods. Cell = physical coefficient.
    Rows are sorted by stability (terms present in more methods first). No method
    is favored—all columns treated equally. Left annotation shows "N/M" (number
    of methods where term is non-zero) so stability is explicit and unbiased.

    Use this to see which terms are stable across configs (equation plausibility)
    vs method-specific (possible curve fitting or preprocessing-dependent).
    """
    _ensure_dir(output_dir)
    modes = suite.get("modes") or suite.get("results_by_mode") or {}
    if not modes:
        return ""

    term_labels, method_names, matrix, n_per_term = _build_term_stability_matrix(
        modes, zero_tol=zero_tol
    )
    if not term_labels or matrix.size == 0:
        return ""

    M = len(method_names)
    n_terms = len(term_labels)
    if n_terms > max_terms:
        # Keep top max_terms by stability, then by order
        term_labels = term_labels[:max_terms]
        matrix = matrix[:max_terms, :]
        n_per_term = n_per_term[:max_terms]
        n_terms = max_terms

    # Diverging colormap: zero = neutral (white), positive/negative distinct
    abs_max = float(np.nanmax(np.abs(matrix)) + 1e-12)
    vmin, vmax = -abs_max, abs_max
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # 0 at center for RdBu_r

    if figsize is None:
        figsize = (max(8, M * 0.5), max(5, n_terms * 0.28))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(M))
    ax.set_xticklabels(method_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_terms))
    # Row label: term and stability count "fname → tname  (N/M)"
    row_labels = [f"{tl}  ({n_per_term[i]}/{M})" for i, tl in enumerate(term_labels)]
    ax.set_yticklabels(row_labels, fontsize=7)

    ax.set_xlabel("Method (columns = configs)")
    ax.set_ylabel("Term (rows sorted by stability; N/M = methods with term non-zero)")
    ax.set_title("Term stability: physical coefficient by method (stable terms at top; 0 = absent)")

    plt.colorbar(im, ax=ax, label="Physical coefficient", shrink=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_term_stability_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_posthoc_equation_heatmap(suite: Dict, output_dir: str = "outputs/consensus/") -> Dict[str, str]:
    """
    Post-hoc: plot coefficient (Xi) TRUE vs DISCOVERED vs ERROR heatmap for the discovery-
    recommended method (AIC/BIC, never truth). Truth is for post-hoc comparison only; we do not
    use it to pick which config to highlight. Use after run_targeted_consensus_suite so we see
    which terms/equations are wrong for the AIC-best config.
    Returns paths to saved figures; empty dict if no truth or best method.
    """
    _ensure_dir(output_dir)
    val = suite.get("validation", {})
    full_results = suite.get("full_results", {})
    # Discovery recommendation (AIC/BIC), not ξ — truth is for post-hoc only
    best_name = val.get("best_for_discovery") or val.get("best_by_aic") or val.get("best_overall")
    if not best_name or best_name not in full_results:
        return {}
    res = full_results[best_name]
    model = res.get("model")
    if not getattr(model, "rhs_symbolic", None):
        return {}
    fit = res["fit"]
    try:
        kept_names_union, xi_true_union, xi_disc_union = _build_union_xi_matrices_for_fit(res)
    except Exception:
        return {}

    paths = plot_xi_heatmaps(
        kept_names=kept_names_union,
        Xi_true=xi_true_union,
        Xi_disc=xi_disc_union,
        target_names=fit["target_names"],
        output_dir=output_dir,
        prefix="posthoc_equation_error",
        drop_constant_like=True,
    )
    return paths


def print_posthoc_table(suite: Dict) -> None:
    """
    Print full post-hoc table: for each method, R², AIC, BIC, n_terms (complexity),
    stability (term agreement across configs), n_harmful (terms that hurt fit), blind_score,
    and ‖ξ−ξ_true‖/‖ξ_true‖ (post-hoc only). Discovery recommendation uses AIC + stability + harmful.
    """
    val = suite.get("validation", {})
    method_names = val.get("method_names", [])
    r2_matrix = val.get("r2_matrix")
    n = len(method_names)
    if not method_names:
        print("No methods in suite.")
        return

    r2_matrix = np.asarray(r2_matrix, dtype=float) if r2_matrix is not None else None
    r2_means = np.nanmean(r2_matrix, axis=1) if r2_matrix is not None and r2_matrix.shape[0] == n else np.full(n, np.nan)
    aic = val.get("aic_by_method", [np.nan] * n)
    bic = val.get("bic_by_method", [np.nan] * n)
    n_terms = val.get("n_terms_by_method", [0] * n)
    stability = val.get("stability_by_method", [np.nan] * n)
    n_harmful = val.get("n_harmful_by_method", [0] * n)
    blind_score = val.get("blind_score_by_method", [np.nan] * n)
    xi_rel_err = val.get("xi_rel_err_by_method", [])
    xi_frob_err = val.get("xi_frob_err_by_method", [])
    xi_frob_true = val.get("xi_frob_true_by_method", [])
    has_xi = xi_rel_err and len(xi_rel_err) == n
    e_long = val.get("e_long_by_method", [])
    surrogate_j = val.get("surrogate_j_by_method", [])
    has_surrogate = bool(surrogate_j and len(surrogate_j) == n)
    if len(e_long) != n:
        e_long = list(e_long) + [np.nan] * max(0, n - len(e_long))
    if len(surrogate_j) != n:
        surrogate_j = list(surrogate_j) + [np.nan] * max(0, n - len(surrogate_j))
    if len(aic) != n:
        aic = list(aic) + [np.nan] * (n - len(aic))
    if len(bic) != n:
        bic = list(bic) + [np.nan] * (n - len(bic))
    if len(n_terms) != n:
        n_terms = list(n_terms) + [0] * (n - len(n_terms))
    if len(stability) != n:
        stability = list(stability) + [np.nan] * (n - len(stability))
    if len(n_harmful) != n:
        n_harmful = list(n_harmful) + [0] * (n - len(n_harmful))
    if len(blind_score) != n:
        blind_score = list(blind_score) + [np.nan] * (n - len(blind_score))

    print("\n" + "=" * 100)
    print("  POST-HOC: all criteria per method (discovery = AIC + stability + low n_harmful; ‖ξ−ξ_true‖ = post-hoc only)")
    print("=" * 100)
    # Header: Method | R² | AIC | BIC | n_terms | stability | n_harmful | blind | Xi metrics
    header = "  %-38s  %7s  %10s  %10s  %6s  %6s  %4s  %7s" % ("Method", "R²", "AIC", "BIC", "n_term", "stab", "harm", "blind")
    if has_xi:
        header += "  %8s  %8s" % ("‖ξ−ξ‖", "‖ξ−ξ‖F")
    print(header)
    print("-" * 100)
    for i in range(n):
        name = method_names[i] if i < len(method_names) else "—"
        r2 = r2_means[i] if i < len(r2_means) else np.nan
        ai = aic[i] if i < len(aic) else np.nan
        bi = bic[i] if i < len(bic) else np.nan
        nt = n_terms[i] if i < len(n_terms) else 0
        st = stability[i] if i < len(stability) else np.nan
        nh = n_harmful[i] if i < len(n_harmful) else 0
        bl = blind_score[i] if i < len(blind_score) else np.nan
        xi = xi_rel_err[i] if has_xi and i < len(xi_rel_err) else np.nan
        xi_f = xi_frob_err[i] if has_xi and i < len(xi_frob_err) else np.nan
        el = e_long[i] if i < len(e_long) else np.nan
        jv = surrogate_j[i] if i < len(surrogate_j) else np.nan
        r2_s = f"{r2:.3f}" if np.isfinite(r2) else "—"
        ai_s = f"{ai:.1f}" if np.isfinite(ai) else "—"
        bi_s = f"{bi:.1f}" if np.isfinite(bi) else "—"
        st_s = f"{st:.2f}" if np.isfinite(st) else "—"
        bl_s = f"{bl:.3f}" if np.isfinite(bl) else "—"
        el_s = f"{el:.4g}" if np.isfinite(el) else "—"
        j_s = f"{jv:.3f}" if np.isfinite(jv) else "—"
        xi_s = f"{xi:.4f}" if np.isfinite(xi) else "—"
        xi_f_s = f"{xi_f:.3g}" if np.isfinite(xi_f) else "—"
        line = f"  {name:<38}  {r2_s:>7}  {ai_s:>10}  {bi_s:>10}  {nt:>6}  {st_s:>6}  {nh:>4}  {bl_s:>7}"
        if has_surrogate:
            line += f"  {el_s:>8}  {j_s:>6}"
        if has_xi:
            line += f"  {xi_s:>8}  {xi_f_s:>8}"
        print(line)
    print("-" * 100)
    best_discovery = val.get("best_for_discovery") or val.get("best_by_aic")
    best_xi = val.get("best_by_xi_err")
    best_j = val.get("best_by_surrogate_j")
    print(f"  Discovery (no truth): {best_discovery or '—'}  (prefer: low AIC, high stability, low n_harmful, high blind_score)")
    if best_j:
        print(f"  Surrogate J (4-test): best = {best_j}  (lower J ≈ better equation match when truth unknown)")
    if best_xi and has_xi:
        print(f"  Post-hoc only:        best ‖ξ−ξ_true‖ = {best_xi}")
        if xi_frob_true and len(xi_frob_true) == n:
            den = float(np.nanmedian(np.asarray(xi_frob_true, dtype=float)))
            if np.isfinite(den):
                print(f"  Xi Frobenius denominator (union basis) ≈ {den:.3g}")
    overfit = val.get("overfit_warning")
    if overfit:
        print(f"\n  ⚠ {overfit}")
    leg = "  Columns: R²=curve fit; AIC/BIC=parsimony (lower better); n_term=complexity; stab=term stability; harm=terms that hurt; blind=adjusted R² + importance − harmful"
    if has_surrogate:
        leg += "; E_long=rollout RMSE; J=surrogate (E_short+E_long+C+S)"
    leg += "; ‖ξ−ξ‖=post-hoc."
    print(leg)
    print("=" * 100 + "\n")


def plot_consensus_comparison_with_best(suite: Dict, output_dir: str = "outputs/consensus/") -> str:
    """
    Time series: Truth vs SINDy for all methods, all in physical derivative units.
    Each method's prediction is Theta(Z) @ coef_kept = Y_phys (denormalization done in fit).
    Highlighted method (thick line) and title use best_for_display (ξ or AIC).
    """
    _ensure_dir(output_dir)
    anchor = suite.get("anchor_meta") or suite.get("anchor")
    if anchor is None:
        raise ValueError("suite must contain 'anchor_meta'.")
    modes = suite.get("modes", {})
    if not modes:
        raise ValueError("suite must contain 'modes'.")
    validation = suite.get("validation", {})
    best_per_state = validation.get("best_per_state", [])
    best_for_display = validation.get("best_for_display") or validation.get("best_overall", "")
    target_names = list(list(modes.values())[0]["target_names"])
    n_targets = len(target_names)

    t = np.asarray(anchor["t"]).ravel()
    Z = np.asarray(anchor["Z"])
    model = anchor["model"]

    if Z.shape[1] != len(model.measured_names):
        raise ValueError("Z columns != model.measured_names.")
    symbol_to_idx = {name: i for i, name in enumerate(model.measured_names)}
    ordered_indices = [symbol_to_idx[s.name] for s in model.all_symbols]
    Z_ordered = Z[:, ordered_indices]

    F_true = np.zeros((len(t), n_targets))
    for i, f_num in enumerate(model.rhs_lambdified):
        F_true[:, i] = np.asarray(f_num(*Z_ordered.T), dtype=float).ravel()

    method_list = list(modes.keys())
    fig, axes = plt.subplots(n_targets, 1, figsize=(10, 2.8 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]

    for i in range(n_targets):
        ax = axes[i]
        ax.plot(t, F_true[:, i], "k-", alpha=0.7, lw=2, label="Ground Truth")
        for idx, (name, fit) in enumerate(modes.items()):
            lib = fit["library"]
            aff = affine_from_sklearn_scaler(fit["scaler"])
            if fit.get("use_physical_library"):
                Theta_full = lib.transform(Z_scaled=Z, Z_phys=Z)
            else:
                Theta_full = lib.transform(Z_scaled=aff.transform(Z), Z_phys=Z)
            F_sindy = Theta_full[:, np.asarray(fit["kept_idx"])] @ np.asarray(fit["coef_kept"])
            c = plt.cm.tab20(idx % 20) if method_list else "gray"
            is_best = name == best_for_display
            ax.plot(t, F_sindy[:, i], "--" if not is_best else "-", color=c,
                    lw=2.5 if is_best else 1.2, label=f"{name}" + (" (best)" if is_best else ""))
        ax.set_ylabel(f"$\\dot{{{target_names[i]}}}$")
        ax.set_title(f"Highlighted: {best_for_display}", fontsize=9, loc="right")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    criterion = validation.get("best_for_display_criterion", "by ξ if truth, else AIC")
    plt.suptitle(f"Consensus: Truth vs SINDy (best: {best_for_display} — {criterion})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_comparison_with_best.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path