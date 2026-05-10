# sindy/pareto.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt


def pareto_curve_xy(pareto_list: List[Dict[str, Any]], var_y: float, eps: float = 1e-12):
    """
    Convert pareto dicts -> arrays for plotting/scoring.
    x = complexity (nonzeros)
    y = NMSE = mse / var(y)
    """
    if pareto_list is None or len(pareto_list) == 0:
        raise ValueError("pareto_list is empty")

    if not np.isfinite(var_y) or float(var_y) < 0:
        raise ValueError(f"var_y must be finite and nonnegative, got {var_y}")

    P = sorted(pareto_list, key=lambda r: (r["complexity"], r["mse"]))
    x = np.array([r["complexity"] for r in P], dtype=float)
    y = np.array([r["mse"] for r in P], dtype=float) / (float(var_y) + eps)
    return P, x, y


def knee_by_curvature(
    pareto_list: List[Dict[str, Any]],
    var_y: float,
    eps: float = 1e-12,
    use_log: bool = True,
):
    """
    Knee by max distance from the line between endpoints in (x, y) space.
    Uses y = log(NMSE) by default (more stable when errors span decades).
    Returns a pareto dict.
    """
    P, x, nmse = pareto_curve_xy(pareto_list, var_y, eps=eps)

    if len(P) == 1:
        return P[0]

    y = np.log(nmse + eps) if use_log else nmse.copy()

    # Normalize x and y to [0,1]
    x_span = float(x.max() - x.min())
    y_span = float(y.max() - y.min())
    xn = (x - x.min()) / (x_span + eps)
    yn = (y - y.min()) / (y_span + eps)

    a = np.array([xn[0], yn[0]], dtype=float)
    b = np.array([xn[-1], yn[-1]], dtype=float)
    ab = b - a
    ab2 = float(np.dot(ab, ab)) + eps

    # Distance from point to line a->b
    d = np.empty(len(P), dtype=float)
    for i in range(len(P)):
        p = np.array([xn[i], yn[i]], dtype=float)
        proj = a + ab * (np.dot(p - a, ab) / ab2)
        d[i] = float(np.linalg.norm(p - proj))

    return P[int(np.argmax(d))]


def pick_with_sparsity_knob(
    pareto_list: List[Dict[str, Any]],
    var_y: float,
    lam: float = 0.02,
    eps: float = 1e-12,
):
    """
    A smooth knob: minimize log(NMSE) + lam * normalized_complexity.
    - lam = 0 -> accuracy only (densest)
    - bigger lam -> sparser
    """
    P, x, nmse = pareto_curve_xy(pareto_list, var_y, eps=eps)

    if len(P) == 1:
        return P[0]

    # Normalize complexity to [0,1] so lam has consistent meaning
    xn = (x - x.min()) / (float(x.max() - x.min()) + eps)
    score = np.log(nmse + eps) + float(lam) * xn

    return P[int(np.argmin(score))]


# Default max lambda for dial=0 (full sparsity); dial=1 uses lam=0 (full error reduction)
_PARETO_DIAL_LAM_MAX = 0.2


def bic_signal_variance(Y: np.ndarray, *, equal_weight_per_target: bool) -> float:
    """
    Variance of the regression targets for scaling the BIC MSE floor (matches ``var_y`` in
    :func:`sindy.fit.fit_sindy` when the same ``equal_weight_per_target`` flag is used).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if equal_weight_per_target and Y.shape[1] > 1:
        return float(np.mean(np.var(Y, axis=0)))
    return float(np.var(Y))


def bic_for_sindy_sweep_candidate(
    mse: float,
    k_nonzero: int,
    n_obs: float,
    *,
    mse_floor: float,
) -> float:
    """
    Multivariate SINDy BIC aligned with :func:`sindy.pipeline.fit_sindy_main` post-fit BIC.

    ``mse`` is the mean squared residual over all samples and targets (same as ``RSS / n_obs`` for
    ``n_obs = n_samples * n_targets``). We use

    ``BIC = n_obs * ln(MSE_eff) + k * ln(n_obs)``,

    with ``MSE_eff = max(mse, mse_floor)`` when ``mse_floor > 0``. The floor should be set from
    signal scale (e.g. ``max(fraction * var(Y), epsilon)``) so small-scale targets (e.g. ``C_m``)
    are not treated as pure noise while large-scale forces still ignore floating-point dust.

    ``k`` is the total number of nonzero coefficients in Xi (all equations).
    """
    n_obs = float(max(n_obs, 1.0))
    mse_eff = float(mse)
    if mse_floor > 0.0:
        mse_eff = max(mse_eff, float(mse_floor))
    if mse_eff <= 0.0:
        mse_eff = 1e-300
    return n_obs * float(np.log(mse_eff)) + float(k_nonzero) * float(np.log(n_obs))


def pick_by_bic(
    sweep_results: List[Dict[str, Any]],
    n_samples: int,
    n_targets: int,
    Y: np.ndarray,
    *,
    equal_weight_per_target: bool = False,
    variance_fraction: float = 1e-3,
    mse_floor_epsilon: float = 1e-12,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Pick one model from a full STLSQ threshold sweep by minimum BIC.

    Uses every candidate in ``sweep_results`` (not only the Pareto frontier), so a dominated
    point can still win if it balances RSS and complexity better under BIC.

    MSE floor: ``max(variance_fraction * signal_variance(Y), mse_floor_epsilon)`` with the same
    variance definition as ``fit_sindy`` / Pareto MSE (mean of per-target variances when
    ``equal_weight_per_target`` and multiple columns). Pass bootstrap ``Y`` in ensemble mode.
    """
    if not sweep_results:
        raise ValueError("sweep_results is empty")
    n_obs = float(max(int(n_samples) * int(n_targets), 1))
    sig_var = bic_signal_variance(Y, equal_weight_per_target=equal_weight_per_target)
    if variance_fraction > 0.0:
        mse_floor = max(float(variance_fraction) * float(sig_var), float(mse_floor_epsilon))
    else:
        mse_floor = 0.0
    scores = np.array(
        [
            bic_for_sindy_sweep_candidate(
                float(r["mse"]),
                int(r["complexity"]),
                n_obs,
                mse_floor=mse_floor,
            )
            for r in sweep_results
        ],
        dtype=float,
    )
    i = int(np.argmin(scores))
    return sweep_results[i], scores


def pick_by_dial(
    pareto_list: List[Dict[str, Any]],
    var_y: float,
    dial: float,
    lam_max: float = _PARETO_DIAL_LAM_MAX,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Select one Pareto point by a continuous dial in [0, 1].
    - dial = 0: prioritize sparsity (minimize complexity; sparsest reasonable model).
    - dial = 1: prioritize error reduction (minimize NMSE; densest model).
    - dial in (0, 1): smooth blend via pick_with_sparsity_knob with lam = (1 - dial) * lam_max.
    """
    dial = max(0.0, min(1.0, float(dial)))
    lam = (1.0 - dial) * float(lam_max)
    return pick_with_sparsity_knob(pareto_list, var_y, lam=lam, eps=eps)


def plot_pareto_frontier(
    pareto_list: List[Dict[str, Any]],
    var_y: float,
    pick: Optional[Dict[str, Any]] = None,
    picks: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "",
    ax=None,
):
    """
    Plot complexity vs NMSE (log y-scale). Optionally annotate selected pick(s).
    - pick: single dict
    - picks: dict{name: dict}
    """
    P, x, nmse = pareto_curve_xy(pareto_list, var_y)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    ax.plot(x, nmse, "o-", ms=3, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Complexity (# nonzeros)")
    ax.set_ylabel("NMSE (MSE / Var(y))")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    eps = 1e-12

    def mark(r: Dict[str, Any], label: str, color: str):
        nmse_pt = float(r["mse"]) / (float(var_y) + eps)
        ax.plot([float(r["complexity"])], [nmse_pt], "o", ms=8, color=color)
        ax.annotate(label, (float(r["complexity"]), nmse_pt), textcoords="offset points", xytext=(6, 6))

    if pick is not None:
        mark(pick, "selected", "C3")

    if picks is not None:
        colors = ["C1", "C2", "C4", "C5", "C6", "C7"]
        for i, (name, r) in enumerate(picks.items()):
            mark(r, str(name), colors[i % len(colors)])

    return ax


def plot_bic_selection(
    fit: Dict[str, Any],
    *,
    title: str = "",
    out_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Two-panel figure showing how BIC selects the model from the STLSQ threshold sweep.

    Top panel  — NMSE vs complexity (log y): all sweep points (grey), Pareto-optimal
                 frontier (blue line), and BIC-selected model (red star).
    Bottom panel — BIC score vs complexity: all sweep points (grey dots), BIC curve
                   connecting unique complexities (blue), and selected minimum (red star).

    Uses ``fit["pareto"]`` (full sweep, 50 dicts with ``mse``, ``complexity``, ``threshold``),
    ``fit["bic_scores"]`` (parallel array), ``fit["var_y"]``, ``fit["best_complexity"]``.
    """
    pareto_list = fit.get("pareto") or []
    var_y       = float(fit.get("var_y", 1.0))
    best_c      = int(fit.get("best_complexity", 0))
    eps         = 1e-12

    if not pareto_list:
        raise ValueError("fit dict missing 'pareto' sweep data")

    n_obs = float(fit.get("n_obs") or max(1, len(pareto_list)))

    raw_bic = fit.get("bic_scores")
    bic_posthoc = raw_bic is None or (hasattr(raw_bic, "__len__") and len(raw_bic) == 0)
    if bic_posthoc:
        # Compute BIC post-hoc: used when dial-based selection was chosen instead of BIC
        bic_scores = np.array(
            [bic_for_sindy_sweep_candidate(float(r["mse"]), int(r["complexity"]), n_obs, mse_floor=0.0)
             for r in pareto_list],
            dtype=float,
        )
    else:
        bic_scores = np.asarray(raw_bic, dtype=float)

    # Sort full sweep by complexity then mse for consistent plotting
    order = sorted(range(len(pareto_list)), key=lambda i: (pareto_list[i]["complexity"], pareto_list[i]["mse"]))
    sweep_c    = np.array([pareto_list[i]["complexity"] for i in order], dtype=float)
    sweep_nmse = np.array([pareto_list[i]["mse"]        for i in order], dtype=float) / (var_y + eps)
    sweep_bic  = bic_scores[order]

    # Pareto-optimal front: for each unique complexity, keep the lowest NMSE point
    unique_c  = np.unique(sweep_c)
    front_c, front_nmse, front_bic = [], [], []
    for uc in unique_c:
        mask = sweep_c == uc
        best_idx = np.argmin(sweep_nmse[mask])
        front_c.append(uc)
        front_nmse.append(sweep_nmse[mask][best_idx])
        front_bic.append(sweep_bic[mask][np.argmin(sweep_bic[mask])])
    front_c    = np.array(front_c)
    front_nmse = np.array(front_nmse)
    front_bic  = np.array(front_bic)

    # Selected point indices
    bic_min_idx   = int(np.argmin(sweep_bic))
    sel_c         = sweep_c[bic_min_idx]
    sel_nmse      = sweep_nmse[bic_min_idx]
    sel_bic       = sweep_bic[bic_min_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=False)

    # ── Top: NMSE vs complexity ──────────────────────────────────────────────
    ax1.scatter(sweep_c, sweep_nmse, s=18, color="grey", alpha=0.45, zorder=2, label="All sweep points")
    ax1.plot(front_c, front_nmse, "o-", ms=5, lw=1.4, color="#1f77b4", zorder=3, label="Pareto front")
    ax1.plot(sel_c, sel_nmse, "*", ms=14, color="#d62728", zorder=5, label=f"BIC selected (k={int(sel_c)})")
    ax1.axvline(sel_c, color="#d62728", lw=0.8, ls="--", alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("NMSE  (MSE / Var(y))", fontsize=9)
    ax1.set_title(title or "BIC model selection", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.25)

    # ── Bottom: BIC vs complexity ────────────────────────────────────────────
    bic_note = " (post-hoc)" if bic_posthoc else ""
    ax2.scatter(sweep_c, sweep_bic, s=18, color="grey", alpha=0.45, zorder=2, label="All sweep points")
    ax2.plot(front_c, front_bic, "o-", ms=5, lw=1.4, color="#1f77b4", zorder=3, label=f"Min BIC per complexity{bic_note}")
    ax2.plot(sel_c, sel_bic, "*", ms=14, color="#d62728", zorder=5, label=f"Min BIC  (k={int(sel_c)})")
    ax2.axvline(sel_c, color="#d62728", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("Complexity  (# nonzero coefficients)", fontsize=9)
    ax2.set_ylabel("BIC score", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.25)

    formula = r"$\mathrm{BIC} = n \ln(\mathrm{MSE}) + k \ln(n)$"
    ax2.text(0.02, 0.04, formula, transform=ax2.transAxes, fontsize=8, color="#555555")

    fig.tight_layout(h_pad=2.5)
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig, (ax1, ax2)

