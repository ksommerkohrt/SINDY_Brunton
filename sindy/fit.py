# sindy/fit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sindy.pareto import knee_by_curvature, pick_with_sparsity_knob, pick_by_dial, pick_by_bic


@dataclass
class SINDyFitConfig:
    mode: str = "pareto"
    thresholds: np.ndarray | None = None
    ensemble_B: int = 100
    ensemble_frac: float = 0.8
    map_lam: float = 1e-3
    map_sigma2: float = 1.0
    sparsity_bias: float = 0.0
    # Ridge (L2) penalty for least-squares. Larger values stabilize regression when columns are
    # collinear (e.g. Lorenz x, z, x*z). Use prefer_parsimony=False + alpha_ridge=1e-3..1e-2 to
    # keep all terms and still converge.
    alpha_ridge: float = 1e-6
    # "bic" = minimum BIC over full threshold sweep; "knee" = Pareto knee; "last" = densest Pareto point
    pareto_pick: str = "bic"
    # Continuous dial [0, 1]: 0 = prioritize sparsity, 1 = prioritize error reduction. If set, overrides pareto_pick.
    pareto_dial: Optional[float] = None
    # Best practice for multi-equation systems: if True, MSE and var_y are per-target then averaged,
    # so each equation contributes equally regardless of scale (avoids high-variance eqns dominating).
    equal_weight_per_target: bool = True
    # BIC MSE floor = max(bic_mse_variance_fraction * signal_variance(Y), bic_mse_floor_epsilon).
    # Matches target scale (small C_m vs large NF). Set variance_fraction to 0.0 to disable floor.
    bic_mse_variance_fraction: float = 1e-3
    bic_mse_floor_epsilon: float = 1e-12


def fit_sindy(
    Theta: np.ndarray,
    Y: np.ndarray,
    cfg: SINDyFitConfig,
    plot: bool = False,           
    plot_path: str = "pareto.png"  
) -> dict:
    """
    Generalized SINDy fitting engine.
    """
    Theta = np.asarray(Theta, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    N, K = Theta.shape

    equal_weight = getattr(cfg, "equal_weight_per_target", False)
    if equal_weight and Y.shape[1] > 1:
        var_y = float(np.mean(np.var(Y, axis=0))) + 1e-12
    else:
        var_y = np.var(Y) + 1e-12

    alpha_ridge = getattr(cfg, "alpha_ridge", 1e-6)

    if cfg.thresholds is None:
        K = Theta.shape[1]
        if K > 0:
            I = np.eye(K, dtype=float)
            try:
                Xi_ls = np.linalg.solve(Theta.T @ Theta + alpha_ridge * I, Theta.T @ Y)
            except np.linalg.LinAlgError:
                Xi_ls = np.linalg.lstsq(Theta.T @ Theta + alpha_ridge * I, Theta.T @ Y, rcond=None)[0]
            max_coef = float(np.max(np.abs(Xi_ls))) if Xi_ls.size else 1.0
        else:
            max_coef = 1.0
        cfg.thresholds = np.logspace(-6, 2, 50) * max_coef

    if cfg.mode == "pareto":
        stlsq = AdaptiveSTLSQ(alpha_ridge=alpha_ridge)
        results, pareto = stlsq.pareto_analysis(
            Theta, Y, cfg.thresholds,
            equal_weight_per_target=equal_weight,
        )
        # Dial takes precedence: 0 = sparsity, 1 = error reduction
        dial = getattr(cfg, "pareto_dial", None)
        pick = (getattr(cfg, "pareto_pick", "bic") or "bic").lower()
        bic_scores: np.ndarray | None = None
        if dial is not None and 0 <= dial <= 1:
            best = pick_by_dial(pareto, var_y, dial=dial)
        elif pick in ("bic", "per_target_bic"):
            best, bic_scores = pick_by_bic(
                results,
                N,
                Y.shape[1],
                Y,
                equal_weight_per_target=equal_weight,
                variance_fraction=float(getattr(cfg, "bic_mse_variance_fraction", 1e-2)),
                mse_floor_epsilon=float(getattr(cfg, "bic_mse_floor_epsilon", 1e-12)),
            )
        elif pick == "last":
            best = pareto[-1]
        elif cfg.sparsity_bias == 0:
            best = knee_by_curvature(pareto, var_y, use_log=True)
        else:
            best = pick_with_sparsity_knob(pareto, var_y, lam=cfg.sparsity_bias)
        out: Dict[str, Any] = {
            "coef": best["coef"],
            "best_mse": best["mse"],
            "best_complexity": best["complexity"],
            "best_threshold": best["threshold"],
            "pareto": pareto,
            "var_y": var_y,
            "mode": "pareto",
        }
        if bic_scores is not None:
            out["bic_scores"] = bic_scores
            out["best_bic"] = float(np.min(bic_scores))
        return out

    if cfg.mode == "pareto_ensemble":
        return ensemble_pareto_analysis(
            Theta, Y, cfg.thresholds,
            B=cfg.ensemble_B,
            frac=cfg.ensemble_frac,
            sparsity_bias=cfg.sparsity_bias,
            pareto_pick=getattr(cfg, "pareto_pick", "knee"),
            pareto_dial=getattr(cfg, "pareto_dial", None),
            equal_weight_per_target=equal_weight,
            alpha_ridge=alpha_ridge,
            bic_mse_variance_fraction=float(getattr(cfg, "bic_mse_variance_fraction", 1e-2)),
            bic_mse_floor_epsilon=float(getattr(cfg, "bic_mse_floor_epsilon", 1e-12)),
        )

    if cfg.mode == "bayes_map":
        reg = MAPSINDyRegressor(lam=cfg.map_lam, sigma2=cfg.map_sigma2)
        reg.fit(Theta, Y)
        return {
            "coef": reg.coef_,
            "mode": "bayes_map",
        }

    if cfg.mode == "bayes_map_ensemble":
        return ensemble_map_analysis(
            Theta, Y,
            lam=cfg.map_lam,
            sigma2=cfg.map_sigma2,
            B=cfg.ensemble_B,
            frac=cfg.ensemble_frac,
        )

    raise ValueError(f"Unknown SINDy mode {cfg.mode}")

    
def drop_constant_like_columns(
    Theta: np.ndarray,
    feature_names: List[str],
    sp_features: List[Any],
    cv_thresh: float = 1e-3,
    constant_name: str = "1",
) -> Tuple[np.ndarray, List[int], List[str], List[Any]]:
    Theta = np.asarray(Theta, float)
    if Theta.ndim != 2:
        raise ValueError(f"Theta must be 2D, got {Theta.shape}")
    if Theta.shape[1] != len(feature_names) or Theta.shape[1] != len(sp_features):
        raise ValueError(
            f"feature_names/sp_features length mismatch: "
            f"Theta has {Theta.shape[1]} cols, feature_names={len(feature_names)}, sp_features={len(sp_features)}"
        )

    nfeat = Theta.shape[1]
    kept_idx: List[int] = []

    for j in range(nfeat):
        nm = feature_names[j].strip()
        if nm == constant_name:
            kept_idx.append(j)
            continue

        col = Theta[:, j]
        mu = float(np.mean(col))
        sig = float(np.std(col))
        cv = sig / (abs(mu) + 1e-12)

        if np.isfinite(cv) and cv < cv_thresh:
            continue
        kept_idx.append(j)

    Theta_nc = Theta[:, kept_idx]
    kept_names = [feature_names[j] for j in kept_idx]
    kept_sp = [sp_features[j] for j in kept_idx]
    return Theta_nc, kept_idx, kept_names, kept_sp


def score_complexity(feature_name: str) -> int:
    """
    Heuristic library complexity: higher → richer monomial / interaction.
    Used by :func:`prefer_parsimony` to break correlation ties without relying on column order.
    """
    if feature_name == "1":
        return 0
    score = 1
    score += feature_name.count("*")
    score += feature_name.count("**")
    return score


def _factor_base_symbol(factor: str) -> str:
    """One multiplicative factor with any trailing ``**exponent`` removed (exponent ignored)."""
    s = factor.strip()
    if not s or s == "1":
        return ""
    if "**" in s:
        base, _, _ = s.partition("**")
        return base.strip()
    return s


def feature_roots(name: str) -> frozenset[str]:
    """
    Base-ingredient set for a library name: split on ``*``, strip ``**power`` per factor, dedupe.

    Examples: ``alpha`` and ``alpha**2`` → both ``frozenset({'alpha'})``;
    ``q_dyn*alpha`` and ``q_dyn*alpha**2`` → both ``frozenset({'q_dyn', 'alpha'})``;
    ``Q`` vs ``elv_act`` → different sets, so highly correlated pairs in different families are not resolved.
    """
    name = str(name).strip()
    if name == "1":
        return frozenset()
    bases: List[str] = []
    for part in name.split("*"):
        b = _factor_base_symbol(part)
        if b:
            bases.append(b)
    return frozenset(bases)


def prefer_parsimony(
    Theta: np.ndarray,
    feature_names: List[str],
    sp_features: List[Any],
    threshold: float = 0.995,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, List[int], List[str], List[Any], List[Tuple[str, str, float, str]]]:
    """
    Greedy **Pearson** correlation pruning with **same-variable (same-root) parsimony**.

    When |correlation| exceeds ``threshold``, a pair is resolved **only if**
    :func:`feature_roots` agrees for both names (same multiset of base symbols after stripping
    ``**exponents`` on each ``*``-separated factor). Then the survivor is the **simpler** term by
    :func:`score_complexity` (lower score). If roots differ—e.g. ``Q`` vs ``elv_act``, or
    ``alpha`` vs ``q_dyn*alpha``—the pair is **ignored** and both columns can remain.

    Important: high |corr| on data does not mean only one term belongs in the true model.
    With ``SINDyRunConfig.prefer_parsimony=False`` the pipeline keeps all Θ columns (use ridge instead).

    Returns
    -------
    dropped_report : list of (dropped_name, kept_name, |r|, reason_tag)
    """
    Theta = np.asarray(Theta, dtype=float)
    if Theta.ndim != 2:
        raise ValueError(f"Theta must be 2D, got {Theta.shape}")
    if Theta.shape[1] != len(feature_names) or Theta.shape[1] != len(sp_features):
        raise ValueError(
            f"feature_names/sp_features length mismatch: "
            f"Theta has {Theta.shape[1]} cols, feature_names={len(feature_names)}, sp_features={len(sp_features)}"
        )
    if Theta.shape[1] == 0:
        return Theta.copy(), [], [], [], []
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in [0,1], got {threshold}")

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(Theta.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)

    kept_idx: List[int] = []
    dropped_report: List[Tuple[str, str, float, str]] = []

    for i in range(Theta.shape[1]):
        ni = feature_names[i]
        si = score_complexity(ni)
        redundant = False

        for j_pos, j in enumerate(kept_idx):
            if corr[i, j] <= threshold + eps:
                continue
            nj = feature_names[j]
            if feature_roots(ni) != feature_roots(nj):
                continue

            sj = score_complexity(nj)

            prefer_replace_j_with_i = si < sj
            swap_reason = "parsimony_same_family_swap_simpler"
            drop_i_reason = "parsimony_same_family_keep_simpler"

            if prefer_replace_j_with_i:
                dropped_report.append((nj, ni, float(corr[i, j]), swap_reason))
                kept_idx[j_pos] = i
                redundant = True
                break

            dropped_report.append((ni, nj, float(corr[i, j]), drop_i_reason))
            redundant = True
            break

        if not redundant:
            kept_idx.append(i)

    Theta_clean = Theta[:, kept_idx]
    kept_names = [feature_names[i] for i in kept_idx]
    kept_sp = [sp_features[i] for i in kept_idx]
    return Theta_clean, kept_idx, kept_names, kept_sp, dropped_report


def remove_collinear_features(
    Theta: np.ndarray,
    feature_names: List[str],
    sp_features: List[Any],
    threshold: float = 0.995,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, List[int], List[str], List[Any], List[Tuple[str, str, float, str]]]:
    """
    Deprecated name; calls :func:`prefer_parsimony` with default keyword arguments.
    """
    return prefer_parsimony(Theta, feature_names, sp_features, threshold=threshold, eps=eps)


class MAPSINDyRegressor:
    def __init__(self, lam=1e-3, sigma2=1.0, max_iter=1000, tol=1e-6):
        self.lam = float(lam)
        self.sigma2 = float(sigma2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_ = None

    def _soft(self, z, thr):
        return np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)

    def fit(self, Theta: np.ndarray, Y: np.ndarray) -> "MAPSINDyRegressor":
        Theta = np.asarray(Theta, float)
        Y = np.asarray(Y, float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        N, K = Theta.shape
        _, d = Y.shape
        Xi = np.zeros((K, d))
        L = (np.linalg.norm(Theta, 2) ** 2) / self.sigma2 + 1e-12

        for j in range(d):
            xi = np.zeros(K)
            yj = Y[:, j]
            for _ in range(self.max_iter):
                grad = (Theta.T @ (Theta @ xi - yj)) / self.sigma2
                xi_new = self._soft(xi - grad / L, self.lam / L)
                if np.max(np.abs(xi_new - xi)) < self.tol:
                    break
                xi = xi_new
            Xi[:, j] = xi

        self.coef_ = Xi
        return self

def ensemble_pareto_analysis(
    Theta: np.ndarray,
    Y: np.ndarray,
    thresholds: np.ndarray,
    B: int = 100,
    frac: float = 0.8,
    random_state: int = 0,
    sparsity_bias: float = 0.0,
    pareto_pick: str = "knee",
    pareto_dial: Optional[float] = None,
    equal_weight_per_target: bool = False,
    alpha_ridge: float = 1e-6,
    bic_mse_variance_fraction: float = 1e-3,
    bic_mse_floor_epsilon: float = 1e-12,
) -> dict:
    rng = np.random.default_rng(random_state)
    N, K = Theta.shape
    _, d = Y.shape
    if equal_weight_per_target and d > 1:
        var_y = float(np.mean(np.var(Y, axis=0))) + 1e-12
    else:
        var_y = np.var(Y) + 1e-12

    stlsq = AdaptiveSTLSQ(alpha_ridge=alpha_ridge)
    coefs = np.zeros((B, K, d), float)
    use_dial = pareto_dial is not None and 0 <= pareto_dial <= 1
    pick_l = (pareto_pick or "bic").lower()
    use_last = pick_l == "last"
    use_bic = pick_l in ("bic", "per_target_bic")

    for b in range(B):
        idx = rng.choice(N, size=int(frac * N), replace=True)
        results, pareto = stlsq.pareto_analysis(
            Theta[idx], Y[idx], thresholds,
            equal_weight_per_target=equal_weight_per_target,
        )
        n_b = int(len(idx))
        if use_dial:
            best = pick_by_dial(pareto, var_y, dial=pareto_dial)
        elif use_bic:
            best, _ = pick_by_bic(
                results,
                n_b,
                Y.shape[1],
                Y[idx],
                equal_weight_per_target=equal_weight_per_target,
                variance_fraction=bic_mse_variance_fraction,
                mse_floor_epsilon=bic_mse_floor_epsilon,
            )
        elif use_last:
            best = pareto[-1]
        elif sparsity_bias == 0:
            best = knee_by_curvature(pareto, var_y, use_log=True)
        else:
            best = pick_with_sparsity_knob(pareto, var_y, lam=sparsity_bias)
        coefs[b] = best["coef"]

    inclusion_probs = (np.abs(coefs) > 0).mean(axis=0)
    # Median is more robust to outliers in ensemble runs
    coef_median = np.median(coefs, axis=0)

    return {
        "coef": coef_median,
        "inclusion_probs": inclusion_probs,
        "var_y": var_y,
        "mode": "pareto_ensemble",
    }


def ensemble_map_analysis(
    Theta: np.ndarray,
    Y: np.ndarray,
    lam: float,
    sigma2: float,
    B: int = 100,
    frac: float = 0.8,
    random_state: int = 0,
) -> dict:
    rng = np.random.default_rng(random_state)
    N, K = Theta.shape
    _, d = Y.shape

    coefs = np.zeros((B, K, d), float)
    for b in range(B):
        idx = rng.choice(N, size=int(frac * N), replace=True)
        reg = MAPSINDyRegressor(lam=lam, sigma2=sigma2)
        reg.fit(Theta[idx], Y[idx])
        coefs[b] = reg.coef_

    inclusion_probs = (np.abs(coefs) > 0).mean(axis=0)
    coef_median = np.median(coefs, axis=0)

    return {
        "coef": coef_median,
        "inclusion_probs": inclusion_probs,
        "mode": "bayes_map_ensemble",
    }

@dataclass
class AdaptiveSTLSQ:
    # L2 strength for sklearn Ridge inside each STLSQ least-squares step (~1e-3 for nearly
    # collinear Θ, e.g. α vs sin(α)). Passed through from SINDyRunConfig.alpha_ridge in normal use.
    alpha_ridge: float = 1e-6
    max_iter: int = 20
    tol: float = 1e-8

    def fit(self, X: np.ndarray, y: np.ndarray, threshold: float) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        thr = float(threshold)
        alpha = float(self.alpha_ridge)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.shape}")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(f"y must be 1D or 2D, got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Row mismatch: X {X.shape}, y {y.shape}")
        if X.shape[1] == 0:
            return np.zeros((0, y.shape[1]), dtype=float)

        n_features = X.shape[1]
        n_targets = y.shape[1]

        ridge0 = Ridge(alpha=alpha, fit_intercept=False)
        ridge0.fit(X, y)
        cc0 = np.asarray(ridge0.coef_, dtype=float)
        if cc0.ndim == 1:
            coef = cc0.reshape(-1, 1)
        else:
            coef = cc0.T

        for _ in range(self.max_iter):
            coef_old = coef.copy()

            mask = np.abs(coef) > thr
            coef[~mask] = 0.0

            for j in range(n_targets):
                active = mask[:, j]
                if int(np.sum(active)) == 0:
                    continue

                Xa = X[:, active]
                rj = Ridge(alpha=alpha, fit_intercept=False)
                rj.fit(Xa, y[:, j])
                coef[active, j] = np.asarray(rj.coef_, dtype=float).ravel()

            if float(np.max(np.abs(coef - coef_old))) < float(self.tol):
                break

        # Debias: plain least squares on the final support (undoes Ridge shrinkage on kept terms).
        for j in range(n_targets):
            act = np.abs(coef[:, j]) > thr
            if not np.any(act):
                continue
            Xa = X[:, act]
            coef[act, j] = np.linalg.lstsq(Xa, y[:, j], rcond=None)[0]

        return coef

    def fit_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        thresholds: np.ndarray,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        thresholds = np.asarray(thresholds, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Row mismatch: X {X.shape}, y {y.shape}")
        if thresholds.ndim != 1 or thresholds.size == 0:
            raise ValueError("thresholds must be a non-empty 1D array")

        kf = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(random_state))
        cv_scores: List[float] = []

        for thr in thresholds:
            fold_errs: List[float] = []
            for tr_idx, va_idx in kf.split(X):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]
                coef = self.fit(Xtr, ytr, float(thr))
                ypred = Xva @ coef
                fold_errs.append(float(np.mean((yva - ypred) ** 2)))
            cv_scores.append(float(np.mean(fold_errs)))

        best_i = int(np.argmin(cv_scores))
        best_thr = float(thresholds[best_i])
        best_coef = self.fit(X, y, best_thr)

        return best_coef, best_thr, {"thresholds": thresholds, "cv_scores": np.asarray(cv_scores, dtype=float)}

    def fit_with_aic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        thresholds: np.ndarray,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        thresholds = np.asarray(thresholds, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Row mismatch: X {X.shape}, y {y.shape}")
        if thresholds.ndim != 1 or thresholds.size == 0:
            raise ValueError("thresholds must be a non-empty 1D array")

        n = int(X.shape[0])
        aic_scores: List[float] = []

        for thr in thresholds:
            coef = self.fit(X, y, float(thr))
            ypred = X @ coef
            rss = float(np.sum((y - ypred) ** 2))
            k = int(np.count_nonzero(coef))

            aic = n * float(np.log(rss / max(n, 1) + 1e-10)) + 2.0 * k
            if n - k - 1 > 0:
                aic += 2.0 * k * (k + 1) / (n - k - 1)
            aic_scores.append(float(aic))

        best_i = int(np.argmin(aic_scores))
        best_thr = float(thresholds[best_i])
        best_coef = self.fit(X, y, best_thr)

        return best_coef, best_thr, {"thresholds": thresholds, "aic_scores": np.asarray(aic_scores, dtype=float)}

    def pareto_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        thresholds: np.ndarray,
        equal_weight_per_target: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        thresholds = np.asarray(thresholds, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Row mismatch: X {X.shape}, y {y.shape}")
        if thresholds.ndim != 1 or thresholds.size == 0:
            raise ValueError("thresholds must be a non-empty 1D array")

        results: List[Dict[str, Any]] = []
        for thr in thresholds:
            coef = self.fit(X, y, float(thr))
            ypred = X @ coef
            if equal_weight_per_target and y.shape[1] > 1:
                mse_by_target = np.mean((y - ypred) ** 2, axis=0)
                mse = float(np.mean(mse_by_target))
            else:
                mse = float(np.mean((y - ypred) ** 2))
            complexity = int(np.count_nonzero(coef))
            results.append(
                {"threshold": float(thr), "mse": mse, "complexity": complexity, "coef": coef}
            )

        # Pareto set (non-dominated): minimize mse and complexity
        pareto: List[Dict[str, Any]] = []
        for r in results:
            dominated = False
            for o in results:
                if (o["mse"] <= r["mse"] and o["complexity"] < r["complexity"]) or (
                    o["mse"] < r["mse"] and o["complexity"] <= r["complexity"]
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)

        # Stable order: increasing complexity, then mse
        pareto = sorted(pareto, key=lambda d: (d["complexity"], d["mse"]))

        return results, pareto
        
def summarize_consensus(
    results_by_mode: Dict[str, Dict],
    feature_names: Optional[List[str]] = None,
    Xi_true: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    pi_threshold: float = 0.8,
    coef_tol_frac: float = 0.25,
    zero_tol: float = 1e-10,
    max_lines_per_section: int = 15,
    true_decimal_fmt: str = ".5f",
) -> Dict[str, List[str]]:
    """
    Automated Consensus Report.
    Handles key mapping (coef_phys vs coef_kept vs coef) and probability inference internally.
    If Xi_true is provided, adds a true_coefficients section (decimal form) for reference.
    """
    mode_names = list(results_by_mode.keys())
    if not mode_names:
        return {"core_terms": [], "unstable_terms": [], "missing_true_terms": [], "true_coefficients": [], "notes": ["No modes provided."]}

    first_res = results_by_mode[mode_names[0]]
    if feature_names is None:
        feature_names = first_res.get("kept_names", first_res.get("feature_names"))
        if feature_names is None:
            raise ValueError("feature_names must be provided or exist within the result dicts.")
    # AUTOMATION: Internal helper to resolve coefficient keys
    def get_xi(res):
        # Prefer physical-unit coefficients for diagnostics/reporting.
        return np.asarray(res.get("coef_phys", res.get("coef_kept", res.get("coef"))), float)

    K, d = get_xi(results_by_mode[mode_names[0]]).shape
    if target_names is None:
        target_names = first_res.get("target_names")
    if target_names is None:
        target_names = [f"target_{j}" for j in range(d)]

    # Per-mode probability keys: list π_ens, π_bayes_map, π_bayes_ens separately in report
    def _pi_key(mode: str) -> str:
        m = mode.lower()
        if m == "bayes_ens" or (m != "ensemble" and "bayes" in m and "ens" in m):
            return "pi_bayes_ens"
        if m == "bayes_map" or ("map" in m and "bayes" in m):
            return "pi_bayes_map"
        if m == "ensemble" or ("ensemble" in m and "bayes" not in m):
            return "pi_ens"
        return "pi_det"

    def gather_entry(k: int, j: int):
        info = {}
        for mode, res in results_by_mode.items():
            Xi = get_xi(res)
            c = float(Xi[k, j])
            nz = abs(c) > zero_tol
            info[mode] = {"coef": c, "nonzero": nz}
            key = _pi_key(mode)
            if "inclusion_probs" in res and res["inclusion_probs"] is not None:
                pis = np.asarray(res["inclusion_probs"], float)
                p_val = float(pis[k, j])
                info[mode][key] = p_val
            else:
                info[mode]["pi_det"] = 1.0 if nz else 0.0
        return info

    def get_pi_summary(entry) -> str:
        """List each method's probability separately: π_ens, π_bayes_map, π_bayes_ens."""
        parts = []
        for mode in mode_names:
            v = entry.get(mode, {})
            pk = _pi_key(mode)
            if pk in v and pk != "pi_det":
                parts.append(f"{mode}: π={v[pk]:.2f}")
            else:
                det = 1.0 if v.get("nonzero") else 0.0
                parts.append(f"{mode}: det={det:.0f}")
        return "; ".join(parts) if parts else "π=N/A"

    def get_coef_and_pi_line(entry) -> str:
        """Per-mode coefficient and probability for report line."""
        parts = []
        for mode in mode_names:
            v = entry[mode]
            c = v["coef"]
            pk = _pi_key(mode)
            if pk in v and pk != "pi_det":
                parts.append(f"{mode}: {c:+.3g} (π={v[pk]:.2f})")
            else:
                det = "1" if v.get("nonzero") else "0"
                parts.append(f"{mode}: {c:+.3g} (det={det})")
        return " | ".join(parts)

    def max_pi_all(entry) -> float:
        vals = []
        for v in entry.values():
            for k in ("pi_ens", "pi_bayes_map", "pi_bayes_ens", "pi_bay", "pi", "pi_det"):
                if k in v:
                    vals.append(float(v[k]))
        return max(vals) if vals else 0.0

    core_terms, unstable_terms, missing_true_terms = [], [], []

    for k in range(K):
        fname = feature_names[k]
        for j in range(d):
            entry = gather_entry(k, j)
            if not any(v["nonzero"] for v in entry.values()): continue

            m_pi = max_pi_all(entry)
            all_nz = all(v["nonzero"] for v in entry.values())
            pi_str = get_pi_summary(entry)

            if all_nz and m_pi >= pi_threshold:
                true_val = f"true={float(Xi_true[k,j]):{true_decimal_fmt}}, " if Xi_true is not None else ""
                coef_pi = get_coef_and_pi_line(entry)
                tname = target_names[j] if j < len(target_names) else f"target_{j}"
                core_terms.append(f"{fname} -> {tname}: agrees, {true_val}{coef_pi}")
            else:
                act = [m for m, v in entry.items() if v["nonzero"]]
                coef_pi = get_coef_and_pi_line(entry)
                tname = target_names[j] if j < len(target_names) else f"target_{j}"
                true_val = f"true={float(Xi_true[k,j]):{true_decimal_fmt}}, " if Xi_true is not None else ""
                unstable_terms.append(f"{fname} -> {tname}: active in {act}, {true_val}{coef_pi}")

    # Missing True Terms: terms that are meaningfully non-zero in the true model but SINDy did not select.
    # Use a display-scale threshold so we don't flag numerical-noise "true" coefficients (e.g. 1e-8 from
    # ridge projection) as "missed" — otherwise you get "y*z -> x: TRUE=0.00000 but missed" for Lorenz.
    missing_true_min = 1e-5  # only treat as "true term" if |coef| > this
    if Xi_true is not None:
        Xi_true = np.asarray(Xi_true, float)
        true_norms = np.linalg.norm(Xi_true, axis=1)
        thresh = np.percentile(true_norms[true_norms > 0], 50) if np.any(true_norms > 0) else 0.0
        for k in range(K):
            if true_norms[k] <= thresh: continue
            for j in range(d):
                ct = float(Xi_true[k, j])
                if abs(ct) <= zero_tol: continue
                if abs(ct) < missing_true_min: continue  # skip numerical-noise "true" coefficients
                entry = gather_entry(k, j)
                if all(abs(v["coef"]) <= coef_tol_frac * abs(ct) for v in entry.values()) and max_pi_all(entry) < pi_threshold:
                    tname = target_names[j] if j < len(target_names) else f"target_{j}"
                    missing_true_terms.append(f"{feature_names[k]} -> {tname}: TRUE={float(ct):{true_decimal_fmt}} but missed")

    # True coefficients reference (decimal form) when Xi_true is available
    true_coefficients = []
    if Xi_true is not None:
        for k in range(K):
            for j in range(d):
                c = float(Xi_true[k, j])
                if abs(c) > zero_tol:
                    tname = target_names[j] if j < len(target_names) else f"target_{j}"
                    true_coefficients.append(f"  {feature_names[k]} -> {tname}: true = {c:{true_decimal_fmt}}")

    notes = []
    if unstable_terms:
        notes.append(
            "Unstable/ambiguous terms: methods disagree on inclusion or have low inclusion probability. "
            "Bayesian and ensemble methods often select sparser models than single-run Pareto, so more "
            "terms can appear ambiguous when comparing across methods (expected)."
        )
    return {
        "core_terms": core_terms,
        "unstable_terms": unstable_terms,
        "missing_true_terms": missing_true_terms,
        "true_coefficients": true_coefficients,
        "notes": notes,
    }