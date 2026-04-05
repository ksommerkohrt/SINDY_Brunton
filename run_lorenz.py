#!/usr/bin/env python3
"""
Run SINDy on the Lorenz system to demonstrate toolkit generalizability.

Primary path: **Pareto + BIC** on **unscaled** Lorenz states (``scaler_kind=identity``), then **Pareto
ensemble + BIC** with the same preprocessing (bootstrap median coefficients).

Optional extras:
  - Auto mode: scout → recommended config → single run.
  - Fixed paper-like run (exact derivatives + single threshold) for historical comparison.

Usage (from this repo root): PYTHONPATH=. python run_lorenz.py
Set run_auto=False or run_fixed=False for a faster run.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
from lorenz.simulate import simulate_lorenz
from lorenz.model import (
    build_lorenz_model,
    lorenz_sindy_config_bic_standard,
    lorenz_sindy_config_ensemble_bic_standard,
)
from lorenz.integrated import integrated_3d_comparison
from sindy.pipeline import run_sindy_pipeline_general, run_sindy_pipeline_auto, SINDyRunConfig
from idtools.compare_to_truth import compare_to_truth
from idtools.diagnostic_suite import run_diagnostic_suite, DiagnosticSuiteConfig

OUTPUT_DIR = "outputs/lorenz/"
RUN_AUTO = True
RUN_FIXED_PAPER_LIKE = True
# Forward-integration plot uses every k-th sample of t (full 10⁵ points is slow; set 1 for dense grid).
LORENZ_INTEGRATION_T_STRIDE = 10


def main():
    print("1. Simulating Lorenz system (paper-like: dt=0.001, 100 s, x0=(-8,8,27))...")
    t, Z_phys = simulate_lorenz(
        t_span=(0.0, 100.0),
        dt=0.001,
        x0=(-8.0, 8.0, 27.0),
        random_state=42,
    )
    print(f"   t: {t.size} points, dt ≈ {np.median(np.diff(t)):.6f}, Z_phys: {Z_phys.shape}")

    print("\n2. Building Lorenz model (SymPy RHS + pipeline interface)...")
    model = build_lorenz_model()

    budget = {
        "x": {"lin": 2, "trig": 0, "inv": 0, "sat": 0},
        "y": {"lin": 2, "trig": 0, "inv": 0, "sat": 0},
        "z": {"lin": 2, "trig": 0, "inv": 0, "sat": 0},
    }

    use_exact_derivatives = True
    Z_dot_phys = None
    if use_exact_derivatives:
        Z_dot_phys = np.column_stack([
            model.rhs_lambdified[i](*Z_phys.T) for i in range(len(model.rhs_lambdified))
        ])
        print("   Using exact derivatives from model RHS (Z_dot_phys) — paper-like.")

    results = {}

    # --- 3a. Primary: Pareto sweep + BIC pick + standard scaling ---
    print("\n3a. Running Pareto + BIC (no state scaling before SINDy)...")
    cfg_bic = lorenz_sindy_config_bic_standard()
    res_bic = run_sindy_pipeline_general(
        t=t,
        Z_phys=Z_phys,
        model=model,
        budget=budget,
        config=cfg_bic,
        Z_dot_phys=Z_dot_phys,
        run_diagnostics=True,
        output_dir=OUTPUT_DIR + "bic_standard/",
    )
    results["bic_standard"] = res_bic
    val_bic = res_bic["validation"]
    print(f"   BIC (unscaled) validation: R2(mean) = {val_bic['r2_mean']:.4f}, RMSE = {val_bic['rmse']:.4g}")
    print("   Discovered equations (BIC, raw state):")
    for name, expr in res_bic["fit"]["equations"].items():
        print(f"      d{name}/dt = {expr}")
    print("\n   Post-hoc vs true Lorenz (not used in fitting):")
    compare_to_truth(res_bic, verbose=True)

    print("\n   3D true vs SINDy forward integration (subsampled time grid)...")
    cmp_bic = integrated_3d_comparison(
        res_bic,
        t,
        Z_phys,
        out_path=OUTPUT_DIR + "bic_standard/lorenz_3d_true_vs_sindy.png",
        title="Lorenz BIC (unscaled): reference data vs SINDy integration",
        t_subsample_step=LORENZ_INTEGRATION_T_STRIDE,
    )
    print(f"      RMSE(state) on grid ≈ {cmp_bic['rmse_state']:.4g}  →  {OUTPUT_DIR}bic_standard/lorenz_3d_true_vs_sindy.png")
    plt.close(cmp_bic["fig"])

    # --- 3a2. Pareto ensemble + BIC + same scaling (bootstrap median coefs) ---
    print("\n3a2. Running Pareto ENSEMBLE + BIC (unscaled state; median over bootstrap fits)...")
    cfg_ens = lorenz_sindy_config_ensemble_bic_standard(ensemble_B=50, ensemble_frac=0.8)
    res_ens = run_sindy_pipeline_general(
        t=t,
        Z_phys=Z_phys,
        model=model,
        budget=budget,
        config=cfg_ens,
        Z_dot_phys=Z_dot_phys,
        run_diagnostics=True,
        output_dir=OUTPUT_DIR + "ensemble_bic_standard/",
    )
    results["ensemble_bic_standard"] = res_ens
    val_ens = res_ens["validation"]
    print(f"   Ensemble BIC (unscaled) validation: R2(mean) = {val_ens['r2_mean']:.4f}, RMSE = {val_ens['rmse']:.4g}")
    print("   Discovered equations (ensemble + BIC, raw state):")
    for name, expr in res_ens["fit"]["equations"].items():
        print(f"      d{name}/dt = {expr}")
    inc = res_ens["fit"].get("inclusion_probs")
    if inc is not None:
        print(f"   Term inclusion probabilities: min={float(np.min(inc)):.3f}, max={float(np.max(inc)):.3f}")
    print("\n   Post-hoc vs true Lorenz (ensemble; not used in fitting):")
    compare_to_truth(res_ens, verbose=True)

    print("\n   3D true vs ensemble-SINDy forward integration...")
    cmp_ens = integrated_3d_comparison(
        res_ens,
        t,
        Z_phys,
        out_path=OUTPUT_DIR + "ensemble_bic_standard/lorenz_3d_true_vs_sindy.png",
        title="Lorenz ensemble BIC (unscaled): reference data vs SINDy integration",
        t_subsample_step=LORENZ_INTEGRATION_T_STRIDE,
    )
    print(f"      RMSE(state) on grid ≈ {cmp_ens['rmse_state']:.4g}  →  {OUTPUT_DIR}ensemble_bic_standard/lorenz_3d_true_vs_sindy.png")
    plt.close(cmp_ens["fig"])

    # --- 3b. Optional AUTO mode (scout → recommend → run) ---
    if RUN_AUTO:
        print("\n3b. Running AUTO mode (scout → recommended config → single run)...")
        res_auto = run_sindy_pipeline_auto(
            t=t,
            Z_phys=Z_phys,
            model=model,
            budget=budget,
            Z_dot_phys=Z_dot_phys,
            run_diagnostics=True,
            output_dir=OUTPUT_DIR + "auto/",
            verbose_auto=True,
        )
        results["auto"] = res_auto
        val_auto = res_auto["validation"]
        print(f"   Auto validation: R2(mean) = {val_auto['r2_mean']:.4f}, RMSE = {val_auto['rmse']:.4g}")
        if res_auto.get("auto_config", {}).get("reasons"):
            print("   Auto reasons:")
            for r in res_auto["auto_config"]["reasons"]:
                print(f"      - {r}")

    # --- 3c. Optional fixed paper-like run (single threshold; comparison baseline) ---
    if RUN_FIXED_PAPER_LIKE:
        print("\n3c. Running FIXED paper-like config (exact deriv + single threshold 0.05)...")
        config_fixed = SINDyRunConfig(
            savgol_window=15,
            savgol_poly=3,
            prefer_parsimony=False,
            alpha_ridge=1e-8,
            normalize_library_columns=False,
            use_physical_library=True,
            single_threshold=0.05,
            n_thresholds=1,
            pareto_pick_mode="last",
        )
        res_fixed = run_sindy_pipeline_general(
            t=t,
            Z_phys=Z_phys,
            model=model,
            config=config_fixed,
            budget=budget,
            Z_dot_phys=Z_dot_phys,
            run_diagnostics=True,
            output_dir=OUTPUT_DIR + "fixed/",
        )
        results["fixed"] = res_fixed
        val_fixed = res_fixed["validation"]
        print(f"   Fixed validation: R2(mean) = {val_fixed['r2_mean']:.4f}, RMSE = {val_fixed['rmse']:.4g}")
        print("   Discovered equations (fixed config):")
        for name, expr in res_fixed["fit"]["equations"].items():
            print(f"      d{name}/dt = {expr}")
        print("\n   Unified diagnostics (fixed run):")
        run_diagnostic_suite(res_fixed, config=DiagnosticSuiteConfig(run_excitation=False), verbose=True)

    # --- 4. Summary comparison ---
    print("\n4. Comparison summary")
    print("   BIC (unscaled): R2(mean) = {:.4f}".format(results["bic_standard"]["validation"]["r2_mean"]))
    print("   Ensemble BIC:   R2(mean) = {:.4f}".format(results["ensemble_bic_standard"]["validation"]["r2_mean"]))
    if RUN_AUTO and "auto" in results:
        print("   Auto:           R2(mean) = {:.4f}".format(results["auto"]["validation"]["r2_mean"]))
    if RUN_FIXED_PAPER_LIKE:
        print("   Fixed:          R2(mean) = {:.4f}".format(results["fixed"]["validation"]["r2_mean"]))

    return results


if __name__ == "__main__":
    main()
