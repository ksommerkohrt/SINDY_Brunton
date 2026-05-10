# SINDY_Brunton

Minimal, self-contained **SINDy** workflows in the spirit of Brunton et al. (Pareto STLSQ, optional BIC/dial selection, bootstrap ensemble, forward integration, truth comparison). This repo is **not** the full SINDY Engineering Toolkit—only the packages needed for the demos below.

## Layout

| Path | Role |
|------|------|
| `lorenz/` | Lorenz simulation, `SINDySystemModel` bridge, SINDy configs, forward integration + 3D plots |
| `cylinder_flow/` | Cylinder wake **mean-field (PNAS Eq. 8)** simulator, trajectory validation metrics, **POD + shift-mode** processing for snapshot matrices, SymPy truth model + cylinder-oriented SINDy config |
| `sindy/` | Feature library, STLSQ / Pareto / BIC fit, pipeline runner, diagnostics |
| `idtools/` | Preprocessing (incl. identity scaler), `compare_to_truth`, excitation and auto-config hooks |
| `run_lorenz.py` | CLI: Lorenz BIC/ensemble runs, `compare_to_truth`, PNGs under `outputs/lorenz/` |
| `Run_Lorenz.ipynb` | Notebook version of the Lorenz workflow |
| `Run_Cylinder_Wake.ipynb` | Cylinder: synthetic Eq. 8 trajectory, embedded POD+shift recovery, SINDy fit vs symbolic truth, 3D and time-series plots vs Brunton reference |

## Setup

```bash
cd /path/to/SINDY_Brunton
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run from this directory so imports resolve:

```bash
PYTHONPATH=. python run_lorenz.py
```

Open `Run_Lorenz.ipynb` or `Run_Cylinder_Wake.ipynb` with the kernel working directory set to the repo root.

## Lorenz demo

`lorenz_sindy_config_bic_standard()` uses **`scaler_kind="identity"`** and **`normalize_library_columns=False`** (paper-like Lorenz STLSQ on raw states).

## Cylinder wake (reduced coordinates)

**References:** Brunton et al., PNAS (2016), Eq. 8; Noack et al., J. Fluid Mech. (2003).

- **`cylinder_flow/mean_field_simulator.py`** — Integrate the three-mode ODE \((x,y,z)\) with `scipy.integrate.solve_ivp`.
- **`cylinder_flow/snapshot_pod_shift.py`** — Load `cy10`-style / `cyl0`-style snapshots; **POD from time fluctuations** `X - mean_t(X)`; **shift direction** from `mean_t(X) - u_base` (orthogonal to first two POD modes); **amplitudes** `a_k` by projecting **full** steady-centered snapshots `X - u_base`. Optional **`calibrate_a3_scale`** / **`estimate_a3_scale_lstsq`** to align `a3` with `a1² + a2²`. CLI: `python -m cylinder_flow.snapshot_pod_shift …`.
- **`cylinder_flow/mean_field_sindy_model.py`** — `build_cylinder_mean_field_model()`, polynomial budget for Eq. 8, **`cylinder_sindy_config_bic_standard()`** (`normalize_library_columns=True`, **`pareto_dial=1.0`** so the `dz/dt` row is not over-pruned when Θ is column-normalized).
- **`cylinder_flow/validation.py`** — Compare a candidate \((x,y,z)\) trajectory to the reference integrator.

**Time base:** `t` passed into the pipeline must match the simulation clock of your data (standard Brunton-style cylinder examples are often ~**200** time units; wrong duration rescales every time-coupled coefficient in the identified ODE).

## License

Code was split out from the parent project; retain any original license notices if you redistribute.
