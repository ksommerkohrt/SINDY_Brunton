# SINDY_Brunton

Minimal, self-contained **Lorenz SINDy** demo (Pareto STLSQ, BIC selection, optional bootstrap ensemble, forward integration and 3D plots), extracted for teaching and reproduction in the spirit of Brunton et al.

This is **not** the full SINDY Engineering Toolkit—only the packages and modules required to run `run_lorenz.py` and `Run_Lorenz.ipynb`.

## Layout

| Path | Role |
|------|------|
| `lorenz/` | Lorenz simulation, `SINDySystemModel` bridge, SINDy configs, forward integration + 3D plots |
| `sindy/` | Library, STLSQ/Pareto/BIC fit, pipeline runner, diagnostic figures (no `physics_filters`—Lorenz does not use `library_keep_feature`) |
| `idtools/` | Preprocessing (incl. identity scaler), truth comparison, excitation + auto-config hooks used by the pipeline |
| `run_lorenz.py` | CLI entry: BIC fit, ensemble fit, optional auto/fixed baselines, `compare_to_truth`, 3D PNGs under `outputs/lorenz/` |
| `Run_Lorenz.ipynb` | Notebook version of the same workflow |

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

Or open `Run_Lorenz.ipynb` and execute cells (kernel cwd should be the repo root).

## Note on scaling

`lorenz_sindy_config_bic_standard()` uses **`scaler_kind="identity"`** and **`normalize_library_columns=False`**, so Lorenz states are **not** z-scored or max-abs scaled before building Θ and running STLSQ.

## License

Code was split out from the parent project; retain any original license notices if you redistribute.
