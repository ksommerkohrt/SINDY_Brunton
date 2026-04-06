"""
Build POD + shift-mode coordinates from cylinder wake snapshots (e.g. cy10.snapshot) for SINDy.

Expected layout: ``X[i, j]`` is the *i*-th spatial degree of freedom at time index *j*
(shape ``(n_grid_points, n_timesteps)``), matching the Brunton et al. PNAS cylinder discussion.

Base flow ``u_s`` (e.g. from cyl0.snapshot) must be a length-``n_grid_points`` vector in the
same ordering as rows of ``X``.

**Split construction (Noack / Brunton):** ``phi1``, ``phi2`` come from **time fluctuations**
``X - mean_t(X)`` so the shedding subspace is not polluted by the mean-shift component (which
would otherwise dominate ``svd(X - u_base)`` and collide with the shift mode). Modal amplitudes
``(a1,a2,a3)`` project the **full** steady-centered field ``X - u_base`` onto ``phi1``, ``phi2``,
and the orthogonalized shift direction ``delta_u ∝ mean_t(X) - u_base``.

If your files store time along rows instead, pass ``time_axis=0`` or ``transpose=True`` after load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def estimate_a3_scale_lstsq(
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    *,
    eps: float = 1e-30,
) -> float:
    """
    Scalar ``s`` that minimizes ``‖s·a3 - (a1² + a2²)‖₂`` (same length arrays, one sample per time).

    Use ``a3_calibrated = s * a3`` when POD norms make ``a3`` the wrong amplitude relative to
    ``a1²+a2²`` for a target manifold ``z ≈ x² + y²`` in reduced coordinates.
    """
    a1 = np.asarray(a1, dtype=float).ravel()
    a2 = np.asarray(a2, dtype=float).ravel()
    a3 = np.asarray(a3, dtype=float).ravel()
    if not (a1.size == a2.size == a3.size):
        raise ValueError("a1, a2, a3 must have the same length")
    r2 = a1 * a1 + a2 * a2
    den = float(np.dot(a3, a3))
    if den < eps:
        return 1.0
    return float(np.dot(a3, r2) / den)


@dataclass
class PODShiftResult:
    """Outputs for SINDy (Eq.~8-style three-mode coordinates)."""

    phi1: np.ndarray
    phi2: np.ndarray
    shift_mode: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    a3: np.ndarray
    X_fluctuations: np.ndarray
    X_total_dev: np.ndarray
    u_time_mean: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    a3_scale_applied: float = 1.0

    @property
    def X_dev(self) -> np.ndarray:
        """Alias for :attr:`X_total_dev` (``X - u_base`` per column)."""
        return self.X_total_dev

    @property
    def Z_sindy(self) -> np.ndarray:
        """Shape ``(n_timesteps, 3)`` — rows are time samples for :func:`idtools.preprocess.preprocess_timeseries`."""
        return np.column_stack([self.a1, self.a2, self.a3])

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "phi1": self.phi1,
            "phi2": self.phi2,
            "shift_mode": self.shift_mode,
            "a1": self.a1,
            "a2": self.a2,
            "a3": self.a3,
            "X_fluctuations": self.X_fluctuations,
            "X_total_dev": self.X_total_dev,
            "X_dev": self.X_total_dev,
            "u_time_mean": self.u_time_mean,
            "a3_scale_applied": float(self.a3_scale_applied),
            "U": self.U,
            "S": self.S,
            "Vt": self.Vt,
            "Z_sindy": self.Z_sindy,
        }


def load_flow_snapshot(
    path: PathLike,
    *,
    delimiter: Optional[str] = None,
    skiprows: int = 0,
    dtype: np.dtype = np.float64,
    reshape: Optional[Tuple[int, int]] = None,
    binary_order: str = "C",
    npz_key: Optional[str] = None,
    mat_key: Optional[str] = None,
) -> np.ndarray:
    """
    Load a velocity/pressure snapshot field from disk.

    Tries, in order:

    - ``.npy`` — :func:`numpy.load`
    - ``.npz`` — first array or ``npz_key``
    - ``.mat`` — :func:`scipy.io.loadmat` (needs SciPy); real ndarray under ``mat_key`` or first large 2D array
    - Otherwise — whitespace-delimited text via :func:`numpy.loadtxt`; if that fails,
      raw binary via :func:`numpy.fromfile` (requires ``reshape=(n_grid, n_time)``)

    Parameters
    ----------
    path
        e.g. ``cy10.snapshot`` or ``cyl0.snapshot``.
    delimiter, skiprows
        Passed to :func:`numpy.loadtxt` for text files.
    dtype
        For raw binary fallback.
    reshape
        ``(n_grid_points, n_timesteps)`` required when using raw binary without header.
    binary_order
        ``"C"`` or ``"F"`` for :meth:`ndarray.reshape` of raw binary.
    npz_key
        Array name inside ``.npz``.
    mat_key
        Variable name inside ``.mat`` (e.g. ``"u"``). If omitted, picks the largest 2D real array.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suf = path.suffix.lower()

    if suf == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=float)

    if suf == ".npz":
        z = np.load(path)
        if npz_key is not None:
            return np.asarray(z[npz_key], dtype=float)
        keys = [k for k in z.files if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No arrays in {path}")
        return np.asarray(z[keys[0]], dtype=float)

    if suf == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as e:
            raise ImportError("Reading .mat requires scipy") from e
        md = loadmat(path, squeeze_me=True, struct_as_record=False)
        if mat_key is not None:
            return np.asarray(md[mat_key], dtype=float)
        best = None
        best_size = -1
        for k, v in md.items():
            if k.startswith("__"):
                continue
            if not isinstance(v, np.ndarray):
                continue
            if np.iscomplexobj(v):
                v = np.real(v)
            if v.ndim >= 2 and v.size > best_size:
                best = np.asarray(v, dtype=float)
                best_size = v.size
        if best is None:
            raise ValueError(f"No suitable ndarray found in {path}")
        return best

    try:
        arr = np.loadtxt(path, dtype=float, delimiter=delimiter, skiprows=skiprows)
        return np.asarray(arr, dtype=float)
    except Exception:
        pass

    if reshape is None:
        raise ValueError(
            f"Could not parse {path} as text; provide reshape=(n_grid, n_timesteps) "
            "for raw binary, or convert to .npy / whitespace text."
        )
    raw = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(reshape))
    if raw.size != expected:
        raise ValueError(
            f"File has {raw.size} values but reshape {reshape} needs {expected}"
        )
    return raw.reshape(reshape, order=binary_order)


def load_base_flow(path: PathLike, **kwargs: Any) -> np.ndarray:
    """Alias for :func:`load_flow_snapshot` (steady-state / base field ``u_s``)."""
    u = load_flow_snapshot(path, **kwargs)
    u = np.asarray(u, dtype=float).reshape(-1)
    return u


def ensure_grid_by_time(
    X: np.ndarray,
    *,
    time_axis: int = 1,
) -> np.ndarray:
    """
    Return ``X`` with shape ``(n_grid, n_timesteps)``.

    ``time_axis=1`` means each column is a time snapshot (default).
    ``time_axis=0`` means each row is a time snapshot (transposes).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if time_axis == 1:
        return X
    if time_axis == 0:
        return X.T
    raise ValueError("time_axis must be 0 or 1")


def pod_shift_coefficients(
    X: np.ndarray,
    u_base: np.ndarray,
    *,
    time_axis: int = 1,
    n_pod_modes: int = 2,
    a3_scale: float = 1.0,
    calibrate_a3_scale: bool = False,
) -> PODShiftResult:
    """
    POD from **time fluctuations**; amplitudes from **full deviation** ``X - u_base``.

    Workflow:

    - ``u_time_mean = mean_t(X)``, ``X_fluctuations = X - u_time_mean`` (per column).
    - ``U, S, Vt = svd(X_fluctuations)`` → ``phi1``, ``phi2`` (shedding, without mean-shift energy
      in the leading subspace of ``X - u_base``).
    - ``delta_u = u_time_mean - u_base``, orthogonalized to ``phi1``, ``phi2``, L2-normalized.
    - ``X_total_dev = X - u_base``; ``a1 = phi1 @ X_total_dev``, ``a2 = phi2 @ X_total_dev``,
      ``a3_raw = delta_u @ X_total_dev``.

    Scaling: ``a3 = a3_raw * a3_scale``, optionally multiplied by an LS estimate of the scalar
    that best matches ``a3 → a1² + a2²`` when ``calibrate_a3_scale=True``.
    """
    if n_pod_modes != 2:
        raise NotImplementedError("Only n_pod_modes=2 is wired for phi1/phi2 fields; extend dataclass if needed.")

    X = ensure_grid_by_time(X, time_axis=time_axis)
    n_grid, n_time = X.shape

    u_base = np.asarray(u_base, dtype=float).reshape(-1)
    if u_base.size != n_grid:
        raise ValueError(
            f"u_base has length {u_base.size}, expected n_grid_points={n_grid} (rows of X)"
        )

    u_time_mean = X.mean(axis=1)
    X_fluctuations = X - u_time_mean[:, np.newaxis]

    U, S, Vt = np.linalg.svd(X_fluctuations, full_matrices=False)

    phi1 = U[:, 0].copy()
    phi2 = U[:, 1].copy()

    delta_u = u_time_mean - u_base
    delta_u = delta_u - phi1 * np.dot(phi1, delta_u)
    delta_u = delta_u - phi2 * np.dot(phi2, delta_u)
    norm = np.linalg.norm(delta_u)
    if norm < 1e-30:
        raise ValueError("Shift mode norm ~ 0 after orthogonalization; check u_base and X")
    delta_u = delta_u / norm

    X_total_dev = X - u_base[:, np.newaxis]

    a1 = phi1 @ X_total_dev
    a2 = phi2 @ X_total_dev
    a3_raw = delta_u @ X_total_dev

    scale = float(a3_scale)
    if calibrate_a3_scale:
        scale *= estimate_a3_scale_lstsq(a1, a2, a3_raw)
    a3 = a3_raw * scale

    return PODShiftResult(
        phi1=phi1,
        phi2=phi2,
        shift_mode=delta_u,
        a1=a1,
        a2=a2,
        a3=a3,
        X_fluctuations=X_fluctuations,
        X_total_dev=X_total_dev,
        u_time_mean=u_time_mean,
        U=U,
        S=S,
        Vt=Vt,
        a3_scale_applied=scale,
    )


def process_cylinder_snapshots(
    transient_path: PathLike,
    base_flow_path: PathLike,
    *,
    transient_kw: Optional[Dict[str, Any]] = None,
    base_kw: Optional[Dict[str, Any]] = None,
    time_axis: int = 1,
    a3_scale: float = 1.0,
    calibrate_a3_scale: bool = False,
) -> PODShiftResult:
    """
    Load ``cy10``-style transient data and ``cyl0``-style base flow, then run :func:`pod_shift_coefficients`.

    Set ``calibrate_a3_scale=True`` to rescale ``a3`` toward ``a1² + a2²`` (see
    :func:`estimate_a3_scale_lstsq`) before optional manual ``a3_scale``.

    Example
    -------
    >>> res = process_cylinder_snapshots("cy10.snapshot", "cyl0.snapshot", time_axis=1)
    >>> Z = res.Z_sindy   # (n_timesteps, 3)
    """
    transient_kw = transient_kw or {}
    base_kw = base_kw or {}
    X = load_flow_snapshot(transient_path, **transient_kw)
    u_s = load_base_flow(base_flow_path, **base_kw)
    return pod_shift_coefficients(
        X,
        u_s,
        time_axis=time_axis,
        a3_scale=a3_scale,
        calibrate_a3_scale=calibrate_a3_scale,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="POD + shift mode from cylinder snapshots → SINDy coordinates")
    p.add_argument("transient", type=str, help="Transient snapshots (e.g. cy10.snapshot)")
    p.add_argument("base", type=str, help="Base / steady flow (e.g. cyl0.snapshot)")
    p.add_argument(
        "--time-axis",
        type=int,
        choices=(0, 1),
        default=1,
        help="1: columns are time (n_grid, n_time); 0: rows are time",
    )
    p.add_argument(
        "--reshape",
        type=int,
        nargs=2,
        metavar=("N_GRID", "N_TIME"),
        default=None,
        help="For raw binary: grid size and number of timesteps",
    )
    p.add_argument(
        "--binary-order",
        choices=("C", "F"),
        default="C",
        help="Memory order when reshaping raw binary",
    )
    p.add_argument("--out", type=str, default=None, help="Optional .npz path for Z_sindy and modes")
    p.add_argument(
        "--calibrate-a3",
        action="store_true",
        help="Rescale a3 toward a1^2+a2^2 via least squares (see estimate_a3_scale_lstsq)",
    )
    args = p.parse_args()

    t_kw: Dict[str, Any] = {}
    if args.reshape is not None:
        t_kw["reshape"] = tuple(args.reshape)
        t_kw["binary_order"] = args.binary_order

    res = process_cylinder_snapshots(
        args.transient,
        args.base,
        transient_kw=t_kw,
        time_axis=args.time_axis,
        calibrate_a3_scale=args.calibrate_a3,
    )
    Z = res.Z_sindy
    print("Z_sindy shape (n_timesteps, 3):", Z.shape)
    print("POD singular values S[:4]:", res.S[: min(4, res.S.size)])
    if args.out:
        np.savez(
            args.out,
            Z_sindy=Z,
            a1=res.a1,
            a2=res.a2,
            a3=res.a3,
            phi1=res.phi1,
            phi2=res.phi2,
            shift_mode=res.shift_mode,
            S=res.S,
        )
        print("Wrote", args.out)
