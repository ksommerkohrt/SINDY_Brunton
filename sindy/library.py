from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp


def _identity(x: np.ndarray) -> np.ndarray:
    return x

@dataclass(frozen=True)
class Atom:
    idx: int  # index into state vector
    type: str  # 'lin', 'sin', 'cos', 'inv', 'sat'
    name: str  # human-readable
    func: Callable  # numpy callable mapping column -> column
    sp_expr: sp.Expr  # sympy expression
    input_space: str = "scaled"  # "scaled" or "physical"


class SINDyLibrary:
    """
    Hybrid-feature SINDy library.

    - Linear atoms use Z_scaled by default (so they match your scaled regression space).
    - Trig/inv atoms use Z_phys (so sin(alpha) means sin(alpha_phys), etc.).
    - For backward compatibility, transform(Z) is allowed and will treat Z as both
      Z_scaled and Z_phys (i.e., Z_phys := Z_scaled) which is appropriate for
      older callers that only have one Z matrix.

    After ``transform``, :class:`sindy.pipeline.SINDyRunConfig` can prune Θ via
    ``remove_double_trig_terms`` and ``small_angle_preference`` (see
    ``sindy.pipeline._apply_small_angle_preference``; small-angle linear-vs-sin uses **physical**
    angles from ``Z_phys``, not scaled linear columns).

    Set ``include_constant_term=False`` to omit the leading column of ones (no SINDy intercept).
    """

    def __init__(
        self,
        measured_names: Sequence[str],
        n_hidden: int = 0,
        max_degree: int = 3,
        max_interaction: int = 2,
        custom_budget: Optional[Dict[str, Dict[str, int]]] = None,
        eps: float = 1e-6,
        keep_feature: Optional[Callable[[str], bool]] = None,
        include_constant_term: bool = True,
    ):
        self.measured_names = list(measured_names)
        self.hidden_names = [f"h{i}" for i in range(n_hidden)]
        self.all_names = self.measured_names + self.hidden_names
        self.z_dim = len(self.all_names)

        self.max_degree = int(max_degree)
        self.max_interaction = int(max_interaction)
        self.eps = float(eps)
        self.keep_feature = keep_feature
        self.include_constant_term = bool(include_constant_term)

        self.budgets: Dict[str, Dict[str, int]] = {
            name: {"lin": 1, "trig": 0, "inv": 0, "sat": 0} for name in self.all_names
        }

        if custom_budget:
            for name, budget in custom_budget.items():
                if name in self.budgets:
                    self.budgets[name].update(budget)

        self.sp_symbols = {name: sp.Symbol(name) for name in self.all_names}

        self.active_atoms: List[Atom] = self._build_atoms()
        self.requires_phys: bool = any(a.input_space == "physical" for a in self.active_atoms)

        self.valid_combos: List[Tuple[int, ...]] = self._build_valid_combinations()
        if self.keep_feature is not None:
            self.valid_combos = [
                c for c in self.valid_combos if self.keep_feature(self._combo_feature_name(c))
            ]
        self.feature_names: List[str] = self._build_feature_names()
        self.sp_features: List[sp.Expr] = self._build_sympy_features()
        self.num_terms: int = len(self.feature_names)

    def combo_index_for_library_column(self, library_column_index: int) -> Optional[int]:
        """
        Map a full-library column index (same order as ``feature_names`` / ``transform``) to
        ``valid_combos``. Returns ``None`` for the leading ``"1"`` column when
        ``include_constant_term`` is True.
        """
        j = int(library_column_index)
        if self.include_constant_term:
            if j == 0:
                return None
            return j - 1
        return j

    def _combo_feature_name(self, combo: Tuple[int, ...]) -> str:
        atom_names = [self.active_atoms[i].name for i in combo]
        if len(atom_names) == 1:
            return atom_names[0]
        return "*".join(atom_names)

    def _build_atoms(self) -> List[Atom]:
        atoms: List[Atom] = []
        for i, name in enumerate(self.all_names):
            b = self.budgets[name]
            sym = self.sp_symbols[name]

            if b.get("lin", 0) > 0:
                atoms.append(Atom(i, "lin", name, _identity, sym, input_space="scaled"))

            if b.get("trig", 0) > 0:
                atoms.append(Atom(i, "sin", f"sin({name})", np.sin, sp.sin(sym), input_space="physical"))
                atoms.append(Atom(i, "cos", f"cos({name})", np.cos, sp.cos(sym), input_space="physical"))

            if b.get("inv", 0) > 0:
                atoms.append(
                    Atom(
                        i,
                        "inv",
                        f"1/({name})",
                        lambda x, eps=self.eps: 1.0 / (x + eps * np.sign(x + 1e-12)),
                        1 / sym,
                        input_space="physical",
                    )
                )
            if b.get("sat", 0) > 0:
                atoms.append(
                    Atom(
                        i,
                        "sat",
                        f"tanh({name})",
                        np.tanh,
                        sp.tanh(sym),
                        input_space="physical",
                    )
                )
        return atoms

    def _is_valid_combination(self, combo: Tuple[int, ...]) -> bool:
        # 1. Enforce global degree ceiling (Total length of the tuple)
        if len(combo) > self.max_degree:
            return False

        # 2. Enforce variable interaction ceiling (Unique states involved)
        unique_states = {self.active_atoms[i].idx for i in combo}
        if len(unique_states) > self.max_interaction:
            return False

        # 3. FIXED: Interaction Degree Guard (Blocks V*V*alpha if max_interaction=2)
        if len(unique_states) > 1 and len(combo) > self.max_interaction:
            return False

        # 4. Prevent mixing different atom types for same state (V * sin(V))
        for sidx in unique_states:
            types_for_state = [
                self.active_atoms[i].type for i in combo if self.active_atoms[i].idx == sidx
            ]
            if len(set(types_for_state)) > 1:
                return False

        # 5. Enforce per-state/type budget limits
        counts = Counter()
        for atom_idx in combo:
            atom = self.active_atoms[atom_idx]
            sname = self.all_names[atom.idx]
            type_key = "trig" if atom.type in ("sin", "cos") else atom.type
            counts[(atom.idx, type_key)] += 1
            if counts[(atom.idx, type_key)] > self.budgets[sname].get(type_key, 0):
                return False

        return True
    def _build_valid_combinations(self) -> List[Tuple[int, ...]]:
        valid: List[Tuple[int, ...]] = []
        n_atoms = len(self.active_atoms)
        for degree in range(1, self.max_degree + 1):
            for combo in combinations_with_replacement(range(n_atoms), degree):
                if self._is_valid_combination(combo):
                    valid.append(combo)
        return valid

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        if self.include_constant_term:
            names.append("1")
        for combo in self.valid_combos:
            atom_names = [self.active_atoms[i].name for i in combo]
            if len(atom_names) == 1:
                names.append(atom_names[0])
            else:
                names.append("*".join(atom_names))
        return names

    def _build_sympy_features(self) -> List[sp.Expr]:
        feats: List[sp.Expr] = []
        if self.include_constant_term:
            feats.append(sp.Integer(1))
        for combo in self.valid_combos:
            expr = sp.Integer(1)
            for atom_idx in combo:
                expr *= self.active_atoms[atom_idx].sp_expr
            feats.append(expr)
        return feats

    def transform(self, Z_scaled: np.ndarray, Z_phys: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Backward compatible behavior:
        - If called as transform(Z), we assume caller wants Z used as both scaled and physical.
          This prevents older code (e.g., diagnostics) from breaking immediately.
        - If you actually want hybrid semantics, call transform(Z_scaled=..., Z_phys=...).
        """
        Z_scaled = np.asarray(Z_scaled, dtype=float)
        if Z_scaled.ndim != 2:
            raise ValueError(f"Z_scaled must be 2D, got shape {Z_scaled.shape}")
        if Z_scaled.shape[1] != self.z_dim:
            raise ValueError(f"Z_scaled has {Z_scaled.shape[1]} cols, expected {self.z_dim}")

        if Z_phys is None:
            Z_phys = Z_scaled
        else:
            Z_phys = np.asarray(Z_phys, dtype=float)
            if Z_phys.shape != Z_scaled.shape:
                raise ValueError(f"Z_phys shape {Z_phys.shape} must match Z_scaled {Z_scaled.shape}")

        if self.requires_phys and Z_phys is None:
            raise ValueError("Z_phys is required because the library has physical-space atoms")

        n_samples = Z_scaled.shape[0]
        Theta = np.empty((n_samples, self.num_terms), dtype=float)

        atom_vals: List[np.ndarray] = []
        for atom in self.active_atoms:
            src = Z_scaled if atom.input_space == "scaled" else Z_phys
            col = src[:, atom.idx]
            if atom.type == "lin":
                atom_vals.append(col)
            elif atom.type == "sin":
                atom_vals.append(np.sin(col))
            elif atom.type == "cos":
                atom_vals.append(np.cos(col))
            elif atom.type == "inv":
                atom_vals.append(1.0 / (col + self.eps * np.sign(col + 1e-12)))
            elif atom.type == "sat":
                atom_vals.append(np.tanh(col))
            else:
                raise ValueError(f"Unknown atom type: {atom.type}")

        col_idx = 0
        if self.include_constant_term:
            Theta[:, col_idx] = 1.0
            col_idx += 1
        for combo in self.valid_combos:
            col = np.ones(n_samples, dtype=float)
            for atom_idx in combo:
                col *= atom_vals[atom_idx]
            Theta[:, col_idx] = col
            col_idx += 1

        return Theta

    def get_equations(
        self,
        coefficients: np.ndarray,
        target_names: Optional[Sequence[str]] = None,
        zero_tol: float = 1e-10,
    ) -> Dict[str, sp.Expr]:
        coef = np.asarray(coefficients, dtype=float)
        if coef.ndim != 2:
            raise ValueError("coefficients must be 2D (n_features, n_targets)")
        if coef.shape[0] != self.num_terms:
            raise ValueError(f"coefficients has {coef.shape[0]} rows, expected {self.num_terms}")

        if target_names is None:
            target_names = [f"dx{i}" for i in range(coef.shape[1])]
        if len(target_names) != coef.shape[1]:
            raise ValueError("target_names length must match n_targets")

        eqs: Dict[str, sp.Expr] = {}
        for j, name in enumerate(target_names):
            expr = sp.Integer(0)
            for i, c in enumerate(coef[:, j]):
                if abs(float(c)) > zero_tol:
                    expr += float(c) * self.sp_features[i]
            eqs[name] = sp.nsimplify(expr, rational=False)
        return eqs



