"""
Feature-name predicates for :class:`sindy.library.SINDyLibrary` (classic explicit SINDy only).

These are **not** used by SINDy-PI / :class:`idtools.sindy_pi.parallel_library.ParallelLibrary`.
Pass via ``SINDyLibrary(..., keep_feature=...)`` or ``SINDyRunConfig.library_keep_feature``.

**Longitudinal ``aircraftsim`` (Option 2 vs Option 3)** use **different** RHS libraries; see
:func:`aircraft_option2_nf_sindy_library_kw` and :func:`aircraft_option3_coeff_sindy_library_kw`.

Global Θ pruning (double-trig, small-angle) is configured on :class:`sindy.pipeline.SINDyRunConfig`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

# Option 2 NF library regressor set (must match ``OPTION2_NF_FEATURE_NAMES`` in aircraftsim).
_OPTION2_NF_VARS = frozenset({"q_dyn", "alpha", "Q", "elv_act", "T_act"})


def aircraftsim_physics_informed_keep_feature(
    name: str,
    *,
    forbid_v_in_products: bool = True,
    forbid_raw_t_act: bool = True,
    t_act_requires_alpha_trig: bool = True,
) -> bool:
    """
    Curated feature filter for simplified ``aircraftsim`` trajectories (SINDy targets ``V, alpha, gamma, Q``).

    Smooth flight data makes **linear speed proxies** (e.g. ``V*alpha``) statistically hard to
    distinguish from **aerodynamic forms** (``q_dyn*alpha``, ``1/(V)*...``). This filter **never
    generates** those cheap proxies in :class:`sindy.library.SINDyLibrary` so STLSQ is not tempted.

    Global rules (one Θ shared by all targets):

    - **Speed scaling:** allow the linear monomial ``V`` alone. **Reject** any term whose factors
      include ``V`` together with any other factor (use ``q_dyn`` for ``q̄·state`` crosses and
      ``1/(V)`` for inverse-speed / kinematic chains instead of ``V*state``).
    - **Thrust:** **reject** standalone ``T_act``. Any term containing ``T_act`` must also contain
      ``sin(alpha)`` or ``cos(alpha)`` so thrust enters as **body-resolved** components.

    Pair with ``prefer_parsimony=False`` and sufficient ridge (see ``sindy.pipeline`` defaults when
    collinearity is off).
    """
    name = str(name).strip()
    if name == "1":
        return True
    factors = name.split("*") if "*" in name else [name]

    if forbid_raw_t_act and name == "T_act":
        return False

    if forbid_v_in_products and len(factors) >= 2 and "V" in factors:
        return False

    if t_act_requires_alpha_trig and "T_act" in factors:
        if not any(f in ("sin(alpha)", "cos(alpha)") for f in factors):
            return False

    return True


def aircraft_option2_nf_keep_feature(name: str) -> bool:
    """
    Physics filter for **Option 2** (kinematic LHS, normalized forces / raw thrust on RHS).

    **Dimensional / PIML — kinematics cannot create specific force.** Regressors
    ``alpha``, ``Q``, ``elv_act`` are geometries or rates; alone they carry no fluid energy.
    Every monomial must include **at least one** of ``q_dyn`` (dynamic pressure proxy) or
    ``T_act`` (thrust), so the library cannot express e.g. ``NFx ~ alpha`` or ``NMy ~ elv_act``
    without an energy carrier. **No constant / bias term** in Θ (``"1"``): Newton–Euler specific
    force has no intercept that would imply motion from rest (“levitation”).

    **``sin(alpha)`` / ``cos(alpha)``** are allowed (see :func:`option2_nf_sindy_budget`); any
    other ``sin``/``cos`` is rejected. **No** ``1/(V)``, **no** ``q_dyn*q_dyn``, and **no** more
    than **two distinct** Option-2 regressors per monomial (e.g. reject ``T_act*alpha*Q``).

    .. note::

        :class:`~sindy.library.SINDyLibrary` uses ``max_interaction=3`` so terms like
        ``q_dyn*alpha*alpha`` can form; the **two-distinct** cap is enforced here.
    """
    name = str(name).strip()
    if name == "1":
        return False
    if "1/(V)" in name:
        return False

    factors = name.split("*") if "*" in name else [name]
    for f in factors:
        fs = str(f).strip()
        if "sin(" in fs or "cos(" in fs:
            if fs not in ("sin(alpha)", "cos(alpha)"):
                return False

    if sum(1 for f in factors if f == "q_dyn") >= 2:
        return False

    # PIML: specific force / moment regression must involve fluid energy (q_dyn) and/or thrust.
    if not any(f in ("q_dyn", "T_act") for f in factors):
        return False

    bases: List[str] = []
    for f in factors:
        fs = str(f).strip()
        if fs in ("sin(alpha)", "cos(alpha)"):
            bases.append("alpha")
        elif fs in _OPTION2_NF_VARS:
            bases.append(fs)
    if len(set(bases)) > 2:
        return False

    return True


def aircraft_option2_nf_sindy_library_kw() -> Dict[str, Any]:
    """
    :class:`~sindy.pipeline.SINDyRunConfig` kwargs for **Option 2** NF SINDy
    (``OPTION2_NF_FEATURE_NAMES`` + :func:`aircraft_option2_nf_keep_feature`).
    """
    return {
        "library_keep_feature": aircraft_option2_nf_keep_feature,
        "library_include_constant": False,
        "max_degree": 3,
        "max_interaction": 3,
        "remove_double_trig_terms": True,
        "small_angle_preference": False,
    }


def aircraft_option3_coeff_sindy_library_kw() -> Dict[str, Any]:
    """
    :class:`~sindy.pipeline.SINDyRunConfig` kwargs for **Option 3** coefficient SINDy
    (``OPTION3_COEFF_FEATURE_NAMES`` only — no custom ``keep_feature``).
    """
    return {
        "library_keep_feature": aircraft_option3_coeff_keep_feature,
        "max_degree": 2,
        "max_interaction": 2,
        "remove_double_trig_terms": True,
        "small_angle_preference": False,
    }


def aircraft_option3_coeff_keep_feature(name: str) -> bool:
    """
    Physics filter for **Option 3** coefficient regression.

    Rules applied:

    - **Pure rate constraint:** if ``Q`` appears, the term must be exactly ``Q``.
      Reject ``Q*alpha``, ``Q*elv_act``, ``Q*Q``, etc.
    - **Pure control constraint:** if ``elv_act`` appears, the term must be exactly ``elv_act``.
      Reject ``alpha*elv_act``, ``Q*elv_act``, ``elv_act*elv_act``, etc.

    Other terms (e.g. ``1``, ``alpha``, ``alpha*alpha``) are allowed and controlled by
    Option-3 feature set + polynomial budgets.
    """
    name = str(name).strip()
    if name == "1":
        return True

    factors = name.split("*") if "*" in name else [name]

    if "Q" in factors:
        return name == "Q"
    if "elv_act" in factors:
        return name == "elv_act"
    return True


def aircraft_longitudinal_sindy_library_kw(**_: Any) -> Dict[str, Any]:
    """
    Deprecated: use :func:`aircraft_option2_nf_sindy_library_kw` or
    :func:`aircraft_option3_coeff_sindy_library_kw`. Alias of Option 2 NF kwargs.
    """
    return aircraft_option2_nf_sindy_library_kw()


# --- 6-DOF implicit residual library (q_dyn_S * states + standalone controls) ---

_EVEN_BETA_AERO_FEATURES: Set[str] = frozenset(
    {"q_dyn_S", "q_dyn_S_alpha", "q_dyn_S_Q", "q_dyn_S_P", "q_dyn_S_R"}
)


def aircraft6dof_residual_keep_feature_even_beta(name: str) -> bool:
    """
    For targets that are **even** in sideslip (typical: ``Fx``, ``Fz``, ``My`` body residuals).

    Rejects explicit ``beta`` linear coupling via the ``q_dyn_S_beta`` atom.
    """
    name = str(name).strip()
    if name == "1":
        return True
    factors = name.split("*") if "*" in name else [name]
    if "q_dyn_S_beta" in factors:
        return False
    return True


def aircraft6dof_residual_keep_feature_odd_beta(name: str) -> bool:
    """
    For targets that are **odd** in sideslip (typical: ``Fy``, ``Mx``, ``Mz``).

    Rejects "pure even-in-beta" aerodynamic atoms ``q_dyn_S``, ``q_dyn_S_alpha``, ``q_dyn_S_Q`` alone;
    keeps ``q_dyn_S_beta``, ``q_dyn_S_P``, ``q_dyn_S_R``, and all propulsion/surface channels.

    With ``max_degree=1`` each term is a single atom, so this enforces a minimal parity split.
    """
    name = str(name).strip()
    if name == "1":
        return True
    factors = name.split("*") if "*" in name else [name]
    if len(factors) != 1:
        return True
    atom = factors[0]
    if atom in _EVEN_BETA_AERO_FEATURES:
        return False
    return True


def aircraft6dof_residual_even_beta_library_kw() -> Dict[str, Any]:
    """Attach :func:`aircraft6dof_residual_keep_feature_even_beta` to :class:`~sindy.pipeline.SINDyRunConfig`."""
    return {"library_keep_feature": aircraft6dof_residual_keep_feature_even_beta}


def aircraft6dof_residual_odd_beta_library_kw() -> Dict[str, Any]:
    """Attach :func:`aircraft6dof_residual_keep_feature_odd_beta` to :class:`~sindy.pipeline.SINDyRunConfig`."""
    return {"library_keep_feature": aircraft6dof_residual_keep_feature_odd_beta}


# --- 6-DOF Option 2 (mass-normalized / kinematic rates) — extend longitudinal PIML idea ---

_OPTION6_NF_VARS = frozenset(
    {
        "q_dyn",
        "alpha",
        "beta",
        "P",
        "Q",
        "R",
        "elv_act",
        "ail_act",
        "rud_act",
        "T_act",
    }
)


def aircraft6dof_option2_nf_keep_feature(name: str) -> bool:
    """
    PIML filter for **6-DOF Option 2**: every non-constant monomial includes **``q_dyn``** or **``T_act``**.

    Same spirit as :func:`aircraft_option2_nf_keep_feature` but allows lateral states
    (``beta``, ``P``, ``R``, aileron, rudder). **No** library bias term (``"1"``).
    """
    name = str(name).strip()
    if name == "1":
        return False
    if "1/(V)" in name:
        return False
    factors = name.split("*") if "*" in name else [name]
    for f in factors:
        fs = str(f).strip()
        if "sin(" in fs or "cos(" in fs:
            if fs not in ("sin(alpha)", "cos(alpha)"):
                return False
    if sum(1 for f in factors if f == "q_dyn") >= 2:
        return False
    if not any(f in ("q_dyn", "T_act") for f in factors):
        return False
    bases: List[str] = []
    for f in factors:
        fs = str(f).strip()
        if fs in ("sin(alpha)", "cos(alpha)"):
            bases.append("alpha")
        elif fs in _OPTION6_NF_VARS:
            bases.append(fs)
    if len(set(bases)) > 2:
        return False
    return True


def aircraft6dof_option2_nf_sindy_library_kw() -> Dict[str, Any]:
    """Kwargs for :class:`~sindy.pipeline.SINDyRunConfig` (6-DOF Option 2 NF library)."""
    return {"library_keep_feature": aircraft6dof_option2_nf_keep_feature}


def aircraft6dof_option3_coeff_keep_feature(name: str) -> bool:
    """
    Like :func:`aircraft_option3_coeff_keep_feature` but for full **6-DOF** coefficient regressors.

    Enforces **pure** body-rate and surface factors (no ``Q*alpha``, ``ail_act*beta``, etc.).
    """
    name = str(name).strip()
    if name == "1":
        return True
    factors = name.split("*") if "*" in name else [name]
    for pure in ("Q", "P", "R", "elv_act", "ail_act", "rud_act"):
        if pure in factors:
            return name == pure
    return True


def aircraft6dof_option3_coeff_sindy_library_kw() -> Dict[str, Any]:
    """Option 3 style for 6-DOF coefficient discovery (polynomial degree/interaction caps)."""
    return {
        "library_keep_feature": aircraft6dof_option3_coeff_keep_feature,
        "max_degree": 2,
        "max_interaction": 2,
        "remove_double_trig_terms": True,
        "small_angle_preference": False,
    }
