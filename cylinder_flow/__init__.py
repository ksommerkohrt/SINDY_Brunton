"""Cylinder wake reduced-order dynamics (PNAS SINDy paper, Eq. 8) and validation helpers."""

from .mean_field_simulator import (
    CylinderMeanFieldParams,
    DEFAULT_ILLUSTRATIVE_PARAMS,
    cylinder_mean_field_rhs,
    simulate_cylinder_mean_field,
)
from .validation import (
    compare_to_mean_field_reference,
    trajectory_validation_metrics,
)
from .snapshot_pod_shift import (
    PODShiftResult,
    ensure_grid_by_time,
    estimate_a3_scale_lstsq,
    load_base_flow,
    load_flow_snapshot,
    pod_shift_coefficients,
    process_cylinder_snapshots,
)

__all__ = [
    "CylinderMeanFieldParams",
    "DEFAULT_ILLUSTRATIVE_PARAMS",
    "cylinder_mean_field_rhs",
    "simulate_cylinder_mean_field",
    "trajectory_validation_metrics",
    "compare_to_mean_field_reference",
    "PODShiftResult",
    "ensure_grid_by_time",
    "load_base_flow",
    "load_flow_snapshot",
    "pod_shift_coefficients",
    "process_cylinder_snapshots",
    "estimate_a3_scale_lstsq",
]
