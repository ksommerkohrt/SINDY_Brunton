# sindy/__init__.py
from .library import SINDyLibrary
from .fit import AdaptiveSTLSQ, prefer_parsimony, remove_collinear_features, drop_constant_like_columns
from .pipeline import SINDySystemModel, SINDyRunConfig
from .physics_filters import (
    aircraft_longitudinal_sindy_library_kw,
    aircraft_option2_nf_keep_feature,
    aircraft_option2_nf_sindy_library_kw,
    aircraft_option3_coeff_sindy_library_kw,
)

__all__ = [
    "SINDyLibrary",
    "AdaptiveSTLSQ",
    "prefer_parsimony",
    "remove_collinear_features",
    "SINDySystemModel",
    "SINDyRunConfig",
    "aircraft_longitudinal_sindy_library_kw",
    "aircraft_option2_nf_keep_feature",
    "aircraft_option2_nf_sindy_library_kw",
    "aircraft_option3_coeff_sindy_library_kw",
]