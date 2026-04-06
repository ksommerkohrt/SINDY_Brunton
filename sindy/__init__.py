# sindy/__init__.py — minimal Brunton/Lorenz demo (no aircraft physics_filters re-exports)
from .library import SINDyLibrary
from .fit import AdaptiveSTLSQ, prefer_parsimony, remove_collinear_features, drop_constant_like_columns
from .pipeline import SINDySystemModel, SINDyRunConfig

__all__ = [
    "SINDyLibrary",
    "AdaptiveSTLSQ",
    "prefer_parsimony",
    "remove_collinear_features",
    "SINDySystemModel",
    "SINDyRunConfig",
]
