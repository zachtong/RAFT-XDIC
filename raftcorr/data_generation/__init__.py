"""Synthetic DIC training-data generation (speckle, warp, displacement fields, builder)."""

from .speckle import SpeckleParams, generate_speckle_pattern, get_random_speckle_params
from .warp import inverse_warp_gray
from .displacement_fields import make_displacement_field, register, REGISTRY
from .build_dataset import build_dataset, generate_pair

__all__ = [
    "SpeckleParams",
    "generate_speckle_pattern",
    "get_random_speckle_params",
    "inverse_warp_gray",
    "make_displacement_field",
    "register",
    "REGISTRY",
    "build_dataset",
    "generate_pair",
]
