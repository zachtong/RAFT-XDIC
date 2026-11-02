"""Inverse (backward) warping to synthesize a deformed image from a reference image.

Ported from the legacy MATLAB ``inverse_warp_gray.m``.

Convention (see CONVENTION.md): given the reference image ``f`` and the reference->deformed
displacement field ``(u, v)`` with ``+u`` rightward and ``+v`` downward, the deformed image
``g`` satisfies

    g[y, x] = f[y - v(y, x), x - u(y, x)]

i.e. for each target (deformed) pixel we sample the source (reference) location it came from.
This guarantees full coverage without holes. Interpolation is cubic spline; out-of-bounds
samples are clamped to the border (nearest / replicate), matching the MATLAB 'nearest'
extrapolation.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline


def inverse_warp_gray(f: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the deformed grayscale image. ``f`` is uint8 (or float); ``u``/``v`` are float.

    The output keeps the dtype of ``f`` (uint8 is rounded and clipped to [0, 255]).
    """
    if f.ndim != 2:
        raise ValueError("f must be a 2D grayscale image.")
    H, W = f.shape
    if u.shape != (H, W) or v.shape != (H, W):
        raise ValueError("u and v must match the shape of f.")

    f_in_dtype = f.dtype
    fD = f.astype(np.float64)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs = xx - u  # source columns
    ys = yy - v  # source rows

    # Border replicate: clamp source coords into range (mirrors MATLAB 'nearest' extrap).
    xs = np.clip(xs, 0.0, W - 1.0)
    ys = np.clip(ys, 0.0, H - 1.0)

    # Cubic spline interpolant over the regular reference grid.
    spline = RectBivariateSpline(np.arange(H), np.arange(W), fD, kx=3, ky=3)
    gD = spline.ev(ys.ravel(), xs.ravel()).reshape(H, W)

    if np.issubdtype(f_in_dtype, np.integer):
        info = np.iinfo(f_in_dtype)
        gD = np.clip(np.round(gD), info.min, info.max)
    return gD.astype(f_in_dtype)
