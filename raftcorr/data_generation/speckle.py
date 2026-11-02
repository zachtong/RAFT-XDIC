"""Ellipse-based synthetic speckle pattern generation.

Ported from the legacy MATLAB ``generateSpecklePattern.m`` (ellipse path). The Gaussian
speckle path and the old positional-argument signature of the MATLAB version are dropped;
only the ellipse generator actually used by the dataset driver is kept.

All randomness flows through an explicit ``numpy.random.Generator`` so that generation is
fully reproducible from a seed (the legacy MATLAB code had no seeding).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

# MATLAB imgaussfilt uses a filter size of roughly +/-2 sigma; mirror that here.
_GAUSS_TRUNCATE = 2.0


@dataclass
class SpeckleParams:
    """Per-image speckle parameters (one draw of ``get_random_speckle_params``)."""

    diameter: float
    density: float
    variation: float = 0.3
    eccentricity: float = 0.0
    background: str = "white"  # 'white' (dark dots) or 'black' (bright dots)
    speckle_intensity_range: Sequence[float] = (0.0, 80.0)
    background_intensity_range: Sequence[float] = (200.0, 255.0)
    blur_sigma_range: Sequence[float] | None = None  # (min, max); a sigma is drawn within
    noise_std: float = 0.0

    @property
    def spacing(self) -> float:
        return self.diameter / max(self.density, 1e-12)


def generate_speckle_pattern(
    width: int,
    height: int,
    params: SpeckleParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Generate one ``height x width`` uint8 grayscale speckle image.

    Returns ``(img_uint8, info)`` where ``info`` records the realized random draws
    (intensities, blur sigma) for logging.
    """
    D = float(params.diameter)
    spacing = params.spacing
    rows = int(np.floor(height / spacing))
    cols = int(np.floor(width / spacing))
    xmin = 0.5 * (width - cols * spacing)
    ymin = 0.5 * (height - rows * spacing)

    # Regular grid of speckle centers (rows+1 x cols+1), then jitter.
    cc, rr = np.meshgrid(np.arange(cols + 1), np.arange(rows + 1))
    x = xmin + cc * spacing
    y = ymin + rr * spacing
    lim = 0.5 * params.variation * spacing
    x = x + lim * (2.0 * rng.random(x.shape) - 1.0)
    y = y + lim * (2.0 * rng.random(y.shape) - 1.0)

    # Ellipse axes from eccentricity (0 => circle).
    r0 = 0.5 * D
    e = float(np.clip(params.eccentricity, 0.0, 0.9999))
    maj = r0 / np.sqrt(np.sqrt(1.0 - e * e))
    minr = r0 * np.sqrt(np.sqrt(1.0 - e * e))

    if params.background == "black":
        img = np.zeros((height, width), dtype=np.float64)
        fg = 255.0
    else:
        img = np.full((height, width), 255.0, dtype=np.float64)
        fg = 0.0

    xs_flat = x.ravel()
    ys_flat = y.ravel()
    for cx, cy in zip(xs_flat, ys_flat):
        if cx < -maj or cx > width + maj or cy < -maj or cy > height + maj:
            continue
        x0 = max(0, int(np.floor(cx - maj)))
        x1 = min(width - 1, int(np.ceil(cx + maj)))
        y0 = max(0, int(np.floor(cy - minr)))
        y1 = min(height - 1, int(np.ceil(cy + minr)))
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        mask = ((gx - cx) ** 2) / (maj * maj) + ((gy - cy) ** 2) / (minr * minr) <= 1.0
        sub = img[y0:y1 + 1, x0:x1 + 1]
        sub[mask] = fg

    # Remap binary foreground/background to sampled intensities.
    s_min, s_max = min(params.speckle_intensity_range), max(params.speckle_intensity_range)
    b_min, b_max = min(params.background_intensity_range), max(params.background_intensity_range)
    si = s_min + (s_max - s_min) * rng.random()
    bi = b_min + (b_max - b_min) * rng.random()
    if abs(si - bi) < 1e-6:
        if params.background == "black":
            bi = max(0.0, si - 1.0)
        else:
            si = max(0.0, bi - 1.0)

    img_norm = img / 255.0
    speckle_mask = img_norm if params.background == "black" else (1.0 - img_norm)
    img = speckle_mask * si + (1.0 - speckle_mask) * bi

    blur_sigma = np.nan
    if params.blur_sigma_range is not None:
        smin, smax = params.blur_sigma_range
        blur_sigma = (smax - smin) * rng.random() + smin
        if blur_sigma > 0:
            img = gaussian_filter(img, sigma=blur_sigma, truncate=_GAUSS_TRUNCATE)

    if params.noise_std and params.noise_std > 0:
        img = img + params.noise_std * rng.standard_normal(img.shape)

    img = np.clip(img, 0.0, 255.0)
    img_uint8 = img.astype(np.uint8)

    info = {
        "diameter": params.diameter,
        "density": params.density,
        "spacing": spacing,
        "variation": params.variation,
        "eccentricity": params.eccentricity,
        "background": params.background,
        "speckle_intensity": si,
        "background_intensity": bi,
        "blur_sigma": blur_sigma,
        "noise_std": params.noise_std,
    }
    return img_uint8, info


def get_random_speckle_params(speckle_cfg: dict, rng: np.random.Generator) -> SpeckleParams:
    """Draw per-image speckle parameters from a config dict (see configs/dataset_default.yaml).

    Port of the legacy ``getRandomParameters`` (speckle portion). The noise std is drawn
    separately by the dataset driver, since noise is applied after warping.
    """
    d_lo, d_hi = speckle_cfg["diameter_range"]
    diameter = d_lo + (d_hi - d_lo) * rng.random()

    den_lo, den_hi = speckle_cfg["density_range"]
    density = den_lo + (den_hi - den_lo) * rng.random()

    ecc_max = speckle_cfg.get("eccentricity_max", 0.4)
    eccentricity = min((rng.random() ** 2) * ecc_max, ecc_max)

    if rng.random() < speckle_cfg.get("invert_prob", 0.2):
        background = "black"
        sub = speckle_cfg["black_bg"]
    else:
        background = "white"
        sub = speckle_cfg["white_bg"]

    center_lo, center_hi = speckle_cfg["blur_sigma_center_range"]
    band = speckle_cfg.get("blur_sigma_band", 0.05)
    sigma_center = center_lo + (center_hi - center_lo) * rng.random()
    blur_range = (max(0.0, sigma_center - band), sigma_center + band)

    return SpeckleParams(
        diameter=diameter,
        density=density,
        variation=speckle_cfg.get("variation", 0.3),
        eccentricity=eccentricity,
        background=background,
        speckle_intensity_range=tuple(sub["speckle_intensity"]),
        background_intensity_range=tuple(sub["background_intensity"]),
        blur_sigma_range=blur_range,
        noise_std=0.0,
    )
