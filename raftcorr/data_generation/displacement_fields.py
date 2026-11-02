"""Pluggable registry of ground-truth displacement-field generators.

Each generator returns a ``(u, v)`` pair of ``H x W`` float32 arrays in the canonical
convention (``+u`` rightward, ``+v`` downward; the reference->deformed forward field).

Adding a new deformation type for study #2 (effect of deformation-field type) is just a
matter of registering another function with ``@register("name")`` -- no change to the rest
of the pipeline is needed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from glob import glob
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

FieldFn = Callable[..., tuple[np.ndarray, np.ndarray]]

# name -> (function, applies_independent_rotation)
REGISTRY: dict[str, tuple[FieldFn, bool]] = {}


def register(name: str, independent_rotation: bool = False):
    def deco(fn: FieldFn) -> FieldFn:
        REGISTRY[name] = (fn, independent_rotation)
        return fn
    return deco


def _rotate_scalar(field: np.ndarray, angle_deg: int) -> np.ndarray:
    """Lattice-preserving rotation by a multiple of 90 degrees (pure index reorder)."""
    k = (angle_deg // 90) % 4
    return np.rot90(field, k)


def _smooth_scalar(H: int, W: int, rng: np.random.Generator,
                   disp_range: float, blur_sigma_range) -> np.ndarray:
    """One smooth random scalar field, mapped to [-disp_range, disp_range]."""
    s_lo, s_hi = blur_sigma_range
    sigma = s_lo + (s_hi - s_lo) * rng.random()
    raw = rng.random((H, W))
    blurred = gaussian_filter(raw, sigma=sigma, mode="reflect")
    lo, hi = blurred.min(), blurred.max()
    norm = (blurred - lo) / (hi - lo) if hi > lo else np.zeros_like(blurred)
    return (-disp_range + 2.0 * disp_range * norm).astype(np.float32)


@register("smooth_random", independent_rotation=True)
def smooth_random(H, W, rng, disp_range, blur_sigma_range=(3.0, 40.0), **_):
    """Two independent smooth random fields. Self-contained proxy for the paper's
    blurred-natural-image fields (no external images required)."""
    u = _smooth_scalar(H, W, rng, disp_range, blur_sigma_range)
    v = _smooth_scalar(H, W, rng, disp_range, blur_sigma_range)
    return u, v


@register("smooth_random_peak")
def smooth_random_peak(H, W, rng, disp_range, disp_range_lo=0.0, crop_size=None,
                       blur_sigma_range=(3.0, 40.0), **_):
    """``smooth_random`` rescaled so the PEAK displacement magnitude max|(u, v)| inside the
    centered ``crop_size`` region is uniform in ``[disp_range_lo, disp_range]``.

    This matches the amplitude distribution of the crack generators, whose crop always holds
    both sides of the line and therefore a peak of exactly ``magnitude``. It is the crack-free
    counterpart used by the matched control training set. Not rotation-augmented: the two
    components are independent isotropic noise fields, so a 90-degree rotation adds nothing.
    """
    u = _smooth_scalar(H, W, rng, 1.0, blur_sigma_range)
    v = _smooth_scalar(H, W, rng, 1.0, blur_sigma_range)
    c = crop_size or min(H, W)
    y0, x0 = (H - c) // 2, (W - c) // 2
    peak = float(np.hypot(u[y0:y0 + c, x0:x0 + c], v[y0:y0 + c, x0:x0 + c]).max())
    scale = float(rng.uniform(disp_range_lo, disp_range)) / max(peak, 1e-12)
    return (u * scale).astype(np.float32), (v * scale).astype(np.float32)


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@lru_cache(maxsize=8)
def _list_images(image_dir: str) -> tuple:
    """Sorted image files in ``image_dir`` (globbed once per dir, then cached)."""
    return tuple(
        p for p in sorted(glob(os.path.join(image_dir, "*")))
        if p.lower().endswith(_IMG_EXTS)
    )


@register("natural_image", independent_rotation=True)
def natural_image(H, W, rng, disp_range, image_paths=None, image_dir=None,
                  blur_sigma_range=(3.0, 40.0), renormalize=True, **_):
    """u and v derived from two blurred natural images (the paper's smooth-field method).

    Supply either ``image_dir`` (a folder; every image in it is used, globbed once and
    cached) or ``image_paths`` (an explicit list of >=2 files). Two images are chosen at
    random per sample. Falls back to ``smooth_random`` if no images are available.

    ``renormalize`` (default True): min-max stretch each blurred field to [0, 1] before mapping
    to [-disp_range, disp_range] (matching ``smooth_random``). The raw MATLAB v4 method (and the
    earlier port) did NOT do this. Without it, heavy blur (sigma up to 40) bunches the field
    toward the image mean, biasing displacement toward small values -- but on REAL high-contrast
    photos the effect is modest (measured on the actual dataset: median |disp| ~2 px at
    disp_range=6, 28% of pixels <1 px, extremes still reach 4-6 px; renorm only moves the <1 px
    fraction ~26% -> ~21%). Renormalization guarantees the field spans the full +/-disp_range
    regardless of image contrast. Set False to reproduce the legacy behaviour. NOTE: this is a
    minor correction -- models saturate at their training displacement because of ``disp_range``
    itself, not this flag. (An earlier "~1 px median" claim came from a low-contrast synthetic
    proxy image and overstated the collapse.)
    """
    if image_dir and not image_paths:
        image_paths = list(_list_images(image_dir))
    if not image_paths or len(image_paths) < 2:
        return smooth_random(H, W, rng, disp_range, blur_sigma_range)
    from PIL import Image

    def field_from(path):
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        # Resize to H x W via PIL (bicubic), matching the MATLAB imresize step.
        img = np.asarray(
            Image.fromarray((img * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
            dtype=np.float64,
        ) / 255.0
        s_lo, s_hi = blur_sigma_range
        sigma = s_lo + (s_hi - s_lo) * rng.random()
        blurred = gaussian_filter(img, sigma=sigma, mode="reflect")
        if renormalize:
            lo, hi = blurred.min(), blurred.max()
            blurred = (blurred - lo) / (hi - lo) if hi > lo else np.zeros_like(blurred)
        return (-disp_range + 2.0 * disp_range * blurred).clip(-disp_range, disp_range).astype(np.float32)

    idx = rng.permutation(len(image_paths))[:2]
    return field_from(image_paths[idx[0]]), field_from(image_paths[idx[1]])


@register("rigid_translation")
def rigid_translation(H, W, rng, disp_range, u0=None, v0=None, **_):
    """Spatially constant (u0, v0). If unspecified, drawn uniformly in [-range, range].

    Used by the convention round-trip test (prescribe a known +u shift) and the translation
    benchmark.
    """
    if u0 is None:
        u0 = float(rng.uniform(-disp_range, disp_range))
    if v0 is None:
        v0 = float(rng.uniform(-disp_range, disp_range))
    u = np.full((H, W), u0, dtype=np.float32)
    v = np.full((H, W), v0, dtype=np.float32)
    return u, v


@register("mixed_large")
def mixed_large(H, W, rng, disp_range, rigid_frac=0.8, smooth_range=12.0,
                blur_sigma_range=(15.0, 45.0), **_):
    """Large-displacement training field.

    Large displacement (up to +/-``disp_range``, e.g. 200 px) is supplied by **rigid
    translation**, because a smooth blurred-image field mapped to +/-200 px has displacement
    gradient >> 1 everywhere and would fold the warp (no valid ground truth). A minority of
    samples (``1 - rigid_frac``) use a small, folding-safe ``smooth_random`` field so the model
    also sees local (non-constant) deformation instead of only globally-uniform shifts.

    This is the honest way to train a model that tracks large motion: the training distribution
    must actually contain large displacements. ``rigid_frac`` controls the mix; ``smooth_range``
    is the (small) amplitude of the smooth-deformation component.
    """
    if rng.random() < rigid_frac:
        return rigid_translation(H, W, rng, disp_range)
    return smooth_random(H, W, rng, smooth_range, blur_sigma_range)


@register("uniform_strain")
def uniform_strain(H, W, rng, disp_range, exx=0.0, eyy=0.0, exy=0.0, **_):
    """Linear displacement from a constant strain tensor, centered at the image middle.

    u = exx*xc + exy*yc,  v = exy*xc + eyy*yc  (xc, yc are centered pixel coords).
    """
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xc = xx - (W - 1) / 2.0
    yc = yy - (H - 1) / 2.0
    u = (exx * xc + exy * yc).astype(np.float32)
    v = (exy * xc + eyy * yc).astype(np.float32)
    return u, v


@register("sinusoidal")
def sinusoidal(H, W, rng, disp_range, amplitude=1.0, wavelength=40.0, axis="x", **_):
    """Sinusoidal displacement along one axis (building block for high-frequency benchmarks)."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    if axis == "x":
        u = (amplitude * np.sin(2 * np.pi * xx / wavelength)).astype(np.float32)
        v = np.zeros((H, W), dtype=np.float32)
    else:
        u = np.zeros((H, W), dtype=np.float32)
        v = (amplitude * np.sin(2 * np.pi * yy / wavelength)).astype(np.float32)
    return u, v


@register("crack_mode1_simple")
def crack_mode1_simple(H, W, rng, disp_range, point=None, angle=None, magnitude=None,
                       crop_size=None, disp_range_lo=0.0, info_out=None, **_):
    """Mode I displacement: crack opening (full crack, rigid)

    A line through a random ``point`` at a random ``angle`` splits the image into two
    half-planes. 
    
    One plane is rigidly translated along the line's NORMAL direction --
    opening the crack -- by ``magnitude``; the other half is left at zero displacement.
    Mask applied over opening area
    
    ``point`` (x, y) defaults to uniform random within centered
    ``crop_size`` x ``crop_size`` region. ``angle`` (line orientation) defaults to
    uniform random in [0, 2*pi), measured in the +u-right/+v-down frame. ``magnitude``
    defaults to uniform random in ``[disp_range_lo, disp_range]``.
    """
    if point is None:
        if crop_size is not None:
            y0, x0 = (H - crop_size) / 2.0, (W - crop_size) / 2.0
            px = float(rng.uniform(x0, x0 + crop_size))
            py = float(rng.uniform(y0, y0 + crop_size))
        else:
            px = float(rng.uniform(0, W))
            py = float(rng.uniform(0, H))
    else:
        px, py = point
    if angle is None:
        angle = float(rng.uniform(0, 2 * np.pi))
    if magnitude is None:
        magnitude = float(rng.uniform(disp_range_lo, disp_range))

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    nx, ny = -np.sin(angle), np.cos(angle)
    moving_side = (xx - px) * nx + (yy - py) * ny >= 0

    u = np.where(moving_side, magnitude * nx, 0.0).astype(np.float32)
    v = np.where(moving_side, magnitude * ny, 0.0).astype(np.float32)
    if info_out is not None:
        info_out["point"] = (px, py)
        info_out["angle"] = angle
        info_out["magnitude"] = magnitude
    return u, v


@register("crack_mode1_symm")
def crack_mode1_symm(H, W, rng, disp_range, point=None, angle=None, magnitude=None,
                     crop_size=None, disp_range_lo=0.0, info_out=None, **_):
    """Mode I displacement: crack opening, symmetric (both sides move by magnitude/2)

    Same as ``crack_mode1_simple``, but both planes translate outward, leaving crack
    with 0 displacement
    Crack opens by total magnitude still
    """
    if point is None:
        if crop_size is not None:
            y0, x0 = (H - crop_size) / 2.0, (W - crop_size) / 2.0
            px = float(rng.uniform(x0, x0 + crop_size))
            py = float(rng.uniform(y0, y0 + crop_size))
        else:
            px = float(rng.uniform(0, W))
            py = float(rng.uniform(0, H))
    else:
        px, py = point
    if angle is None:
        angle = float(rng.uniform(0, 2 * np.pi))
    if magnitude is None:
        magnitude = float(rng.uniform(disp_range_lo, disp_range))

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    nx, ny = -np.sin(angle), np.cos(angle)
    dist = (xx - px) * nx + (yy - py) * ny
    half = magnitude / 2.0
    moving_side = dist > half
    stationary_side = dist < -half
    # |dist| <= half: the vacated crack band -- left at 0 (see crack_gap_mask(symmetric=True))

    u = np.where(moving_side, half * nx, np.where(stationary_side, -half * nx, 0.0)).astype(np.float32)
    v = np.where(moving_side, half * ny, np.where(stationary_side, -half * ny, 0.0)).astype(np.float32)
    if info_out is not None:
        info_out["point"] = (px, py)
        info_out["angle"] = angle
        info_out["magnitude"] = magnitude
    return u, v


def crack_gap_mask(H: int, W: int, point: tuple[float, float], angle: float,
                   magnitude: float, symmetric: bool = False) -> np.ndarray:
    """Boolean mask of crack gap: True inside the invalid opening band.

    ``symmetric=False`` (``crack_mode1_simple``): the band is entirely on the moving
    side, from the line out to ``magnitude`` px, treated as invalid ground truth.

    ``symmetric=True`` (``crack_mode1_symm``): both faces retreat by ``magnitude / 2``,
    band left at zero displacement, treated as invalid ground truth.
    """
    px, py = point
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    nx, ny = -np.sin(angle), np.cos(angle)
    dist = (xx - px) * nx + (yy - py) * ny
    if symmetric:
        return np.abs(dist) <= magnitude / 2.0
    return (dist >= 0) & (dist <= magnitude)


_SENSE_STREAM = 0x5E05E  # spawn-key tag of the stream that draws the mode II shear sense
_SPLIT_STREAM = 0x5B117  # spawn-key tag of the stream that draws how the slip is shared


def _side_stream(rng: np.random.Generator, tag: int) -> np.random.Generator:
    """A generator on a fixed child of ``rng``'s seed sequence. Using it draws nothing from ``rng``
    and does not depend on how often it is called, so a sample keeps all its other random values."""
    ss = rng.bit_generator.seed_seq
    return np.random.default_rng(np.random.SeedSequence(ss.entropy, spawn_key=tuple(ss.spawn_key) + (tag,)))


def draw_shear_sense(rng: np.random.Generator) -> int:
    """+1 or -1 with equal probability, from its own stream (``_side_stream``)."""
    return 1 if _side_stream(rng, _SENSE_STREAM).random() < 0.5 else -1


def draw_shear_split(rng: np.random.Generator) -> float:
    """Share of the slip carried by the +normal side, uniform in [0, 1), from its own stream
    (``_side_stream``), independent of the shear sense."""
    return float(_side_stream(rng, _SPLIT_STREAM).random())


@register("crack_mode2_simple")
def crack_mode2_simple(H, W, rng, disp_range, point=None, angle=None, magnitude=None,
                       crop_size=None, disp_range_lo=0.0, sense=None, random_sense=False,
                       split=None, random_split=False, info_out=None, **_):
    """Mode II displacement: in-plane shear (rigid body)

    A line through a random ``point`` at a random ``angle`` splits the image into two
    half-planes. Both halves translate rigidly along the line (the TANGENTIAL direction) in
    opposite directions, so the slip across the line is ``magnitude``; ``split`` sets how it is
    shared. The original generator (``split`` = 1, the default) moves one half by ``magnitude`` and
    leaves the other at exactly zero displacement.

    ``split`` in [0, 1] is the share of the slip carried by the +normal half: it moves by
    ``split * magnitude`` along the sense direction, the other half by ``(1 - split) * magnitude``
    the opposite way. The slip (+ half minus - half) is ``magnitude`` for every split.
    ``random_split`` draws ``split`` uniformly in [0, 1) (``draw_shear_split``) without drawing
    from ``rng``: speckle, line, magnitude, sense and noise are those of the ``split=1`` sample of
    the same seed.

    ``point`` (x, y) defaults to uniform random within centered
    ``crop_size`` x ``crop_size`` region. ``angle`` (line orientation)
    defaults to uniform random in [0, 2*pi), measured in the
    +u-right/+v-down frame. ``magnitude`` defaults to uniform random in
    ``[disp_range_lo, disp_range]``.

    ``sense`` is the shear sense. +1 translates the half-plane on the +normal side of the line
    along +(cos(angle), sin(angle)); -1 translates the same half-plane the opposite way, the mirror
    image of +1. Rotating the line never changes the sense, so without this parameter every crack
    has sense +1 (the original generator, kept as the default so old datasets regenerate exactly).
    ``random_sense`` draws +1 or -1 with equal probability (``draw_shear_sense``) without drawing
    from ``rng``: speckle, line, magnitude and noise are the same as for the ``sense=+1`` sample of
    the same seed.
    """
    if point is None:
        if crop_size is not None:
            y0, x0 = (H - crop_size) / 2.0, (W - crop_size) / 2.0
            px = float(rng.uniform(x0, x0 + crop_size))
            py = float(rng.uniform(y0, y0 + crop_size))
        else:
            px = float(rng.uniform(0, W))
            py = float(rng.uniform(0, H))
    else:
        px, py = point
    if angle is None:
        angle = float(rng.uniform(0, 2 * np.pi))
    if magnitude is None:
        magnitude = float(rng.uniform(disp_range_lo, disp_range))
    if sense is None:
        sense = draw_shear_sense(rng) if random_sense else 1
    if sense not in (1, -1):
        raise ValueError(f"shear sense must be +1 or -1, got {sense!r}")
    if split is None:
        split = draw_shear_split(rng) if random_split else 1.0
    split = float(split)
    if not 0.0 <= split <= 1.0:
        raise ValueError(f"shear split must be in [0, 1], got {split!r}")

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    nx, ny = -np.sin(angle), np.cos(angle)
    plus_side = (xx - px) * nx + (yy - py) * ny >= 0

    tx, ty = sense * np.cos(angle), sense * np.sin(angle)
    plus = split * magnitude
    minus = -(1.0 - split) * magnitude
    # split = 1: the literal 0.0 of the original generator (0.0 * tx would be -0.0 for tx < 0),
    # so datasets made before the split existed regenerate byte for byte
    mu, mv = (0.0, 0.0) if split == 1.0 else (minus * tx, minus * ty)
    u = np.where(plus_side, plus * tx, mu).astype(np.float32)
    v = np.where(plus_side, plus * ty, mv).astype(np.float32)
    if info_out is not None:
        info_out["point"] = (px, py)
        info_out["angle"] = angle
        info_out["magnitude"] = magnitude
        info_out["sense"] = int(sense)
        info_out["split"] = split
    return u, v

@register("crack_mode2_strain")
def crack_mode2_strain(H, W, rng, disp_range, point=None, angle=None, magnitude=None,
                       crop_size=None, disp_range_lo=0.0, sense=None, random_sense=False,
                       split=None, random_split=False,
                       exx=None, eyy=None, exy=None, strain_range=0.02, info_out=None, **_):
    """Mode II displacement: in-plane shear (strain)

    Same in-plane shear crack as ``crack_mode2_simple`` (line with point and angle, rigidly translated
    by magnitude). Added uniform strain field fromm ``uniform_strain`` added over whole image

    ``exx`` / ``eyy`` / ``exy`` default to uniform random draws in
    ``[-strain_range, strain_range]``. Strain field centered at image middle, so the crop sees 
    ``+/- strain * crop_size / 2`` px at its edges -- keep ``strain_range`` small enough 
    that the displacement gradient stays << 1 and the inverse warp does not fold.
    """
    u, v = crack_mode2_simple(H, W, rng, disp_range, point=point, angle=angle,
                              magnitude=magnitude, crop_size=crop_size,
                              disp_range_lo=disp_range_lo, sense=sense,
                              random_sense=random_sense, split=split,
                              random_split=random_split, info_out=info_out)

    if exx is None:
        exx = float(rng.uniform(-strain_range, strain_range))
    if eyy is None:
        eyy = float(rng.uniform(-strain_range, strain_range))
    if exy is None:
        exy = float(rng.uniform(-strain_range, strain_range))

    du, dv = uniform_strain(H, W, rng, disp_range, exx=exx, eyy=eyy, exy=exy)
    u = (u + du).astype(np.float32)
    v = (v + dv).astype(np.float32)

    if info_out is not None:
        info_out["exx"] = exx
        info_out["eyy"] = eyy
        info_out["exy"] = exy
    return u, v

def make_displacement_field(field_type: str, H: int, W: int, rng: np.random.Generator,
                            disp_range: float, field_params: dict | None = None,
                            rotations_deg=(0, 90, 180, 270), info_out: dict | None = None,
                            disp_range_lo: float = 0.0):
    """Dispatch to a registered generator and apply lattice rotation augmentation.

    For field types flagged ``independent_rotation`` (smooth_random / natural_image), u and v
    are rotated independently by a random multiple of 90 degrees, reproducing the legacy
    augmentation. Physically coupled fields (translation/strain/sinusoidal) are not rotated.
    """
    if field_type not in REGISTRY:
        raise KeyError(f"Unknown displacement field_type '{field_type}'. "
                       f"Registered: {sorted(REGISTRY)}")
    fn, independent_rot = REGISTRY[field_type]
    u, v = fn(H, W, rng, disp_range, info_out=info_out, disp_range_lo=disp_range_lo,
             **(field_params or {}))
    if independent_rot and rotations_deg:
        u = _rotate_scalar(u, int(rng.choice(rotations_deg)))
        v = _rotate_scalar(v, int(rng.choice(rotations_deg)))
    return u.astype(np.float32), v.astype(np.float32)
