"""Dataset builder: synthesize (reference, deformed, u, v) training samples.

Port of the legacy MATLAB ``dataset_generation_v4_Zach_modified.m`` driver, rewritten in the
canonical convention (see CONVENTION.md):

  * reference image  -> ``ref/NNNNN_ref.png``   (image1, I0)
  * deformed image   -> ``def/NNNNN_def.png``   (image2, I1)
  * u displacement   -> ``u/NNNNN_u.npy``       (float32, +right)
  * v displacement   -> ``v/NNNNN_v.npy``       (float32, +down)
  * valid mask for mode I samples -> ``valid/NNNNN_valid.npy`` (bool, True=valid, False=crack-gap)

The displacement ``(u, v)`` is the reference->deformed forward field; the deformed image is
produced by inverse-warping the reference with it. No sign flip is applied anywhere.

Reproducibility: a global seed is expanded via ``numpy.random.SeedSequence`` into one
independent generator per sample, so any sample can be regenerated deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .displacement_fields import crack_gap_mask, make_displacement_field
from .speckle import generate_speckle_pattern, get_random_speckle_params


def _center_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    H, W = arr.shape[:2]
    y0 = (H - crop) // 2
    x0 = (W - crop) // 2
    return arr[y0:y0 + crop, x0:x0 + crop]


def generate_pair(cfg: dict, rng: np.random.Generator, crack: bool = False,
                  keep_full: bool = False) -> dict:
    """Generate one training sample. Returns a dict with ref/def/u/v arrays and an info row.

    ``keep_full`` also returns the uncropped canvas under ``"full"`` (noisy ref/def, u, v, and the
    crop offset); it draws no extra random numbers, so the cropped sample is unchanged.
    """
    full = cfg["image"]["full_size"]
    crop = cfg["image"]["crop_size"]

    # 1. Reference speckle (full canvas, no noise yet).
    sp = get_random_speckle_params(cfg["speckle"], rng)
    ref_full, sp_info = generate_speckle_pattern(full, full, sp, rng)

    # 2. Reference->deformed displacement field (full canvas).
    disp = cfg["displacement"]
    range_cfg = disp["range"]
    if isinstance(range_cfg, (list, tuple)):
        r_lo, r_hi = range_cfg
        disp_range = float(r_hi)
        disp_range_lo = float(r_lo)
    else:
        disp_range = float(range_cfg)
        disp_range_lo = 0.0
    field_info: dict = {} if crack else None
    u_full, v_full = make_displacement_field(
        disp["field_type"], full, full, rng,
        disp_range=disp_range,
        field_params=disp.get("field_params"),
        rotations_deg=disp.get("rotations_deg", (0, 90, 180, 270)),
        info_out=field_info,
        disp_range_lo=disp_range_lo,
    )

    # Crack gap band - Mode I (one-sided "simple" or symmetric "symm" opening)
    is_mode1 = bool(field_info) and disp["field_type"] in ("crack_mode1_simple", "crack_mode1_symm")
    gap_mask_full = None
    if is_mode1:
        gap_mask_full = crack_gap_mask(full, full, field_info["point"], field_info["angle"],
                                       field_info["magnitude"],
                                       symmetric=(disp["field_type"] == "crack_mode1_symm"))

    # 3. Deformed image via inverse warp.
    from .warp import inverse_warp_gray
    def_full = inverse_warp_gray(ref_full, u_full, v_full)

    # Paint crack gap band
    if gap_mask_full is not None:
        def_full = def_full.copy()
        def_full[gap_mask_full] = np.uint8(round(sp_info["background_intensity"]))

    # 4. Independent additive Gaussian noise on ref and def (different realizations).
    n_lo, n_hi = cfg["noise"]["std_range"]
    noise_std = n_lo + (n_hi - n_lo) * rng.random()

    def add_noise(img):
        out = img.astype(np.float64) + noise_std * rng.standard_normal(img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    ref_noisy = add_noise(ref_full)
    def_noisy = add_noise(def_full)

    # 5. Center crop everything to remove warp boundary artifacts.
    ref_c = _center_crop(ref_noisy, crop)
    def_c = _center_crop(def_noisy, crop)
    u_c = _center_crop(u_full, crop).astype(np.float32)
    v_c = _center_crop(v_full, crop).astype(np.float32)
    valid_c = None
    if gap_mask_full is not None:
        valid_c = ~_center_crop(gap_mask_full, crop)

    # ``field_info["point"]`` was reported in full-canvas coords; shift into the crop-local
    # frame that consumers (u_c/v_c, run_crack_inference.py) actually operate in.
    if field_info:
        crop_ofs = (full - crop) // 2
        px, py = field_info["point"]
        field_info["point"] = (px - crop_ofs, py - crop_ofs)

    info = {
        "diameter": sp_info["diameter"],
        "density": sp_info["density"],
        "spacing": sp_info["spacing"],
        "eccentricity": sp_info["eccentricity"],
        "background": sp_info["background"],
        "speckle_intensity": sp_info["speckle_intensity"],
        "background_intensity": sp_info["background_intensity"],
        "blur_sigma": sp_info["blur_sigma"],
        "noise_std": noise_std,
        "field_type": disp["field_type"],
        "u_min": float(u_c.min()), "u_max": float(u_c.max()),
        "v_min": float(v_c.min()), "v_max": float(v_c.max()),
    }
    if field_info:
        if is_mode1:
            info["opening_magnitude"] = field_info["magnitude"]
            info["shear_magnitude"] = 0.0
        else:
            info["opening_magnitude"] = 0.0
            info["shear_magnitude"] = field_info["magnitude"]
            info["shear_sense"] = field_info.get("sense")
            # share of the slip on the +normal side; only for configs that set it, so params.csv
            # of the older one-side-at-rest sets regenerates unchanged
            fp = disp.get("field_params") or {}
            if fp.get("random_split") or "split" in fp:
                info["shear_split"] = field_info["split"]
        info["point_x"] = field_info["point"][0]
        info["point_y"] = field_info["point"][1]
        info["angle"] = field_info["angle"]
        # Only strain-carrying fields report these.
        if "exx" in field_info:
            info["exx"] = field_info["exx"]
            info["eyy"] = field_info["eyy"]
            info["exy"] = field_info["exy"]
    out = {"ref": ref_c, "def": def_c, "u": u_c, "v": v_c, "valid": valid_c, "info": info}
    if keep_full:
        out["full"] = {"ref": ref_noisy, "def": def_noisy, "u": u_full.astype(np.float32),
                       "v": v_full.astype(np.float32), "crop_offset": (full - crop) // 2}
    return out


def build_dataset(cfg: dict, out_dir: str | Path, num_pairs: int | None = None,
                  seed: int | None = None, start_index: int = 0,
                  progress: bool = True, crack: bool = False) -> Path:
    """Generate ``num_pairs`` samples into ``out_dir`` following the canonical layout."""
    out_dir = Path(out_dir)
    if num_pairs is None:
        num_pairs = int(cfg["num_pairs"])
    if seed is None:
        seed = int(cfg.get("seed", 0))

    for sub in ("ref", "def", "u", "v"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    child_seeds = np.random.SeedSequence(seed).spawn(num_pairs)
    rows = []

    iterator = range(num_pairs)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="generating")
        except ImportError:
            pass

    for i in iterator:
        rng = np.random.default_rng(child_seeds[i])
        sample = generate_pair(cfg, rng, crack=crack)
        idx = start_index + i
        key = f"{idx:05d}"

        img_fmt = cfg["output"].get("image_format", "png")
        Image.fromarray(sample["ref"]).save(out_dir / "ref" / f"{key}_ref.{img_fmt}")
        Image.fromarray(sample["def"]).save(out_dir / "def" / f"{key}_def.{img_fmt}")
        np.save(out_dir / "u" / f"{key}_u.npy", sample["u"])
        np.save(out_dir / "v" / f"{key}_v.npy", sample["v"])
        if sample["valid"] is not None:
            (out_dir / "valid").mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "valid" / f"{key}_valid.npy", sample["valid"])

        row = {"id": key, "index": idx, **sample["info"]}
        rows.append(row)

    # Write params.csv (pandas if available, else a minimal CSV).
    _write_params_csv(rows, out_dir / "params.csv")

    # Deterministic train/val split.
    val_frac = float(cfg["output"].get("val_fraction", 0.1))
    ids = [r["id"] for r in rows]
    n_val = int(round(len(ids) * val_frac))
    split = {"train": ids[n_val:], "val": ids[:n_val]}
    (out_dir / "split.json").write_text(json.dumps(split, indent=2))

    meta = {"config": cfg, "seed": seed, "num_pairs": num_pairs,
            "start_index": start_index, "crack": crack, "convention": "CONVENTION.md"}
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return out_dir


def _write_params_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
    except ImportError:
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
