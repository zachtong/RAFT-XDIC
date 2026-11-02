#!/usr/bin/env python
"""Small-displacement test: does a model return zero for small motions (a "dead zone")?

Runs on any checkpoint and needs no dataset on disk.

Each test pair is made by the training-data generator itself (``generate_pair``): the training
speckle settings, a 256 px canvas centre-cropped to 128 px, a uniform translation of s px
(``rigid_translation``, which draws no random numbers, so for one seed every shift reuses the same
speckle and noise) along x and along y, and Gaussian noise with a fixed std of 2 grey levels.
Reported per shift: the median over the speckle seeds of the median predicted displacement along
the shift direction over the crop (8 px border dropped). Dead-zone edge = the largest shift whose
measured value is below half the true shift (0 when there is none).

Usage:
    python scripts/check_dead_zone.py --ckpt run/checkpoints/raftcorr-F_best.pth [more ...]
        [--out dead_zone.csv] [--seeds 4] [--device cuda]
Exit code 0 always; read the "dead-zone edge" line.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from raftcorr.data_generation.build_dataset import generate_pair  # noqa: E402
from scripts.run_inference import infer, load_model  # noqa: E402

SPECKLE_CFG = ROOT / "configs" / "dataset_gen" / "mode2_7px_both_senses_split.yaml"
SHIFTS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5)
BORDER = 8
NOISE_STD = 2.0


def pair_cfg(shift: float, axis: str) -> dict:
    cfg = copy.deepcopy(yaml.safe_load(SPECKLE_CFG.read_text()))
    cfg["displacement"] = {"range": [0.0, 1.0], "field_type": "rigid_translation",
                           "field_params": {"u0": shift if axis == "x" else 0.0,
                                            "v0": shift if axis == "y" else 0.0},
                           "rotations_deg": [0]}
    cfg["noise"] = {"std_range": [NOISE_STD, NOISE_STD]}
    return cfg


def make_pairs(n_seeds: int) -> list[tuple[str, float, int, np.ndarray, np.ndarray]]:
    pairs = []
    for axis in ("x", "y"):
        for s in SHIFTS:
            for k in range(n_seeds):
                rng = np.random.default_rng(np.random.SeedSequence(2026_11_01).spawn(n_seeds)[k])
                p = generate_pair(pair_cfg(s, axis), rng)
                pairs.append((axis, s, k, p["ref"], p["def"]))
    return pairs


def measure(ckpt: str, pairs, device: str) -> dict[tuple[str, float], float]:
    model = load_model(ckpt, device)
    per = {}
    for axis, s, _, ref, dfm in pairs:
        u, v = infer(model, ref, dfm, device)
        comp = (u if axis == "x" else v)[BORDER:-BORDER, BORDER:-BORDER]
        per.setdefault((axis, s), []).append(float(np.median(comp)))
    return {key: float(np.median(vals)) for key, vals in per.items()}


def dead_edge(res: dict, axis: str) -> float:
    dead = [s for s in SHIFTS if s > 0 and res[(axis, s)] < 0.5 * s]
    return max(dead) if dead else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None, help="CSV (model, axis, shift, measured)")
    ap.add_argument("--seeds", type=int, default=4, help="speckle images per shift")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = make_pairs(args.seeds)
    rows = []
    for ckpt in args.ckpt:
        res = measure(ckpt, pairs, device)
        print(f"\n{ckpt}  (device {device}, {args.seeds} speckle seeds)")
        print("  true shift (px)   " + "  ".join(f"{s:6.2f}" for s in SHIFTS))
        for axis in ("x", "y"):
            print(f"  measured, {axis}     " + "  ".join(f"{res[(axis, s)]:6.3f}" for s in SHIFTS))
            rows += [[ckpt, axis, s, f"{res[(axis, s)]:.5f}"] for s in SHIFTS]
        ex, ey = dead_edge(res, "x"), dead_edge(res, "y")
        verdict = "none" if max(ex, ey) == 0 else f"up to {max(ex, ey):g} px"
        print(f"  dead-zone edge: x {ex:g} px, y {ey:g} px -> dead zone {verdict}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            csv.writer(fh).writerows([["model", "axis", "shift_px", "measured_px"]] + rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
