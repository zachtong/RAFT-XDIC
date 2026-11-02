#!/usr/bin/env python
"""Run a trained RAFTcorr checkpoint on one image pair and save the (u, v) displacement field.

This is the minimal, canonical inference path -- the same model build + forward as training and
validation. It follows CONVENTION.md exactly: ``image1 = reference``, ``image2 = deformed``, the
returned flow is the reference->deformed field with ``+u`` right and ``+v`` down (no sign flip).

Two ways to call it:

    # (a) two image files
    python scripts/run_inference.py --ckpt runs/smoke/checkpoints/raftcorr-L_best.pth \
        --ref path/to/ref.png --def path/to/def.png --out flow.npy

    # (b) a sample from a generated dataset (uses its stored ground truth for an error readout)
    python scripts/run_inference.py --ckpt <ckpt> --dataset data/smoke --id 00000 --out flow.npy

``--flip`` negates the saved/scored (u, v) and is only for a confirmed fine
checkpoint. Never pass it for a checkpoint trained by this repo's own train.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raftcorr.core import build_raft  # noqa: E402


def load_model(ckpt_path: str, device: str):
    """Load a RAFTcorr wrapper checkpoint ``{"model": state_dict, "model_cfg": {...}}``."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model" in ckpt and "model_cfg" in ckpt):
        raise ValueError("Expected a RAFTcorr checkpoint saved by raftcorr/train.py "
                         "(a dict with 'model' and 'model_cfg').")
    model = build_raft(dict(ckpt["model_cfg"]))
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} "
              f"(nonzero missing => weights did not load correctly)")
    return model.to(device).eval()


def to_tensor(gray: np.ndarray, device: str) -> torch.Tensor:
    """HxW uint8 grayscale -> [1, 3, H, W] float tensor (RAFT expects 3 channels, 0-255)."""
    img3 = np.tile(gray[..., None], (1, 1, 3)).astype(np.float32)
    return torch.from_numpy(img3).permute(2, 0, 1)[None].to(device)


@torch.no_grad()
def infer(model, ref: np.ndarray, deformed: np.ndarray, device: str, iters: int = 20):
    """Return (u, v) each HxW float32, cropped back to the input size."""
    r, d = to_tensor(ref, device), to_tensor(deformed, device)
    H, W = ref.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8          # RAFT needs H, W divisible by 8
    r = F.pad(r, (0, pw, 0, ph), mode="replicate")
    d = F.pad(d, (0, pw, 0, ph), mode="replicate")
    _, flow_up = model(r, d, iters=iters, test_mode=True)
    flow = flow_up[0, :, :H, :W].cpu().numpy()          # (2, H, W): channel 0 = u, 1 = v
    return flow[0], flow[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="RAFTcorr single-pair inference.")
    ap.add_argument("--ckpt", required=True, help="RAFTcorr checkpoint (.pth)")
    ap.add_argument("--ref", help="reference image (image1)")
    ap.add_argument("--def", dest="deformed", help="deformed image (image2)")
    ap.add_argument("--dataset", help="dataset root (use with --id instead of --ref/--def)")
    ap.add_argument("--id", help="sample id inside --dataset, e.g. 00000")
    ap.add_argument("--out", default="flow.npy", help="output .npy path for the (u, v) field")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--flip", action="store_true",
                    help="Negate the predicted (u, v) before saving/scoring (fine checkpoint)")
    args = ap.parse_args()
    if args.flip:
        print("[infer] --flip is set: negating predicted (u, v) before saving/scoring. "
              "Only use this for a confirmed legacy-sign checkpoint (CONVENTION.md #6).")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    gt = None
    if args.dataset and args.id:
        root = Path(args.dataset)
        ref = np.array(Image.open(root / "ref" / f"{args.id}_ref.png").convert("L"), np.uint8)
        deformed = np.array(Image.open(root / "def" / f"{args.id}_def.png").convert("L"), np.uint8)
        u_gt = np.load(root / "u" / f"{args.id}_u.npy")
        v_gt = np.load(root / "v" / f"{args.id}_v.npy")
        gt = (u_gt, v_gt)
    elif args.ref and args.deformed:
        ref = np.array(Image.open(args.ref).convert("L"), np.uint8)
        deformed = np.array(Image.open(args.deformed).convert("L"), np.uint8)
    else:
        ap.error("provide either --ref/--def or --dataset/--id")

    u, v = infer(model, ref, deformed, device, iters=args.iters)
    if args.flip:
        u, v = -u, -v
    np.save(args.out, np.stack([u, v]))
    print(f"[infer] device={device} iters={args.iters}  saved (u,v) -> {args.out}")
    print(f"[infer] predicted u: mean={u.mean():+.3f} range=[{u.min():+.2f}, {u.max():+.2f}]  "
          f"v: mean={v.mean():+.3f} range=[{v.min():+.2f}, {v.max():+.2f}]")
    if gt is not None:
        epe = float(np.mean(np.hypot(u - gt[0], v - gt[1])))
        print(f"[infer] end-point error vs ground truth: {epe:.4f} px")


if __name__ == "__main__":
    main()
