#!/usr/bin/env python
"""CLI: train a RAFTcorr model from a YAML config.

Examples
--------
    python scripts/run_training.py --config configs/train_L.yaml \
        --dataset data/smoke --out runs/smoke_L --num-epochs 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raftcorr.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a RAFTcorr model.")
    ap.add_argument("--config", required=True, help="YAML train config (model + train blocks)")
    ap.add_argument("--dataset", required=True, help="dataset root (with ref/def/u/v + split.json)")
    ap.add_argument("--out", required=True, help="output dir for checkpoints/logs")
    ap.add_argument("--num-epochs", type=int, default=None, help="override config num_epochs")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")
    ap.add_argument("--restore-ckpt", default=None,
                    help="init weights from a checkpoint (strict=False); use a flow-pretrained "
                         "RAFT (raft-things.pth) to get large-displacement extrapolation")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for weight init and shuffle order (unseeded if omitted)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/checkpoints/<name>_state.pth after an interruption")
    ap.add_argument("--mem-fraction", type=float, default=0.8,
                    help="cap PyTorch at this fraction of the GPU's memory (default 0.8)")
    args = ap.parse_args()
    if not 0.0 < args.mem_fraction <= 1.0:
        ap.error("--mem-fraction must be in (0, 1]")

    cfg = yaml.safe_load(Path(args.config).read_text())
    train(cfg, args.dataset, args.out, device=args.device, num_epochs=args.num_epochs,
          restore_ckpt=args.restore_ckpt, seed=args.seed, resume=args.resume,
          mem_fraction=args.mem_fraction)


if __name__ == "__main__":
    main()
