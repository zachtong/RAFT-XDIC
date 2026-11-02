"""Evaluation utilities: endpoint-error (EPE) over a dataset split."""

from __future__ import annotations

import numpy as np
import torch

from .datasets import DICDataset


@torch.no_grad()
def validate(model, dataset_root, split="val", device=None, iters=20, image_ext="png"):
    """Return mean EPE and px-accuracy over a split. Follows CONVENTION.md (no sign flip)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()

    try:
        dataset = DICDataset(dataset_root, split=split, image_ext=image_ext)
    except ValueError:
        # No such split (e.g. tiny smoke dataset with empty val); fall back to train.
        dataset = DICDataset(dataset_root, split="train", image_ext=image_ext)

    epe_list = []
    for i in range(len(dataset)):
        img1, img2, flow_gt, _ = dataset[i]
        img1 = img1[None].to(device)
        img2 = img2[None].to(device)
        _, flow_pr = model(img1, img2, iters=iters, test_mode=True)
        flow_pr = flow_pr[0].cpu()
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    if was_training:
        model.train()
    return {
        "epe": float(np.mean(epe_all)),
        "1px": float(np.mean(epe_all < 1)),
        "3px": float(np.mean(epe_all < 3)),
        "5px": float(np.mean(epe_all < 5)),
    }
