"""The single canonical RAFTcorr dataloader.

Reads the canonical on-disk layout (see CONVENTION.md):

    <root>/ref/NNNNN_ref.png   (image1, reference)
    <root>/def/NNNNN_def.png   (image2, deformed)
    <root>/u/NNNNN_u.npy       (+right)
    <root>/v/NNNNN_v.npy       (+down)
    <root>/split.json          ({"train": [...ids], "val": [...ids]})

It returns ``(img1=reference, img2=deformed, flow=(u, v), valid)`` with **no sign flip**.
This replaces the three legacy near-duplicate dataloaders (one of which negated the flow and
all of which fed [deformed, reference] in the wrong order).

If ``valid/NNNNN_valid.npy`` exists mode 1 sample, added to ``valid`` mask so crack-gap
band is excluded from supervision. Samples without a ``valid/`` folder are unaffected 
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class DICDataset(data.Dataset):
    def __init__(self, root: str | Path, split: str = "train", image_ext: str = "png"):
        self.root = Path(root)
        self.image_ext = image_ext
        self.samples = self._index(split)
        if not self.samples:
            raise ValueError(f"No samples found for split='{split}' under {self.root}")

    def _index(self, split: str):
        split_file = self.root / "split.json"
        if split_file.exists():
            ids = json.loads(split_file.read_text()).get(split, [])
        else:
            # Fall back to all ids found on disk.
            ids = sorted(p.stem.split("_")[0] for p in (self.root / "ref").glob(f"*_ref.{self.image_ext}"))
        samples = []
        for key in ids:
            valid_path = self.root / "valid" / f"{key}_valid.npy"
            samples.append({
                "ref": self.root / "ref" / f"{key}_ref.{self.image_ext}",
                "def": self.root / "def" / f"{key}_def.{self.image_ext}",
                "u": self.root / "u" / f"{key}_u.npy",
                "v": self.root / "v" / f"{key}_v.npy",
                "valid": valid_path if valid_path.exists() else None,
                "id": key,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_gray_3ch(path) -> np.ndarray:
        img = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        return np.tile(img[..., None], (1, 1, 3))  # 1ch -> 3ch (matches GUI + legacy)

    def __getitem__(self, index):
        s = self.samples[index]
        img1 = self._load_gray_3ch(s["ref"])   # image1 = reference
        img2 = self._load_gray_3ch(s["def"])   # image2 = deformed

        u = np.load(s["u"]).astype(np.float32)
        v = np.load(s["v"]).astype(np.float32)
        flow = np.stack([u, v], axis=-1)        # (H, W, 2); channel 0 = u, 1 = v. NO negation.

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        if s["valid"] is not None:
            on_disk_valid = torch.from_numpy(np.load(s["valid"]).astype(bool))
            valid = valid & on_disk_valid
        valid = valid.float()
        return img1, img2, flow, valid


def fetch_dataloader(root, split="train", batch_size=12, num_workers=4,
                     shuffle=True, image_ext="png"):
    dataset = DICDataset(root, split=split, image_ext=image_ext)
    # persistent_workers=False and a small prefetch keep the per-worker shared-memory footprint
    # bounded. With large (e.g. 512x512) images on Windows, persistent workers + prefetch=4 leak
    # shared-memory mappings across epochs until the paging-file commit limit is hit
    # ("Couldn't open shared file mapping ... error 1455") and the run dies mid-training.
    loader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False,
        prefetch_factor=(2 if num_workers > 0 else None),
        drop_last=(split == "train"),
    )
    print(f"[{split}] {len(dataset)} pairs from {root}")
    return loader
