"""Training for the unified RAFTcorr model.

Optimizer/schedule follow the paper: **AdamW + StepLR** (initial LR 1e-4, reduced 10% every
5 epochs, i.e. gamma=0.9 with a 5-epoch step). Loss is the standard RAFT sequence loss with
exponential weighting (gamma).

All data flows through the single canonical dataloader (raftcorr/datasets.py), so the
reference/deformed ordering and the displacement sign follow CONVENTION.md with no negation.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .core import build_raft
from .datasets import fetch_dataloader
from .evaluate import validate

MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.9, max_flow=MAX_FLOW):
    """L1 loss over the GRU prediction sequence with exponential weighting."""
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


def fetch_optimizer(model, train_cfg, steps_per_epoch):
    """Build optimizer + LR scheduler.

    Two schedules (``train_cfg["schedule"]``):
      * ``steplr`` (default) -- AdamW + StepLR, the paper's finetuning schedule: initial LR
        reduced 10% every ``lr_step_epochs`` epochs. Good when starting from pretrained weights.
      * ``onecycle`` -- AdamW + OneCycleLR with warmup (``pct_start``), the original RAFT
        *from-scratch* schedule. A brief warmup then linear anneal is what lets a from-scratch
        model escape the zero-flow init and actually learn large motion; plain constant-LR
        StepLR from scratch barely moves off zero.

    The scheduler is stepped once per iteration in the training loop.
    """
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("wdecay", 1e-5),
            eps=train_cfg.get("epsilon", 1e-8),
            betas=(0.9, 0.999),
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("wdecay", 1e-5),
            eps=train_cfg.get("epsilon", 1e-8),
        )
    schedule = str(train_cfg.get("schedule", "steplr")).lower()
    if schedule == "onecycle":
        num_steps = max(1, steps_per_epoch * int(train_cfg["num_epochs"]))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=train_cfg["lr"], total_steps=num_steps + 10,
            pct_start=float(train_cfg.get("pct_start", 0.05)),
            cycle_momentum=False, anneal_strategy="linear",
        )
    else:
        step_epochs = int(train_cfg.get("lr_step_epochs", 5))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, step_epochs * steps_per_epoch),
            gamma=float(train_cfg.get("lr_gamma", 0.9)),
        )
    return optimizer, scheduler


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA): fixes weight init and data order.
    cuDNN / atomic-add kernels are still nondeterministic, so two seeded runs agree only
    statistically, not bit for bit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_atomic(obj, path: Path) -> None:
    """torch.save to a temporary file, then rename, so a crash mid-write keeps the old file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(cfg: dict, dataset_root, out_dir, device=None, num_epochs=None, restore_ckpt=None,
          seed=None, resume=False, mem_fraction=None):
    """Train one model. ``cfg`` is a loaded train YAML (has ``model`` and ``train`` blocks).

    ``restore_ckpt``: optional path to initialize weights from (loaded strict=False, ``module.``
    prefix stripped). Use an optical-flow-pretrained RAFT (e.g. raft-things.pth) here to get
    large-displacement extrapolation that from-scratch DIC training never learns.
    ``seed``: optional RNG seed (weight init + shuffle order); None keeps the old unseeded behaviour.
    The seed, the init checkpoint and the dataset root are recorded in run_config.json (written at
    start) and summary.json (written at the end).
    ``resume``: continue from ``checkpoints/<name>_state.pth`` (model, optimizer, scheduler, grad
    scaler, epoch, best validation EPE), written after every epoch. The data order after a resume
    differs from an uninterrupted run; the epochs resumed from are recorded in summary.json.
    ``mem_fraction``: on CUDA, cap PyTorch at this fraction of the GPU's memory, so an overrun
    raises an out-of-memory error instead of spilling into shared system memory.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if mem_fraction is not None and str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction))

    model_cfg = dict(cfg["model"])
    train_cfg = dict(cfg["train"])
    if num_epochs is not None:
        train_cfg["num_epochs"] = num_epochs
    state_path = out_dir / "checkpoints" / f"{cfg.get('name', 'raftcorr')}_state.pth"
    state = None
    if resume:
        if not state_path.exists():
            raise FileNotFoundError(f"--resume given but no training state at {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=False)

    if seed is not None:
        # first, so an invalid seed fails before anything is written; offset by the resumed epoch
        # so a resumed run does not replay the shuffle order of epoch 1
        seed_everything(int(seed) + (state["epoch"] if state else 0))
    provenance = {"seed": seed, "restore_ckpt": str(restore_ckpt) if restore_ckpt else None,
                  "dataset_root": str(dataset_root),
                  "resumed_from_epochs": (state.get("resumed_from_epochs", []) + [state["epoch"]])
                  if state else []}
    if state is None:
        (out_dir / "run_config.json").write_text(json.dumps(
            {**provenance, "model_cfg": model_cfg, "train_cfg": train_cfg, "device": device,
             "mem_fraction": mem_fraction, "torch": torch.__version__,
             "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}, indent=2))
    print(f"[init] seed={seed} restore_ckpt={provenance['restore_ckpt']} "
          f"resume_from_epoch={state['epoch'] if state else None} mem_fraction={mem_fraction}")

    model = build_raft(model_cfg).to(device)
    if restore_ckpt:
        _ck = torch.load(restore_ckpt, map_location=device, weights_only=False)
        _sd = _ck["model"] if isinstance(_ck, dict) and "model" in _ck else _ck
        _sd = {k.replace("module.", ""): v for k, v in _sd.items()}
        _miss, _unexp = model.load_state_dict(_sd, strict=False)
        print(f"[restore] {restore_ckpt}: missing={len(_miss)} unexpected={len(_unexp)}")

    # Freeze BatchNorm only when FINETUNING (restoring pretrained weights). Training from
    # scratch with frozen BN keeps the context encoder's BN stuck at its random-init running
    # stats, which cripples learning -- original RAFT only freezes BN in finetune stages, and
    # trains BN from scratch in the chairs (from-scratch) stage. Override via train_cfg.
    freeze_bn = bool(train_cfg.get("freeze_bn", restore_ckpt is not None)) \
        and not model_cfg.get("small", False)
    print(f"[bn] freeze_bn={freeze_bn} (restore_ckpt={'yes' if restore_ckpt else 'no'})")
    model.train()
    if freeze_bn:
        model.freeze_bn()

    image_ext = cfg.get("image_ext", "png")
    train_loader = fetch_dataloader(dataset_root, "train",
                                    batch_size=train_cfg["batch_size"],
                                    num_workers=train_cfg.get("num_workers", 4),
                                    image_ext=image_ext)
    steps_per_epoch = max(1, len(train_loader))
    num_epochs_v = int(train_cfg["num_epochs"])
    num_steps = steps_per_epoch * num_epochs_v

    optimizer, scheduler = fetch_optimizer(model, train_cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=model_cfg.get("mixed_precision", True)
                                  and device == "cuda")
    iters = train_cfg.get("iters", 12)
    gamma = train_cfg.get("gamma", 0.9)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    global_step, best_val, start_epoch = 0, float("inf"), 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        global_step, best_val, start_epoch = state["global_step"], state["best_val"], state["epoch"]
        print(f"[resume] {state_path}: epoch {start_epoch}, best val EPE {best_val:.4f}")

    csv_path = out_dir / "training_log.csv"
    if state is None:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_epe", "val_epe",
                                    "val_1px", "val_3px", "val_5px", "lr"])

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    for epoch in range(start_epoch, num_epochs_v):
        model.train()
        if freeze_bn:
            model.freeze_bn()
        running_loss, running_epe, n = 0.0, 0.0, 0

        for img1, img2, flow, valid in train_loader:
            img1, img2 = img1.to(device), img2.to(device)
            flow, valid = flow.to(device), valid.to(device)

            optimizer.zero_grad()
            flow_preds = model(img1, img2, iters=iters)
            loss, metrics = sequence_loss(flow_preds, flow, valid, gamma=gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            running_loss += loss.item()
            running_epe += metrics["epe"]
            n += 1

        train_loss = running_loss / n
        train_epe = running_epe / n

        val = {"epe": float("nan"), "1px": 0, "3px": 0, "5px": 0}
        if (epoch + 1) % train_cfg.get("val_freq_epochs", 1) == 0:
            val = validate(model, dataset_root, device=device, iters=iters, image_ext=image_ext)

        lr = scheduler.get_last_lr()[0]
        peak = (f" | gpu_peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
                if str(device).startswith("cuda") else "")
        print(f"epoch {epoch+1:3d}/{num_epochs_v} | loss {train_loss:.4f} | "
              f"train_epe {train_epe:.4f} | val_epe {val['epe']:.4f} | lr {lr:.2e}{peak}",
              flush=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_epe, val["epe"],
                                    val["1px"], val["3px"], val["5px"], lr])

        writer.add_scalar("train/loss_epoch", train_loss, global_step)
        writer.add_scalar("train/epe_epoch", train_epe, global_step)
        writer.add_scalar("train/lr_epoch", lr, global_step)
        if not math.isnan(val["epe"]):
            writer.add_scalar("val/epe", val["epe"], global_step)
            writer.add_scalar("val/1px", val["1px"], global_step)
            writer.add_scalar("val/3px", val["3px"], global_step)
            writer.add_scalar("val/5px", val["5px"], global_step)

        ckpt = {"model": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch + 1}
        torch.save(ckpt, out_dir / "checkpoints" / f"{cfg.get('name', 'raftcorr')}_latest.pth")
        if train_cfg.get("save_every_epoch", False):
            torch.save(ckpt, out_dir / "checkpoints"
                       / f"{cfg.get('name', 'raftcorr')}_epoch{epoch + 1:03d}.pth")
        if val["epe"] < best_val:
            best_val = val["epe"]
            torch.save(ckpt, out_dir / "checkpoints" / f"{cfg.get('name', 'raftcorr')}_best.pth")
        _save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "epoch": epoch + 1, "global_step": global_step, "best_val": best_val,
                      "resumed_from_epochs": provenance["resumed_from_epochs"]}, state_path)

    (out_dir / "summary.json").write_text(json.dumps(
        {"best_val_epe": best_val, "num_epochs": num_epochs_v,
         "steps_per_epoch": steps_per_epoch, "model_cfg": model_cfg, **provenance}, indent=2))
    writer.close()
    return model
