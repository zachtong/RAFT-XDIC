#!/bin/bash
# =============================================================================
# Smoke test on gh-dev (30 min) before the 24 h training job. Submit from the repository root:
#     sbatch vista/smoke.slurm
# Checks, in order: env and imports, GPU, dataset (extracts the uploaded tarball into $SCRATCH;
# the training job reuses it), a 2-epoch training on a 400-pair subset that is killed after
# epoch 1 and resumed (the resume path of the real job), the measured time per step, and the
# small-displacement test script. Ends with "SMOKE OK" or stops at the first failure.
# =============================================================================
#SBATCH --job-name=xdic_smoke
#SBATCH --output=vista/logs/%x_%j.out
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00

set -euo pipefail
ENV_FILE="${SLURM_SUBMIT_DIR:-.}/vista/env_vista.sh"
[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE not found: submit from the repository root"; exit 1; }
source "$ENV_FILE"
cd "$PROJECT_DIR"

echo "=== smoke | job ${SLURM_JOB_ID:-local} on $(hostname) @ $(date) ==="
echo "commit $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "--- 1. env and imports"
python - <<'EOF'
import importlib, sys
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(), "no GPU visible"
for m in ("numpy", "scipy", "PIL", "yaml", "pandas", "tensorboard", "cv2",
          "raftcorr.train", "raftcorr.datasets", "raftcorr.data_generation.build_dataset",
          "scripts.run_inference"):
    importlib.import_module(m)
print("imports ok")
EOF

echo "--- 2. dataset"
prepare_data

echo "--- 3. 400-pair subset (symlinks + reduced split.json)"
SMOKE_DATA="$DATA_ROOT/${DATASET}_smoke"
rm -rf "$SMOKE_DATA"; mkdir -p "$SMOKE_DATA"
for d in ref def u v; do ln -s "$DATA/$d" "$SMOKE_DATA/$d"; done
python - "$DATA/split.json" "$SMOKE_DATA/split.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
json.dump({"train": s["train"][:400], "val": s["val"][:80]}, open(sys.argv[2], "w"))
print("train 400, val 80")
EOF

echo "--- 4. train 2 epochs, kill after epoch 1, resume"
SMOKE_OUT="$RUNS_ROOT/${RUN}_smoke"
STATE="$SMOKE_OUT/checkpoints/raftcorr-F_state.pth"
rm -rf "$SMOKE_OUT"
ARGS=(--config "$TRAIN_CFG" --dataset "$SMOKE_DATA" --out "$SMOKE_OUT" --num-epochs 2 --seed "$SEED" --mem-fraction "$MEM_FRACTION")
T0=$(date +%s)
python -u scripts/run_training.py "${ARGS[@]}" &
PID=$!
until [ -f "$STATE" ] || ! kill -0 "$PID" 2>/dev/null; do sleep 2; done
T1=$(date +%s)
[ -f "$STATE" ] || { echo "ERROR: training ended before writing $STATE"; wait "$PID" || true; exit 1; }
kill "$PID"; wait "$PID" || true
echo "[smoke] epoch 1 (200 steps + 80 validation pairs + startup) took $((T1 - T0)) s; killed, resuming"
python -u scripts/run_training.py "${ARGS[@]}" --resume
python - "$SMOKE_OUT" <<'EOF'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "training_log.csv")))
summ = json.load(open(out / "summary.json"))
print("epochs logged:", [r["epoch"] for r in rows], "| resumed_from_epochs:", summ["resumed_from_epochs"],
      "| val EPE:", [round(float(r["val_epe"]), 3) for r in rows])
assert summ["resumed_from_epochs"] == [1] and rows[-1]["epoch"] == "2", "resume did not continue at epoch 2"
print("resume ok")
EOF
echo "[smoke] full run estimate: about $(( (T1 - T0) * 18 / 60 )) min per epoch of 3,600 steps (upper bound: includes startup), x $EPOCHS epochs"

echo "--- 5. small-displacement test script"
python -u scripts/check_dead_zone.py --seeds 1 --ckpt "$SMOKE_OUT/checkpoints/raftcorr-F_latest.pth"

echo "SMOKE OK @ $(date)"
