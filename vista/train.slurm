#!/bin/bash
# =============================================================================
# Training run on a Vista GH200 node (paths, data and recipe: vista/env_vista.sh).
#
# Submit from the repository root:   sbatch vista/train.slurm                    (RAFT-XDIC)
#                                    sbatch --job-name=control --export=ALL,RUN_VARIANT=control vista/train.slurm
#                                                                                (RAFT-DIC, matched control)
# Safe to resubmit any number of times: a finished run exits at once; an interrupted one resumes
# from checkpoints/raftcorr-F_state.pth (model, optimizer, OneCycle scheduler, AMP scaler, epoch).
# Next 24 h window if needed:        sbatch --dependency=afterany:<jobid> [same options] vista/train.slurm
# At the end it runs scripts/check_dead_zone.py on the best and latest checkpoints and copies the
# curated results to $WORK/raft-xdic/runs/<run>/ (not purged).
# =============================================================================
#SBATCH --job-name=xdic_split
#SBATCH --output=vista/logs/%x_%j.out
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# No --gres on Vista: the gh partition gives a whole node with its GPU.
# E-mail on end/failure: add --mail-type=END,FAIL --mail-user=<address> to the sbatch command.

set -euo pipefail
ENV_FILE="${SLURM_SUBMIT_DIR:-.}/vista/env_vista.sh"
[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE not found: submit from the repository root"; exit 1; }
source "$ENV_FILE"
cd "$PROJECT_DIR"

OUT="$RUNS_ROOT/$RUN"
SAFE="$SAFE_ROOT/$RUN"
STATE="$OUT/checkpoints/raftcorr-F_state.pth"
mkdir -p "$OUT" "$SAFE"

echo "=== $RUN | job ${SLURM_JOB_ID:-local} on $(hostname) @ $(date) ==="
echo "commit $(git rev-parse HEAD) ($(git status --porcelain | wc -l) uncommitted files)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

safekeep() {
    mkdir -p "$SAFE/checkpoints"
    cp -f "$OUT"/training_log.csv "$OUT"/run_config.json "$OUT"/summary.json "$OUT"/git_commits.txt \
          "$OUT"/dead_zone.csv "$SAFE/" 2>/dev/null || true
    cp -f "$OUT"/checkpoints/raftcorr-F_best.pth "$OUT"/checkpoints/raftcorr-F_latest.pth \
          "$OUT"/checkpoints/raftcorr-F_epoch*.pth "$SAFE/checkpoints/" 2>/dev/null || true
    echo "[safekeep] curated results -> $SAFE"
}

dead_zone_check() {
    [ -f "$OUT/dead_zone.csv" ] && { echo "[dead zone] already checked: $OUT/dead_zone.csv"; return 0; }
    srun python -u scripts/check_dead_zone.py --seeds 4 --out "$OUT/dead_zone.csv" \
        --ckpt "$OUT/checkpoints/raftcorr-F_best.pth" "$OUT/checkpoints/raftcorr-F_latest.pth"
}

# 1. Completion guard first: a finished run exits 0 even if its dataset has been purged.
#    The state file stores the number of finished epochs. No silent fallback: an unreadable file
#    stops the job instead of restarting a finished run from epoch 0.
DONE=0
if [ -f "$STATE" ]; then
    EP=$(python -c "import torch; print(int(torch.load('$STATE', map_location='cpu', weights_only=False)['epoch']))") \
        || { echo "ERROR: $STATE exists but cannot be read: inspect it (best: $OUT/checkpoints/raftcorr-F_best.pth)"; exit 1; }
    if [ "$EP" -ge "$EPOCHS" ] && [ -f "$OUT/summary.json" ]; then
        echo "[done] $EP of $EPOCHS epochs finished"
        DONE=1
    else
        echo "[guard] $EP of $EPOCHS epochs finished: resuming"
    fi
fi

if [ "$DONE" -eq 0 ]; then
    # 2. Inputs.
    prepare_data
    [ -f "$TRAIN_CFG" ] || { echo "ERROR: $TRAIN_CFG missing"; exit 1; }
    echo "$(date -Iseconds) job ${SLURM_JOB_ID:-local} commit $(git rev-parse HEAD)" >> "$OUT/git_commits.txt"

    # 3. Train (fresh, or full-state resume).
    if [ -f "$STATE" ]; then
        RESUME="--resume"
    else
        [ -f "$OUT/training_log.csv" ] && echo "WARNING: $OUT has a training log but no state file: starting from scratch"
        RESUME=""
    fi
    srun python -u scripts/run_training.py \
        --config "$TRAIN_CFG" \
        --dataset "$DATA" \
        --out "$OUT" \
        --num-epochs "$EPOCHS" \
        --seed "$SEED" \
        --mem-fraction "$MEM_FRACTION" $RESUME
fi

# 4. After the last epoch: the small-displacement test, then copy the results to $WORK.
if [ -f "$OUT/summary.json" ]; then
    dead_zone_check
fi
safekeep
echo "=== $RUN finished or stopped @ $(date) ==="
