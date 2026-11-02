#!/bin/bash
# Single place for the Vista environment and paths. Every job in vista/ sources this file.
# Jobs must be submitted from the repository root (cd "$WORK/projects/<repo>"; sbatch vista/<job>),
# because SLURM runs a copy of the job script and only $SLURM_SUBMIT_DIR points back to the clone.
#
# Env: an existing aarch64 conda env (torch 2.5.1, CUDA 12.4, numpy 1.26); vista/smoke.slurm
# checks that it still imports.

source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/envs/raftcorr"

export PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Two runs share this file. Pick one when submitting (default: xdic):
#   sbatch vista/train.slurm                                                        -> RAFT-XDIC
#   sbatch --job-name=control --export=ALL,RUN_VARIANT=control vista/train.slurm    -> RAFT-DIC
# Both training sets have 7,200 + 800 pairs and share the same 6,400 smooth-field pairs:
#   xdic    : + 1,600 mode II crack pairs (configs/dataset_gen/mode2_7px_both_senses_split.yaml)
#   control : + 1,600 smooth-field pairs (same speckle and displacement range), the matched control
# Labels are reference-grid displacements (where each reference pixel went); pixels without a unique
# label, where the synthetic warp folds, are excluded by the valid/ masks shipped with the data.
# Each set is uploaded once as a tarball. The tarball stays on $WORK (kept); the extracted copy lives
# on $SCRATCH (purged after ~10 idle days) and is re-extracted automatically when missing.
export RUN_VARIANT="${RUN_VARIANT:-xdic}"
case "$RUN_VARIANT" in
    xdic)
        export DATASET="mode2_mixed_both_senses_split_refgrid"
        export RUN="xdic_split_refgrid_onecycle_seed0" ;;
    control)
        export DATASET="smooth_mixed_refgrid"
        export RUN="control_refgrid_onecycle_seed0" ;;
    *)
        echo "ERROR: RUN_VARIANT must be xdic or control, got '$RUN_VARIANT'"
        return 1 2>/dev/null || exit 1 ;;
esac
export TARBALL="$WORK/raft-xdic/data/${DATASET}.tar.gz"
export TARBALL_SHA256_FILE="$PROJECT_DIR/vista/${DATASET}.tar.gz.sha256"
export DATA_ROOT="$SCRATCH/raft-xdic/data"
export DATA="$DATA_ROOT/$DATASET"

# Recipe (both runs): configs/train_F_onecycle.yaml, seed 0, from scratch, 30 epochs x 3,600 steps, batch 2.
export TRAIN_CFG="configs/train_F_onecycle.yaml"
export EPOCHS=30
export SEED=0
export MEM_FRACTION=0.8          # cap PyTorch at 80 % of the GPU memory
export RUNS_ROOT="$SCRATCH/raft-xdic/runs"
export SAFE_ROOT="$WORK/raft-xdic/runs"   # curated copies, not purged

mkdir -p "$DATA_ROOT" "$RUNS_ROOT" "$SAFE_ROOT" "$PROJECT_DIR/vista/logs"

# Extract the dataset from the tarball on $WORK when the $SCRATCH copy is missing or incomplete.
# split.json is the completion marker (combine_datasets.py writes it last).
prepare_data() {
    if [ -f "$DATA/split.json" ] && [ -f "$DATA/.extract_ok" ]; then
        echo "[data] $DATA present"
        return 0
    fi
    [ -f "$TARBALL" ] || { echo "ERROR: dataset tarball missing: $TARBALL (upload it first)"; return 1; }
    [ -f "$TARBALL_SHA256_FILE" ] || { echo "ERROR: checksum file missing: $TARBALL_SHA256_FILE"; return 1; }
    echo "[data] checking $TARBALL"
    local want got
    want=$(awk '{print $1}' "$TARBALL_SHA256_FILE")
    got=$(sha256sum "$TARBALL" | awk '{print $1}')
    [ "$want" = "$got" ] || { echo "ERROR: checksum mismatch for $TARBALL (want $want, got $got): re-upload it"; return 1; }
    if [ -e "$DATA" ]; then                       # an incomplete copy: set aside, not deleted
        mv "$DATA" "${DATA}_incomplete_$(date +%Y%m%d_%H%M%S)"
        echo "[data] set aside an incomplete copy of $DATA"
    fi
    echo "[data] extracting to $DATA_ROOT"
    tar -xzf "$TARBALL" -C "$DATA_ROOT"
    local d n
    for d in ref def u v; do
        n=$(find "$DATA/$d" -type f | wc -l)
        [ "$n" -eq 8000 ] || { echo "ERROR: $DATA/$d has $n files, expected 8000"; return 1; }
    done
    [ -f "$DATA/split.json" ] || { echo "ERROR: $DATA/split.json missing after extraction"; return 1; }
    touch "$DATA/.extract_ok"
    echo "[data] ready: 8000 pairs in $DATA"
}
