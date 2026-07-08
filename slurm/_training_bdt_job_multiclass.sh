#!/bin/bash
# =============================================================================
# _training_bdt_job_multiclass.sh — SLURM worker script for BDT training (fixed conda)
# =============================================================================

#SBATCH --job-name=training_bdt_multiclass
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/farm_out/%u/suli/training_bdt_%j.out
#SBATCH --error=/farm_out/%u/suli/training_bdt_%j.err
#SBATCH --requeue
#SBATCH --account=clas12

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# TMPDIR (safe for SLURM/module edge cases)
# ─────────────────────────────────────────────────────────────
export TMPDIR=/tmp

# ─────────────────────────────────────────────────────────────
# CONDA (FIXED: use system conda hook, no conda.sh)
# ─────────────────────────────────────────────────────────────

echo "Initializing conda..."

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found"
    exit 1
fi

eval "$(/usr/bin/conda shell.bash hook)"

conda activate suli2026_pid

echo "Conda activated: $CONDA_PREFIX"
echo "Python: $(which python)"

# ─────────────────────────────────────────────────────────────
# INPUTS FROM SBATCH EXPORT
# ─────────────────────────────────────────────────────────────

: "${REPO_ROOT:?ERROR: REPO_ROOT not set}"
: "${DATASET_DIR:?ERROR: DATASET_DIR not set}"
: "${MODEL_DIR:?ERROR: MODEL_DIR not set}"
: "${FEATURES_FILE:?ERROR: FEATURES_FILE not set}"

REWEIGHT_MAP="${REWEIGHT_MAP:-}"

# ─────────────────────────────────────────────────────────────
# MOVE INTO REPO
# ─────────────────────────────────────────────────────────────

cd "${REPO_ROOT}/suli2026_pid"

# ─────────────────────────────────────────────────────────────
# SCRATCH SPACE
# ─────────────────────────────────────────────────────────────

if [ -d "/volatile/clas12/${USER}" ]; then
    SCRATCH_BASE="/volatile/clas12/${USER}/SULI/scratch"
else
    SCRATCH_BASE="/tmp/${USER}"
fi

mkdir -p "${SCRATCH_BASE}"
SCRATCH="${SCRATCH_BASE}/training_bdt_${SLURM_JOB_ID:-local}"
rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"

trap 'echo "Cleaning scratch: ${SCRATCH}"; rm -rf "${SCRATCH}"' EXIT

# ─────────────────────────────────────────────────────────────
# PROVENANCE
# ─────────────────────────────────────────────────────────────

GIT_SHA="$(git -C "${REPO_ROOT}/suli2026_pid" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# ─────────────────────────────────────────────────────────────
# OUTPUT DIR
# ─────────────────────────────────────────────────────────────

mkdir -p "${MODEL_DIR}"

# ─────────────────────────────────────────────────────────────
# RUN TRAINING
# ─────────────────────────────────────────────────────────────

PYTHON_CMD=(
    python
    scripts/training/train_bdt_multiclass.py
    --dataset-dir "${DATASET_DIR}"
    --features-file "${FEATURES_FILE}"
    --outdir "${MODEL_DIR}"
    --overwrite
)

if [ -n "${REWEIGHT_MAP}" ]; then
    PYTHON_CMD+=(--reweight-map "${REWEIGHT_MAP}")
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BDT training job start"
echo "host: $(hostname)"
echo "git: ${GIT_SHA}"
echo "python: $(which python)"
echo "features: ${FEATURES_FILE}"
echo "command: ${PYTHON_CMD[*]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

"${PYTHON_CMD[@]}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "BDT training DONE"
echo "model: ${MODEL_DIR}/model.joblib"
echo "features: ${FEATURES_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"