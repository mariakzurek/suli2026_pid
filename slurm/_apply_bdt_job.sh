#!/bin/bash
# =============================================================================
# _apply_bdt_job.sh — SLURM worker script for one BDT-scoring task.
#
# NOT INVOKED DIRECTLY.  Submitted as an array task by submit_apply_bdt.sh.
# Each task picks one ROOT file from APPLY_FILE_LIST using $SLURM_ARRAY_TASK_ID,
# runs scripts/apply_bdt.py, and writes the scored ROOT to APPLY_OUTPUT_DIR.
#
# Environment variables (set by submit_apply_bdt.sh via --export):
#   APPLY_FILE_LIST     text file, one input ROOT path per line
#   APPLY_MODEL         path to model.joblib wrapper dict
#   APPLY_OUTPUT_DIR    directory for scored output ROOT files
#   APPLY_TREE          ROOT tree name (default: PhysicsEvents)
#   APPLY_OVERWRITE     "--overwrite" or "" (set by submit wrapper)
#   APPLY_THRESHOLD     (optional) global score threshold float
#   APPLY_THRESHOLD_CSV (optional) path to per-bin threshold CSV
#   APPLY_BATCH_SIZE    (optional) events per batch for large files
#   REPO_ROOT           sibling-parent of suli2026_pid/ (e.g. ~/CLAS/SULI/ on
#                       laptop, /work/clas12/$USER/SULI/ on ifarm). Computed
#                       dynamically by the submit wrapper from SCRIPT_DIR/../..;
#                       not a hardcoded path.
# =============================================================================

#SBATCH --job-name=apply_bdt
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/farm_out/%u/suli/apply_bdt_%j_%a.out
#SBATCH --error=/farm_out/%u/suli/apply_bdt_%j_%a.err
#SBATCH --requeue
#SBATCH --account=clas12

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# TMPDIR — required before any module or conda operation.
# The default SLURM TMPDIR (/scratch/slurm/<jobid>/) can fill
# the module lock-file quota.  /tmp is safe on JLab compute nodes.
# ─────────────────────────────────────────────────────────────
export TMPDIR=/tmp

# ─────────────────────────────────────────────────────────────
# MODULE INIT
# Bash batch/srun shells are non-login; the `module` function is
# not auto-sourced.  Source it explicitly before any module command.
#
# NOTE: Do NOT `module load clas12` here.  The CLAS12 module sets
# PYTHONPATH to its own site-packages, which overrides the conda env
# and causes numpy imports to fail (conda Python 3.11 ends up loading
# numpy from CLAS12's Python 3.13 path).  apply_bdt.py is a
# Python-only worker — it needs only the conda env, not CLAS12 ROOT.
# See suli-pid skill §3 "Do not module load clas12 in Python-only jobs."
# ─────────────────────────────────────────────────────────────
if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck source=/dev/null
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        # shellcheck source=/dev/null
        source /usr/share/Modules/init/bash
    else
        echo "WARNING: cannot locate modules init script — skipping module setup." >&2
    fi
fi

# Register the JLab module paths so `module` knows where to look,
# but do NOT load clas12 (see note above).
module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles 2>/dev/null || true
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles               2>/dev/null || true
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# CONDA — three-hook probe in priority order.
# (1) Personal miniforge at /work/ — recommended location.
# (2) Personal miniforge at $HOME — common fallback.
# (3) JLab system conda — last resort (conda info --base → /usr,
#     which looks wrong but is correct for this install).
# Never source ~/.bashrc — dotfiles are unavailable in batch shells.
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
: "${REPO_ROOT:?ERROR: REPO_ROOT not set — submit via submit_apply_bdt.sh}"
: "${APPLY_FILE_LIST:?ERROR: APPLY_FILE_LIST not set}"
: "${APPLY_MODEL:?ERROR: APPLY_MODEL not set}"
: "${APPLY_OUTPUT_DIR:?ERROR: APPLY_OUTPUT_DIR not set}"

APPLY_TREE="${APPLY_TREE:-PhysicsEvents}"
APPLY_OVERWRITE="${APPLY_OVERWRITE:-}"
APPLY_THRESHOLD="${APPLY_THRESHOLD:-}"
APPLY_THRESHOLD_CSV="${APPLY_THRESHOLD_CSV:-}"
APPLY_BATCH_SIZE="${APPLY_BATCH_SIZE:-}"

# ─────────────────────────────────────────────────────────────
# PYTHONPATH collision guard
# If (despite the no-clas12-module rule above) PYTHONPATH is set
# and contains paths that would shadow the conda env's packages,
# unset it so conda's site-packages take precedence.
# ─────────────────────────────────────────────────────────────
if [ -n "${PYTHONPATH:-}" ]; then
    echo "WARNING: PYTHONPATH is set (${PYTHONPATH}); unsetting to avoid" >&2
    echo "         shadowing the conda env's numpy/sklearn/uproot." >&2
    unset PYTHONPATH
fi

# ─────────────────────────────────────────────────────────────
# MOVE INTO REPO ROOT (required for Python package imports)
# ─────────────────────────────────────────────────────────────
cd "${REPO_ROOT}/suli2026_pid"

# ─────────────────────────────────────────────────────────────
# SCRATCH SPACE (per-task, cleaned on exit)
# ─────────────────────────────────────────────────────────────
if [ -d "/volatile/clas12/${USER}" ]; then
    SCRATCH_BASE="/volatile/clas12/${USER}/SULI/scratch"
else
    SCRATCH_BASE="/tmp/${USER}"
fi
mkdir -p "${SCRATCH_BASE}"
SCRATCH="${SCRATCH_BASE}/apply_bdt_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"
trap 'echo "Cleaning scratch: ${SCRATCH}"; rm -rf "${SCRATCH}"' EXIT

# ─────────────────────────────────────────────────────────────
# PICK THIS TASK'S INPUT FILE
# ─────────────────────────────────────────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
INPUT_ROOT="$(sed -n "$((TASK_ID + 1))p" "${APPLY_FILE_LIST}")"

if [ -z "${INPUT_ROOT}" ]; then
    echo "ERROR: No file at index ${TASK_ID} in ${APPLY_FILE_LIST}" >&2
    exit 1
fi
if [ ! -f "${INPUT_ROOT}" ]; then
    echo "ERROR: Input ROOT file not found: ${INPUT_ROOT}" >&2
    exit 1
fi

STEM="$(basename "${INPUT_ROOT}" .root)"
OUTPUT_ROOT="${APPLY_OUTPUT_DIR}/${STEM}.root"

mkdir -p "${APPLY_OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────
# PROVENANCE
# ─────────────────────────────────────────────────────────────
GIT_SHA="$(git -C "${REPO_ROOT}/suli2026_pid" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# ─────────────────────────────────────────────────────────────
# BUILD PYTHON COMMAND
# ─────────────────────────────────────────────────────────────
PYTHON_CMD=(
    python
    scripts/apply_bdt.py
    --input  "${INPUT_ROOT}"
    --model  "${APPLY_MODEL}"
    --output "${OUTPUT_ROOT}"
    --tree   "${APPLY_TREE}"
)

if [ -n "${APPLY_OVERWRITE}" ]; then
    PYTHON_CMD+=("${APPLY_OVERWRITE}")
fi
if [ -n "${APPLY_THRESHOLD}" ]; then
    PYTHON_CMD+=(--threshold "${APPLY_THRESHOLD}")
fi
if [ -n "${APPLY_THRESHOLD_CSV}" ]; then
    PYTHON_CMD+=(--threshold-csv "${APPLY_THRESHOLD_CSV}")
fi
if [ -n "${APPLY_BATCH_SIZE}" ]; then
    PYTHON_CMD+=(--batch-size "${APPLY_BATCH_SIZE}")
fi

# ─────────────────────────────────────────────────────────────
# START BANNER
# ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "apply_bdt task start"
echo "host    : $(hostname)"
echo "cwd     : $(pwd)"
echo "python  : $(which python)"
echo "git sha : ${GIT_SHA}"
echo "task id : ${TASK_ID}"
echo "input   : ${INPUT_ROOT}"
echo "output  : ${OUTPUT_ROOT}"
echo "command : ${PYTHON_CMD[*]}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
"${PYTHON_CMD[@]}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "apply_bdt task DONE"
echo "output: ${OUTPUT_ROOT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
