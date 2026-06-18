#!/bin/bash
# =============================================================================
# _training_bdt_job.sh — SLURM worker script for BDT training.
#
# This script is NOT meant to be invoked directly by the user.  It is
# submitted by submit_training_bdt.sh via sbatch.  The underscore prefix
# marks it as internal.
#
# Path variables are injected by submit_training_bdt.sh via --export:
#   REPO_ROOT    — ~/CLAS/SULI/ (parent of both suli2026_pid/ and clas12_analysis_software/)
#   DATASET_DIR  — directory produced by build_dataset.py (contains manifest.json)
#   MODEL_DIR    — output directory for model.joblib and plots
#   REWEIGHT_MAP — (optional) path to .npz reweight map; may be unset
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SLURM DIRECTIVES
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=training_bdt
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/farm_out/%u/suli/training_bdt_%j.out
#SBATCH --error=/farm_out/%u/suli/training_bdt_%j.err
#SBATCH --requeue

# Account: clas12.  Cluster default partition is production; no --partition
# directive is needed.
#SBATCH --account=clas12

# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── TMPDIR override ───────────────────────────────────────────────────────────
# SLURM sets TMPDIR to a per-job scratch area where the Modules system cannot
# write its lockfiles ("no space left on device" even though df shows free
# space).  Override to /tmp (compute-node local, ~8 GB, ample for lockfiles).
# Must be set BEFORE any module commands.  Confirmed on ifarm 2026-06.
export TMPDIR=/tmp

# ── Module environment ────────────────────────────────────────────────────────
# Bash subshells (including SLURM batch jobs) don't run the login init that
# defines the `module` function.  Source it explicitly.
if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck source=/dev/null
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        # shellcheck source=/dev/null
        source /usr/share/Modules/init/bash
    else
        echo "ERROR: cannot locate modules init script; tried"
        echo "  /etc/profile.d/modules.sh"
        echo "  /usr/share/Modules/init/bash"
        echo "Set MODULESHOME or source the appropriate init manually."
        exit 1
    fi
fi
module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module load clas12

# ── Defensive conda activation ────────────────────────────────────────────────
# Probe the three known hook locations.  Never source ~/.bashrc — that runs
# interactive-shell logic (aliases, PS1 changes, etc.) that can fail or
# produce garbage in a batch log.  Error with a concrete pointer if none found.
CONDA_ACTIVATED=0
for CONDA_BASE in \
    "/work/clas12/${USER}/miniconda3" \
    "${HOME}/miniconda3" \
    "/apps/anaconda3"
do
    CONDA_HOOK="${CONDA_BASE}/etc/profile.d/conda.sh"
    if [ -f "${CONDA_HOOK}" ]; then
        # shellcheck source=/dev/null
        source "${CONDA_HOOK}"
        if conda activate suli2026_pid 2>/dev/null; then
            echo "Conda activated: suli2026_pid (hook: ${CONDA_HOOK})"
            CONDA_ACTIVATED=1
            break
        fi
    fi
done

if [ "${CONDA_ACTIVATED}" -eq 0 ]; then
    echo "ERROR: Could not activate conda env 'suli2026_pid'."
    echo "       Checked hooks at:"
    echo "         /work/clas12/${USER}/miniconda3/etc/profile.d/conda.sh"
    echo "         \${HOME}/miniconda3/etc/profile.d/conda.sh"
    echo "         /apps/anaconda3/etc/profile.d/conda.sh"
    echo "       See cooper_day1_and_week1.md §4f for setup instructions."
    exit 1
fi

# ── Validate injected paths ───────────────────────────────────────────────────
: "${REPO_ROOT:?ERROR: REPO_ROOT not set — run via submit_training_bdt.sh}"
: "${DATASET_DIR:?ERROR: DATASET_DIR not set — run via submit_training_bdt.sh}"
: "${MODEL_DIR:?ERROR: MODEL_DIR not set — run via submit_training_bdt.sh}"
# REWEIGHT_MAP is optional; default to empty string if unset.
REWEIGHT_MAP="${REWEIGHT_MAP:-}"

# ── Change to repo root ───────────────────────────────────────────────────────
# Required so `from scripts.baseline_chi2pid import ...` resolves correctly
# in evaluate.py and any other script that imports from the project package.
cd "${REPO_ROOT}/suli2026_pid"

# ── Scratch directory (cleanup on exit) ───────────────────────────────────────
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

# ── Git SHA for provenance ────────────────────────────────────────────────────
GIT_SHA="$(git -C "${REPO_ROOT}/suli2026_pid" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "${MODEL_DIR}"

# ── Build Python command as a bash array ─────────────────────────────────────
# Using an array avoids word-splitting on paths with spaces and makes the
# optional --reweight-map append cleanly without inline ${VAR:+...} expansion.
PYTHON_CMD=(
    python
    scripts/training/train_bdt.py
    --dataset-dir "${DATASET_DIR}"
    --outdir      "${MODEL_DIR}"
    --overwrite
)

# Append optional reweight map only if the variable is non-empty.
if [ -n "${REWEIGHT_MAP}" ]; then
    PYTHON_CMD+=(--reweight-map "${REWEIGHT_MAP}")
fi

# ── Start banner ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BDT training job start"
echo "  hostname   : $(hostname)"
echo "  cwd        : $(pwd)"
echo "  python     : $(which python)"
echo "  git_sha    : ${GIT_SHA}"
echo "  command    : ${PYTHON_CMD[*]}"
echo "  REPO_ROOT  : ${REPO_ROOT}"
echo "  DATASET_DIR: ${DATASET_DIR}"
echo "  MODEL_DIR  : ${MODEL_DIR}"
echo "  REWEIGHT_MAP: ${REWEIGHT_MAP:-<none>}"
echo "  scratch    : ${SCRATCH}"
echo "  time       : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Execute ───────────────────────────────────────────────────────────────────
"${PYTHON_CMD[@]}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BDT training job DONE"
echo "  model.joblib : ${MODEL_DIR}/model.joblib"
echo "  time         : $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
