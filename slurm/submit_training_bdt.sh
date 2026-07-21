#!/bin/bash
# =============================================================================
# submit_training_bdt.sh — Submit the BDT training job as a single SLURM job.
#
# Usage (from the suli2026_pid/ repo root on ifarm):
#   ./slurm/submit_training_bdt.sh [OPTIONS]
#
# Options:
#   --dataset-dir PATH   Dataset directory from build_dataset.py
#                        (default: /volatile/clas12/$USER/SULI/datasets/v01)
#   --model-dir PATH     Output directory for model.joblib and plots
#                        (default: /volatile/clas12/$USER/SULI/models/v01)
#   --features-file PATH Feature list for train_bdt.py (REQUIRED)
#   --reweight-map PATH  Optional .npz reweight map; omit for unweighted run
#   --time HH:MM:SS      Wall-time limit (default: 01:00:00)
#   --mem N[G]           Memory request  (default: 8G)
#   --cpus N             CPUs per task   (default: 32)
#   --dry-run            Print the sbatch command without submitting
#
# Account: clas12 (no --partition; cluster default is production).
# Logs:    /farm_out/$USER/suli/training_bdt_<jobid>.{out,err}
# =============================================================================

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/_training_bdt_job.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
DATASET_DIR="/volatile/clas12/${USER}/SULI/datasets/v01"
MODEL_DIR="/volatile/clas12/${USER}/SULI/models/v01"
FEATURES_FILE=""
REWEIGHT_MAP=""
JOB_TIME="01:00:00"
JOB_MEM="8G"
JOB_CPUS="32"
DRY_RUN=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir)   DATASET_DIR="$2";   shift 2 ;;
        --model-dir)     MODEL_DIR="$2";     shift 2 ;;
        --features-file) FEATURES_FILE="$2"; shift 2 ;;
        --reweight-map)  REWEIGHT_MAP="$2";  shift 2 ;;
        --time)          JOB_TIME="$2";      shift 2 ;;
        --mem)           JOB_MEM="$2";       shift 2 ;;
        --cpus)          JOB_CPUS="$2";      shift 2 ;;
        --dry-run)       DRY_RUN=1;          shift ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "Usage: $0 [--dataset-dir PATH] [--model-dir PATH] [--features-file PATH]" >&2
            echo "          [--reweight-map PATH] [--time HH:MM:SS]" >&2
            echo "          [--mem NG] [--cpus N] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# ── Required arguments ────────────────────────────────────────────────────────
if [ -z "${FEATURES_FILE}" ]; then
    echo "ERROR: --features-file is required" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "  ./slurm/submit_training_bdt.sh \\" >&2
    echo "      --features-file scripts/training/features_tier1.txt" >&2
    exit 1
fi

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "${WORKER_SCRIPT}" ]; then
    echo "ERROR: Worker script not found: ${WORKER_SCRIPT}"
    exit 1
fi

# Verify features file exists.
if [ ! -f "${FEATURES_FILE}" ]; then
    echo "ERROR: Features file not found: ${FEATURES_FILE}"
    exit 1
fi

# Skip manifest check in dry-run — useful for testing the command without real data.
if [ "${DRY_RUN}" -eq 0 ]; then
    MANIFEST="${DATASET_DIR}/manifest.json"
    if [ ! -f "${MANIFEST}" ]; then
        echo "ERROR: Dataset manifest not found: ${MANIFEST}"
        echo "       Run build_dataset.py first."
        exit 1
    fi
fi

# ── Create log directory (sbatch %u/%j tokens require it to exist) ────────────
FARM_LOG_DIR="/farm_out/${USER}/suli"

if [ "${DRY_RUN}" -eq 0 ]; then
    mkdir -p "${FARM_LOG_DIR}"
fi

# ── Build export list ─────────────────────────────────────────────────────────
EXPORT_VARS="ALL"
EXPORT_VARS+=",REPO_ROOT=${REPO_ROOT}"
EXPORT_VARS+=",DATASET_DIR=${DATASET_DIR}"
EXPORT_VARS+=",MODEL_DIR=${MODEL_DIR}"
EXPORT_VARS+=",FEATURES_FILE=${FEATURES_FILE}"

if [ -n "${REWEIGHT_MAP}" ]; then
    EXPORT_VARS+=",REWEIGHT_MAP=${REWEIGHT_MAP}"
fi

# ── Assemble sbatch command ───────────────────────────────────────────────────
SBATCH_CMD=(
    sbatch
    --job-name=training_bdt
    --account=clas12
    --time="${JOB_TIME}"
    --mem="${JOB_MEM}"
    --cpus-per-task="${JOB_CPUS}"
    --output="${FARM_LOG_DIR}/training_bdt_%j.out"
    --error="${FARM_LOG_DIR}/training_bdt_%j.err"
    --export="${EXPORT_VARS}"
    "${WORKER_SCRIPT}"
)

# ── Submit or dry-run ─────────────────────────────────────────────────────────
echo ""
echo "BDT training job configuration:"
echo "  REPO_ROOT     : ${REPO_ROOT}"
echo "  DATASET_DIR   : ${DATASET_DIR}"
echo "  MODEL_DIR     : ${MODEL_DIR}"
echo "  FEATURES_FILE : ${FEATURES_FILE}"
echo "  REWEIGHT_MAP  : ${REWEIGHT_MAP:-<none — unweighted run>}"
echo "  time          : ${JOB_TIME}"
echo "  mem           : ${JOB_MEM}"
echo "  cpus          : ${JOB_CPUS}"
echo "  worker        : ${WORKER_SCRIPT}"
echo "  logs          : ${FARM_LOG_DIR}/training_bdt_<jobid>.{out,err}"
echo ""

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  --dry-run: would run:"
    echo "  ${SBATCH_CMD[*]}"
    echo ""
    exit 0
fi

JOB_ID=$("${SBATCH_CMD[@]}" --parsable)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Submitted BDT training job: ${JOB_ID}"
echo ""
echo "  Follow-up commands:"
echo "    squeue -u \$USER"
echo "    tail -f ${FARM_LOG_DIR}/training_bdt_${JOB_ID}.out"
echo "    scancel ${JOB_ID}"
echo "    sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"