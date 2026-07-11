#!/bin/bash
# =============================================================================
# submit_apply_bdt.sh — Submit a BDT-scoring array job over a directory of ROOT files.
#
# Usage (from the suli2026_pid/ repo root on ifarm):
#   ./slurm/submit_apply_bdt.sh [OPTIONS]
#
# Required options:
#   --model PATH          model.joblib wrapper dict from train_bdt.py
#   --output-dir PATH     directory to write scored ROOT files
#
# Input (exactly one of):
#   --input-dir PATH      directory of input ROOT files (one task per file)
#   --file-list PATH      text file listing input ROOT paths (one per line)
#
# Threshold options (mutually exclusive; omit for bdt_score only):
#   --threshold FLOAT     global BDT score threshold; writes bdt_pass = score > t
#   --threshold-csv PATH  per-(p,theta)-bin CSV (p_low,p_high,theta_low,theta_high,t_optimal)
#
# Optional:
#   --tree NAME           ROOT tree name (default: PhysicsEvents)
#   --batch-size N        events per batch for large files (default: no batching)
#   --force               reprocess files that already have an output (default: skip existing)
#   --time HH:MM:SS       wall-time limit per task (default: 00:30:00)
#   --mem  NG             memory per task (default: 8G)
#   --cpus N              CPUs per task (default: 4)
#   --max-concurrent N    max simultaneous array tasks (default: 50)
#   --dry-run             print sbatch command without submitting
#
# Account: clas12 (no --partition; cluster default is production).
# Logs:    /farm_out/$USER/suli/apply_bdt_<jobid>_<taskid>.{out,err}
#
# Resume-friendly by default: tasks whose output ROOT already exists are skipped.
# Pass --force to reprocess all files.
#
# Examples:
#   # Score all data_v01 files, no threshold (bdt_score branch only):
#   ./slurm/submit_apply_bdt.sh \
#       --input-dir  /volatile/clas12/$USER/SULI/data_v01 \
#       --model      /work/clas12/$USER/SULI/models/tier1_v01/model.joblib \
#       --output-dir /volatile/clas12/$USER/SULI/scored_data_v01
#
#   # With per-bin thresholds from evaluate.py:
#   ./slurm/submit_apply_bdt.sh \
#       --input-dir    /volatile/clas12/$USER/SULI/data_v01 \
#       --model        /work/clas12/$USER/SULI/models/tier1_v01/model.joblib \
#       --output-dir   /volatile/clas12/$USER/SULI/scored_data_v01 \
#       --threshold-csv /volatile/clas12/$USER/SULI/eval/v01/per_bin_thresholds.csv
#
#   # Smoke test on 2 files:
#   ls /volatile/clas12/$USER/SULI/mc_v01/*.root | head -2 > /tmp/smoke_files.txt
#   ./slurm/submit_apply_bdt.sh \
#       --file-list  /tmp/smoke_files.txt \
#       --model      /work/clas12/$USER/SULI/models/tier1_v01/model.joblib \
#       --output-dir /volatile/clas12/$USER/SULI/scored_mc_v01 \
#       --dry-run
# =============================================================================

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/_apply_bdt_job.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
INPUT_DIR=""
FILE_LIST_ARG=""
MODEL_PATH=""
OUTPUT_DIR=""
THRESHOLD_ARG=""
THRESHOLD_CSV_ARG=""
TREE_NAME="PhysicsEvents"
BATCH_SIZE_ARG=""
FORCE=0
JOB_TIME="00:30:00"
JOB_MEM="8G"
JOB_CPUS="4"
MAX_CONCURRENT=50
DRY_RUN=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir)       INPUT_DIR="$2";          shift 2 ;;
        --file-list)       FILE_LIST_ARG="$2";      shift 2 ;;
        --model)           MODEL_PATH="$2";         shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";         shift 2 ;;
        --threshold)       THRESHOLD_ARG="$2";      shift 2 ;;
        --threshold-csv)   THRESHOLD_CSV_ARG="$2";  shift 2 ;;
        --tree)            TREE_NAME="$2";           shift 2 ;;
        --batch-size)      BATCH_SIZE_ARG="$2";     shift 2 ;;
        --force|-f)        FORCE=1;                 shift ;;
        --time)            JOB_TIME="$2";           shift 2 ;;
        --mem)             JOB_MEM="$2";            shift 2 ;;
        --cpus)            JOB_CPUS="$2";           shift 2 ;;
        --max-concurrent)  MAX_CONCURRENT="$2";     shift 2 ;;
        --dry-run)         DRY_RUN=1;               shift ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "" >&2
            echo "Usage: $0 --model PATH --output-dir PATH" >&2
            echo "          (--input-dir PATH | --file-list PATH)" >&2
            echo "          [--threshold FLOAT | --threshold-csv PATH]" >&2
            echo "          [--tree NAME] [--batch-size N] [--force]" >&2
            echo "          [--time HH:MM:SS] [--mem NG] [--cpus N]" >&2
            echo "          [--max-concurrent N] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

# ── Validate required arguments ───────────────────────────────────────────────
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: --model is required." >&2
    exit 1
fi
if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: --output-dir is required." >&2
    exit 1
fi
if [ -z "${INPUT_DIR}" ] && [ -z "${FILE_LIST_ARG}" ]; then
    echo "ERROR: Exactly one of --input-dir or --file-list is required." >&2
    exit 1
fi
if [ -n "${INPUT_DIR}" ] && [ -n "${FILE_LIST_ARG}" ]; then
    echo "ERROR: --input-dir and --file-list are mutually exclusive." >&2
    exit 1
fi
if [ -n "${THRESHOLD_ARG}" ] && [ -n "${THRESHOLD_CSV_ARG}" ]; then
    echo "ERROR: --threshold and --threshold-csv are mutually exclusive." >&2
    exit 1
fi

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "${WORKER_SCRIPT}" ]; then
    echo "ERROR: Worker script not found: ${WORKER_SCRIPT}" >&2
    exit 1
fi

if [ "${DRY_RUN}" -eq 0 ]; then
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: Model file not found: ${MODEL_PATH}" >&2
        exit 1
    fi
    if [ -n "${THRESHOLD_CSV_ARG}" ] && [ ! -f "${THRESHOLD_CSV_ARG}" ]; then
        echo "ERROR: --threshold-csv not found: ${THRESHOLD_CSV_ARG}" >&2
        exit 1
    fi
fi

# ── Build file list ──────────────────────────────────────────────────────────
FARM_LOG_DIR="/farm_out/${USER}/suli"
GENERATED_LIST="${SCRIPT_DIR}/_apply_bdt_file_list.txt"

if [ -n "${INPUT_DIR}" ]; then
    if [ "${DRY_RUN}" -eq 0 ] && [ ! -d "${INPUT_DIR}" ]; then
        echo "ERROR: --input-dir not found: ${INPUT_DIR}" >&2
        exit 1
    fi
    echo "Scanning ${INPUT_DIR} for ROOT files ..."
    find -L "${INPUT_DIR}" -name '*.root' -type f 2>/dev/null | sort > "${GENERATED_LIST}.all"
else
    if [ "${DRY_RUN}" -eq 0 ] && [ ! -f "${FILE_LIST_ARG}" ]; then
        echo "ERROR: --file-list not found: ${FILE_LIST_ARG}" >&2
        exit 1
    fi
    cp "${FILE_LIST_ARG}" "${GENERATED_LIST}.all"
fi

TOTAL_ALL=$(wc -l < "${GENERATED_LIST}.all")

if [ "${FORCE}" -eq 1 ]; then
    echo "  --force: processing all ${TOTAL_ALL} input files (existing outputs will be overwritten)."
    cp "${GENERATED_LIST}.all" "${GENERATED_LIST}"
    SKIP_EXISTING_FLAG="--overwrite"
else
    # Skip files whose output ROOT already exists.
    true > "${GENERATED_LIST}"
    skipped=0
    while IFS= read -r root_path; do
        stem=$(basename "${root_path}" .root)
        if [ -f "${OUTPUT_DIR}/${stem}.root" ]; then
            skipped=$((skipped + 1))
        else
            echo "${root_path}" >> "${GENERATED_LIST}"
        fi
    done < "${GENERATED_LIST}.all"
    echo "  Found ${TOTAL_ALL} input ROOT files."
    echo "  ${skipped} already have outputs (skipped). Pass --force to reprocess."
    SKIP_EXISTING_FLAG=""
fi
rm -f "${GENERATED_LIST}.all"

N_FILES=$(wc -l < "${GENERATED_LIST}")
if [ "${N_FILES}" -eq 0 ]; then
    echo "Nothing to do — all input files already have outputs (or no input files found)."
    echo "Pass --force to reprocess everything."
    exit 0
fi

LAST_IDX=$((N_FILES - 1))

echo "File list written: ${GENERATED_LIST}"
echo "  Files queued for SLURM: ${N_FILES}"
echo "  Array indices: 0-${LAST_IDX}"

# ── Create output and log directories ────────────────────────────────────────
if [ "${DRY_RUN}" -eq 0 ]; then
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${FARM_LOG_DIR}"
fi

# ── Build export list ─────────────────────────────────────────────────────────
EXPORT_VARS="ALL"
EXPORT_VARS+=",REPO_ROOT=${REPO_ROOT}"
EXPORT_VARS+=",APPLY_FILE_LIST=${GENERATED_LIST}"
EXPORT_VARS+=",APPLY_MODEL=${MODEL_PATH}"
EXPORT_VARS+=",APPLY_OUTPUT_DIR=${OUTPUT_DIR}"
EXPORT_VARS+=",APPLY_TREE=${TREE_NAME}"
EXPORT_VARS+=",APPLY_OVERWRITE=${SKIP_EXISTING_FLAG}"

if [ -n "${THRESHOLD_ARG}" ]; then
    EXPORT_VARS+=",APPLY_THRESHOLD=${THRESHOLD_ARG}"
fi
if [ -n "${THRESHOLD_CSV_ARG}" ]; then
    EXPORT_VARS+=",APPLY_THRESHOLD_CSV=${THRESHOLD_CSV_ARG}"
fi
if [ -n "${BATCH_SIZE_ARG}" ]; then
    EXPORT_VARS+=",APPLY_BATCH_SIZE=${BATCH_SIZE_ARG}"
fi

# ── Assemble sbatch command ───────────────────────────────────────────────────
SBATCH_CMD=(
    sbatch
    --job-name=apply_bdt
    --account=clas12
    --array="0-${LAST_IDX}%${MAX_CONCURRENT}"
    --time="${JOB_TIME}"
    --mem="${JOB_MEM}"
    --cpus-per-task="${JOB_CPUS}"
    --output="${FARM_LOG_DIR}/apply_bdt_%j_%a.out"
    --error="${FARM_LOG_DIR}/apply_bdt_%j_%a.err"
    --requeue
    --export="${EXPORT_VARS}"
    "${WORKER_SCRIPT}"
)

# ── Print configuration summary ───────────────────────────────────────────────
echo ""
echo "BDT apply job configuration:"
echo "  REPO_ROOT      : ${REPO_ROOT}"
echo "  model          : ${MODEL_PATH}"
echo "  output dir     : ${OUTPUT_DIR}"
echo "  tree           : ${TREE_NAME}"
if [ -n "${THRESHOLD_ARG}" ]; then
    echo "  threshold      : ${THRESHOLD_ARG} (global)"
elif [ -n "${THRESHOLD_CSV_ARG}" ]; then
    echo "  threshold csv  : ${THRESHOLD_CSV_ARG} (per-bin)"
else
    echo "  threshold      : none (bdt_score only)"
fi
if [ -n "${BATCH_SIZE_ARG}" ]; then
    echo "  batch size     : ${BATCH_SIZE_ARG}"
fi
echo "  tasks          : ${N_FILES} (array 0–${LAST_IDX}, ≤${MAX_CONCURRENT} concurrent)"
echo "  time / mem     : ${JOB_TIME} / ${JOB_MEM}"
echo "  cpus           : ${JOB_CPUS}"
echo "  worker         : ${WORKER_SCRIPT}"
echo "  logs           : ${FARM_LOG_DIR}/apply_bdt_<jobid>_<taskid>.{out,err}"
echo ""

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  --dry-run: would run:"
    echo "  ${SBATCH_CMD[*]}"
    echo ""
    exit 0
fi

JOB_ID=$("${SBATCH_CMD[@]}" --parsable)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Submitted apply_bdt array job: ${JOB_ID}"
echo "  Tasks: ${N_FILES} (array 0–${LAST_IDX}, ≤${MAX_CONCURRENT} concurrent)"
echo ""
echo "  Follow-up commands:"
echo "    squeue -u \$USER"
echo "    tail -f ${FARM_LOG_DIR}/apply_bdt_${JOB_ID}_0.out"
echo "    scancel ${JOB_ID}"
echo "    sacct -j ${JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
