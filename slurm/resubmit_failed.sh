#!/bin/bash
# =============================================================================
# resubmit_failed.sh — resubmit only the failed tasks from a previous array job
#
# Usage:
#   ./slurm/resubmit_failed.sh <ORIG_JOBID> <mc|data>
#
# Examples:
#   ./slurm/resubmit_failed.sh 1234567 mc
#   ./slurm/resubmit_failed.sh 1234567 data
#
# What it does:
#   1. Queries sacct for the original job to find which array indices failed
#      (FAILED, TIMEOUT, or CANCELLED state).
#   2. Builds a comma-separated index list.
#   3. Submits a new sbatch array restricted to exactly those indices, using
#      the SAME file list written at the original submit time
#      (slurm/_mc_file_list.txt or slurm/_data_file_list.txt).
#
# Because the index-to-file mapping is fixed by the file list, each re-submitted
# task processes the same HIPO file it failed on.
#
# If you need to resubmit after regenerating the file list (e.g. you added more
# files), use submit_mc.sh / submit_data.sh instead.
# =============================================================================

set -euo pipefail

ORIG_JOBID="${1:?usage: resubmit_failed.sh <orig_jobid> <mc|data>}"
SAMPLE="${2:?usage: resubmit_failed.sh <orig_jobid> <mc|data>}"

if [ "${SAMPLE}" != "mc" ] && [ "${SAMPLE}" != "data" ]; then
    echo "ERROR: SAMPLE must be 'mc' or 'data', got: ${SAMPLE}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE_LIST="${SCRIPT_DIR}/_${SAMPLE}_file_list.txt"
ARRAY_SCRIPT="${SCRIPT_DIR}/_pid_training_array.sh"

if [ ! -f "${FILE_LIST}" ]; then
    echo "ERROR: File list not found: ${FILE_LIST}"
    echo "       The file list is created at submit time.  If it was deleted,"
    echo "       re-run submit_${SAMPLE}.sh (but that will re-submit ALL tasks)."
    exit 1
fi

# ── Identify failed array indices via sacct ───────────────────────────────────
echo "Querying sacct for failed tasks in job ${ORIG_JOBID} ..."

FAILED_INDICES=()
while IFS='|' read -r jobid state; do
    # Skip parent job row and sub-steps (.batch, .extern)
    if [[ "${jobid}" != *_* ]]; then continue; fi
    if [[ "${jobid}" == *.batch ]] || [[ "${jobid}" == *.extern ]]; then continue; fi

    state_clean="${state%% *}"
    if [[ "${state_clean}" == "FAILED" || "${state_clean}" == "TIMEOUT" || "${state_clean}" == "CANCELLED" ]]; then
        arr_idx="${jobid##*_}"
        FAILED_INDICES+=("${arr_idx}")
    fi
done < <(sacct -j "${ORIG_JOBID}" \
    --format=JobID,State \
    --parsable2 --noheader 2>/dev/null || true)

if [ "${#FAILED_INDICES[@]}" -eq 0 ]; then
    echo "No failed, timed-out, or cancelled tasks found for job ${ORIG_JOBID}."
    echo "Nothing to resubmit."
    exit 0
fi

FAILED_LIST=$(IFS=','; echo "${FAILED_INDICES[*]}")
echo "  Failed array indices (${#FAILED_INDICES[@]} tasks): ${FAILED_LIST}"

# ── Confirm output directory (same as original submission) ────────────────────
OUTPUT_DIR="/volatile/clas12/${USER}/SULI/${SAMPLE}_v01"

echo ""
echo "About to submit:"
echo "  sbatch --array=${FAILED_LIST}%50 _pid_training_array.sh ${SAMPLE}"
echo "  File list  : ${FILE_LIST}"
echo "  Output dir : ${OUTPUT_DIR}"
echo ""
read -rp "Proceed? [y/N] " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# ── Submit ───────────────────────────────────────────────────────────────────
# --output/--error go to /farm_out/ matching the _pid_training_array.sh directives.
NEW_JOB_ID=$(sbatch --parsable \
    --array="${FAILED_LIST}%50" \
    "${ARRAY_SCRIPT}" "${SAMPLE}")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Resubmitted ${#FAILED_INDICES[@]} task(s) as job: ${NEW_JOB_ID}"
echo "  Original job    : ${ORIG_JOBID}"
echo "  New job         : ${NEW_JOB_ID}"
echo "  Array indices   : ${FAILED_LIST}"
echo ""
echo "  Monitor with:"
echo "    ./slurm/check_status.sh ${NEW_JOB_ID}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
