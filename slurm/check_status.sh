#!/bin/bash
# =============================================================================
# check_status.sh — summarize per-task status of a submitted PID array job
#
# Usage:
#   ./slurm/check_status.sh <JOBID>
#
# Example:
#   ./slurm/check_status.sh 1234567
#
# Reports:
#   - Full sacct table: JobID, State, ExitCode, Elapsed, CPUTimeRAW, MaxRSS
#   - Count of COMPLETED / FAILED / TIMEOUT / CANCELLED / RUNNING / PENDING
#   - List of failed array indices (ready to paste into resubmit_failed.sh)
# =============================================================================

set -euo pipefail

JOBID="${1:?usage: check_status.sh <array_jobid>}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Array job status: ${JOBID}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Pull the full accounting table from the SLURM database.
# Fields:
#   JobID       — jobid_arrayidx (e.g. 1234567_42)
#   State       — COMPLETED, FAILED, CANCELLED, RUNNING, PENDING, TIMEOUT, ...
#   ExitCode    — exit:signal (e.g. 0:0 = success, 1:0 = non-zero exit)
#   Elapsed     — wall-clock time used (HH:MM:SS)
#   CPUTimeRAW  — core-seconds consumed
#   MaxRSS      — peak resident-set memory
RAW=$(sacct -j "${JOBID}" \
    --format=JobID,State,ExitCode,Elapsed,CPUTimeRAW,MaxRSS \
    --parsable2 --noheader 2>/dev/null || true)

if [ -z "${RAW}" ]; then
    echo "No sacct records found for job ${JOBID}."
    echo "The job may still be queued (sacct only shows jobs once they have started)."
    echo "Check queue status with: squeue -j ${JOBID}"
    exit 0
fi

# Print column header
printf "%-25s  %-12s  %-9s  %-10s  %-12s  %-10s\n" \
    "JobID" "State" "ExitCode" "Elapsed" "CPUTimeRAW" "MaxRSS"
printf '%0.s─' {1..82}
printf '\n'

declare -A STATE_COUNT
FAILED_INDICES=()

while IFS='|' read -r jobid state exitcode elapsed cputime maxrss; do
    # Skip the parent array job row (no underscore) and sub-steps (.batch, .extern)
    if [[ "${jobid}" != *_* ]]; then continue; fi
    if [[ "${jobid}" == *.batch ]] || [[ "${jobid}" == *.extern ]]; then continue; fi

    # Normalize state: sacct sometimes appends " by <uid>" to CANCELLED
    state_clean="${state%% *}"

    STATE_COUNT["${state_clean}"]=$(( ${STATE_COUNT["${state_clean}"]:-0} + 1 ))

    # Collect indices for tasks that need resubmission
    if [[ "${state_clean}" == "FAILED" || "${state_clean}" == "TIMEOUT" || "${state_clean}" == "CANCELLED" ]]; then
        arr_idx="${jobid##*_}"
        FAILED_INDICES+=("${arr_idx}")
    fi

    printf "%-25s  %-12s  %-9s  %-10s  %-12s  %-10s\n" \
        "${jobid}" "${state_clean}" "${exitcode}" "${elapsed}" "${cputime}" "${maxrss}"
done <<< "${RAW}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Summary:"
for state in COMPLETED RUNNING PENDING FAILED TIMEOUT CANCELLED; do
    count="${STATE_COUNT[${state}]:-0}"
    if [ "${count}" -gt 0 ]; then
        printf "    %-12s : %d\n" "${state}" "${count}"
    fi
done
echo ""

if [ "${#FAILED_INDICES[@]}" -gt 0 ]; then
    FAILED_LIST=$(IFS=','; echo "${FAILED_INDICES[*]}")
    echo "  Failed / timed-out / cancelled array indices:"
    echo "    ${FAILED_LIST}"
    echo ""
    echo "  To resubmit failed tasks (specify mc or data as appropriate):"
    echo "    ./slurm/resubmit_failed.sh ${JOBID} mc"
    echo "    ./slurm/resubmit_failed.sh ${JOBID} data"
else
    echo "  No failed tasks detected."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
