#!/bin/bash
# =============================================================================
# submit_mc.sh — submit the MC PID-training ntuple production as a slurm array
#
# Usage (from the suli2026_pid/ repo root on ifarm):
#   ./slurm/submit_mc.sh [N_FILES]
#
#   N_FILES  (optional) Number of HIPO files to process.
#            Default: all files found in the MC source directory (~318).
#            Pass a small number (e.g. 2) for a smoke test.
#
# Examples:
#   cd ~/CLAS/SULI/suli2026_pid
#   ./slurm/submit_mc.sh          # full run (~318 tasks)
#   ./slurm/submit_mc.sh 2        # smoke test (2 tasks)
#
# SLURM ACCOUNT / PARTITION NOTE:
# ─────────────────────────────────────────────────────────────────────────────
# This script submits with no --account or --partition directive, letting SLURM
# use your user default.  If sbatch rejects the submission with an error such
# as "no default account", "no QOS assigned", or "invalid partition specified":
#
#   1. Find your account:     sacctmgr show user $USER
#   2. Find valid partitions: sinfo -s
#   3. Edit slurm/_pid_training_array.sh and uncomment the two lines:
#        #SBATCH --account=<your_account>
#        #SBATCH --partition=<partition_name>
#      substituting the values from steps 1–2.
# ─────────────────────────────────────────────────────────────────────────────
#
# /cache/ STAGING NOTE:
# MC source is on /work/ (not /cache/), so tape staging is not needed here.
# See submit_data.sh for the /cache/ staging caveat.
# =============================================================================

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
# Layout: ~/CLAS/SULI/
#           ├── clas12_analysis_software/
#           └── suli2026_pid/
#                 └── slurm/   ← this script lives here
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK="${REPO_ROOT}/clas12_analysis_software"

MC_INPUT_DIR="/work/clas12/zurek/SULI/clasdis_rich_on"
OUTPUT_DIR="/volatile/clas12/${USER}/SULI/mc_v01"
FILE_LIST="${SCRIPT_DIR}/_mc_file_list.txt"
ARRAY_SCRIPT="${SCRIPT_DIR}/_pid_training_array.sh"

# ── Args ─────────────────────────────────────────────────────────────────────
N_LIMIT="${1:-0}"   # 0 means "all files"

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ ! -d "${MC_INPUT_DIR}" ]; then
    echo "ERROR: MC input directory not found: ${MC_INPUT_DIR}"
    exit 1
fi
if [ ! -d "${FRAMEWORK}" ]; then
    echo "ERROR: clas12_analysis_software not found as a sibling of suli2026_pid."
    echo "       Expected: ${FRAMEWORK}"
    echo "       Detected REPO_ROOT: ${REPO_ROOT}"
    echo ""
    echo "       Fix: symlink your clas12_analysis_software clone into place:"
    echo "         ln -s /path/to/your/clas12_analysis_software ${FRAMEWORK}"
    echo "       Or edit this script to set FRAMEWORK= directly."
    exit 1
fi
if [ ! -f "${ARRAY_SCRIPT}" ]; then
    echo "ERROR: Array script not found: ${ARRAY_SCRIPT}"
    exit 1
fi

# ── Module load ───────────────────────────────────────────────────────────────
# SLURM sets TMPDIR to a per-job scratch area (/scratch/slurm/<jobid>/...) where
# the Modules system cannot write its lockfiles, causing `module load clas12`
# to fail with "couldn't create error file ... no space left on device" even
# though df shows ample free space. Override to /tmp (compute-node local, ~8 GB
# free, ample for modules' tiny lockfiles). Confirmed on ifarm 2026-06.
export TMPDIR=/tmp

# Required before calling g++ (root-config) and confirming modules work.
# If 'module load clas12' fails, the batch tasks will also fail — fix here first.
# Bash subshells (including SLURM batch jobs and `srun --pty bash`) don't run
# the login init that defines the `module` function. Source it explicitly.
if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
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

# ── One-time converter compile ────────────────────────────────────────────────
# The binary is compiled once here so that the 318 array tasks share a single
# pre-built binary.  This avoids the per-task compile race (spec §1.5, §7.6).
echo "Compiling convert_txt_to_root ..."
(
    cd "${FRAMEWORK}"
    # shellcheck disable=SC2046
    g++ $(root-config --cflags --libs) \
        -o processing_scripts/convert_txt_to_root \
        processing_scripts/convert_txt_to_root.cpp
)
echo "Compile OK: ${FRAMEWORK}/processing_scripts/convert_txt_to_root"

# NOTE ON JAR-SWAP RACE:
# coatjava/bin/run-groovy lines 3–4 do: rm + cp of processing_classes.jar into
# coatjava/lib/services/.  With up to 50 concurrent tasks sharing this checkout
# the rm+cp can race, producing occasional ClassNotFoundException failures.
# Tasks are requeueable; resubmit_failed.sh handles recovery.  If failures are
# frequent, mitigate by rsyncing clas12_analysis_software/ to a per-submission
# directory before submitting (see design spec §7.5).

# ── Build sorted HIPO file list ───────────────────────────────────────────────
# -L follows symlinks; -type f skips dangling symlinks (i.e. the _0 stub file
# that caused "Input HIPO file does not exist" on the first test run).
echo "Scanning ${MC_INPUT_DIR} for HIPO files ..."
find -L "${MC_INPUT_DIR}" -name '*.hipo' -type f 2>/dev/null | sort > "${FILE_LIST}"
TOTAL=$(wc -l < "${FILE_LIST}")
if [ "${TOTAL}" -eq 0 ]; then
    echo "ERROR: No .hipo files found in ${MC_INPUT_DIR}"
    exit 1
fi

# Truncate list to N_LIMIT if requested
if [ "${N_LIMIT}" -gt 0 ] && [ "${N_LIMIT}" -lt "${TOTAL}" ]; then
    head -n "${N_LIMIT}" "${FILE_LIST}" > "${FILE_LIST}.tmp"
    mv "${FILE_LIST}.tmp" "${FILE_LIST}"
    N_FILES="${N_LIMIT}"
else
    N_FILES="${TOTAL}"
fi
LAST_IDX=$((N_FILES - 1))

echo "File list written: ${FILE_LIST}"
echo "  Total HIPO files found : ${TOTAL}"
echo "  Files to submit        : ${N_FILES}"
echo "  Array indices          : 0–${LAST_IDX}"

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ── Submit slurm array ────────────────────────────────────────────────────────
# %50 caps concurrent tasks to avoid hammering the filesystem.
# The array script takes one positional arg: the sample type ("mc").
# It reads the file list from ${SCRIPT_DIR}/_mc_file_list.txt (auto-discovered
# from BASH_SOURCE[0], so no path needs to be passed as an argument).
echo ""
echo "Submitting slurm array (0-${LAST_IDX}%50) ..."
JOB_ID=$(sbatch --parsable \
    --array="0-${LAST_IDX}%50" \
    "${ARRAY_SCRIPT}" mc)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Submitted MC array job  : ${JOB_ID}"
echo "  Tasks                   : ${N_FILES} (array 0–${LAST_IDX}, ≤50 concurrent)"
echo "  File list               : ${FILE_LIST}"
echo "  Outputs land in         : ${OUTPUT_DIR}/"
echo "  Logs                    : /farm_out/${USER}/suli/pid_train_${JOB_ID}_<idx>.{out,err}"
echo ""
echo "  Follow-up commands:"
echo "    squeue -u \$USER"
echo "    ./slurm/check_status.sh ${JOB_ID}"
echo "    ./slurm/resubmit_failed.sh ${JOB_ID} mc"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
