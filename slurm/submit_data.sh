#!/bin/bash
# =============================================================================
# submit_data.sh — submit the RGA Fa18 data PID-training production as a slurm
#                  array
#
# Usage (from the suli2026_pid/ repo root on ifarm):
#   ./slurm/submit_data.sh [N_FILES]
#
#   N_FILES  (optional) Number of HIPO files to process.
#            Default: 20 (the Week-3 spec asks for ≥ 10–20 RGA Fa18 files).
#            Pass 0 to process all files found in the directory (large!).
#            Pass a smaller number (e.g. 2) for a smoke test.
#
# Examples:
#   cd ~/CLAS/SULI/suli2026_pid
#   ./slurm/submit_data.sh         # 20 files (default)
#   ./slurm/submit_data.sh 2       # smoke test (2 tasks)
#   ./slurm/submit_data.sh 0       # all files in directory (can be very large)
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
# ─────────────────────────────────────────────────────────────────────────────
# The data source lives in /cache/clas12/, which is a tape-backed filesystem.
# Files not currently on disk must be staged before the array tasks can read
# them — a task reading from tape will stall for hours and then time out.
#
# Before submitting this array:
#   1. Check which files are on disk (DISK vs TAPE status):
#        jstat /cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/<file>.hipo
#      Or check the whole directory (first N lines):
#        ls /cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/ | head -25
#
#   2. Request staging for files that are on tape:
#        jcache stage /cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/<file>.hipo
#
#   3. Wait until jstat reports DISK status, then run this script.
#      Typical staging time: minutes to several hours depending on tape load.
#
# This script does NOT issue jcache requests automatically.  Stage manually first.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
# Layout: ~/CLAS/SULI/
#           ├── clas12_analysis_software/
#           └── suli2026_pid/
#                 └── slurm/   ← this script lives here
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FRAMEWORK="${REPO_ROOT}/clas12_analysis_software"

DATA_INPUT_DIR="/cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis"
OUTPUT_DIR="/volatile/clas12/${USER}/SULI/data_v01"
FILE_LIST="${SCRIPT_DIR}/_data_file_list.txt"
ARRAY_SCRIPT="${SCRIPT_DIR}/_pid_training_array.sh"

# ── Args ─────────────────────────────────────────────────────────────────────
N_LIMIT="${1:-20}"   # Default: 20 (Week-3 spec: ≥ 10–20); use 0 for all

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ ! -d "${DATA_INPUT_DIR}" ]; then
    echo "ERROR: Data input directory not found: ${DATA_INPUT_DIR}"
    echo "       Check that the /cache/ filesystem is mounted on this node."
    echo "       If files need staging from tape, use jcache (see header above)."
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
# QADB is required by the data groovy script (not needed for MC).
module load qadb/3.3.0

# ── One-time converter compile ────────────────────────────────────────────────
echo "Compiling convert_txt_to_root ..."
(
    cd "${FRAMEWORK}"
    # shellcheck disable=SC2046
    g++ $(root-config --cflags --libs) \
        -o processing_scripts/convert_txt_to_root \
        processing_scripts/convert_txt_to_root.cpp
)
echo "Compile OK: ${FRAMEWORK}/processing_scripts/convert_txt_to_root"

# NOTE ON JAR-SWAP RACE: see submit_mc.sh for explanation.

# ── Build sorted HIPO file list ───────────────────────────────────────────────
# -L follows symlinks; -type f skips dangling symlinks (i.e. any _0 stub files
# that caused "Input HIPO file does not exist" on test runs).
echo "Scanning ${DATA_INPUT_DIR} for HIPO files ..."
find -L "${DATA_INPUT_DIR}" -maxdepth 1 -name '*.hipo' -type f 2>/dev/null | sort > "${FILE_LIST}"
TOTAL=$(wc -l < "${FILE_LIST}")
if [ "${TOTAL}" -eq 0 ]; then
    echo "ERROR: No .hipo files found in ${DATA_INPUT_DIR}"
    echo "       Files may need to be staged from tape — see /cache/ staging note above."
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
echo ""
echo "  REMINDER: confirm these files are on disk before submitting."
echo "  Check with: jstat ${DATA_INPUT_DIR}/<file>.hipo"
echo "  Stage with: jcache stage <file_path>"
echo ""

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ── Submit slurm array ────────────────────────────────────────────────────────
# %50 caps concurrent tasks to avoid hammering the filesystem.
# The array script takes one positional arg: the sample type ("data").
# --export=ALL,... forwards the current environment (same as SLURM default) PLUS
# the explicit path variables.  This is required because SLURM copies the script
# to a per-job spool directory (/var/spool/slurm/d/jobXXX/) before execution, so
# BASH_SOURCE[0] inside the array task resolves to that spool path — NOT to the
# original slurm/ directory.  Without --export the file-list and framework paths
# derived from BASH_SOURCE[0] would be wrong in batch.
echo "Submitting slurm array (0-${LAST_IDX}%50) ..."
JOB_ID=$(sbatch --parsable \
    --array="0-${LAST_IDX}%50" \
    --export=ALL,FILE_LIST="${FILE_LIST}",FRAMEWORK="${FRAMEWORK}",REPO_ROOT="${REPO_ROOT}" \
    "${ARRAY_SCRIPT}" data)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Submitted data array job : ${JOB_ID}"
echo "  Tasks                    : ${N_FILES} (array 0–${LAST_IDX}, ≤50 concurrent)"
echo "  File list                : ${FILE_LIST}"
echo "  Outputs land in          : ${OUTPUT_DIR}/"
echo "  Logs                     : /farm_out/${USER}/suli/pid_train_${JOB_ID}_<idx>.{out,err}"
echo ""
echo "  Follow-up commands:"
echo "    squeue -u \$USER"
echo "    ./slurm/check_status.sh ${JOB_ID}"
echo "    ./slurm/resubmit_failed.sh ${JOB_ID} data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
