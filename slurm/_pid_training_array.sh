#!/bin/bash
# =============================================================================
# _pid_training_array.sh — SLURM array task script for PID ntuple production
#
# This script is NOT meant to be invoked directly by the user.  It is submitted
# by submit_mc.sh or submit_data.sh via sbatch.  The underscore prefix marks it
# as internal.
#
# Each SLURM array task processes exactly one HIPO file identified by
# $SLURM_ARRAY_TASK_ID.  Workflow per task:
#   1. Pick the i-th line from the file list written by submit_*.sh.
#   2. Symlink that one HIPO file into a per-task /scratch/ directory.
#   3. Run the groovy script against the scratch dir (so it sees exactly 1 file).
#   4. Run convert_txt_to_root on the output .txt (on scratch).
#   5. Copy only the final .root to /volatile/.
#   6. Delete all scratch contents on EXIT (even on failure, via trap).
#
# Positional argument (passed by submit_*.sh on the sbatch command line):
#   $1  SAMPLE  — "mc" or "data"
#
# All other paths are auto-derived from the location of this script.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SLURM DIRECTIVES
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=pid_train
#SBATCH --time=00:45:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/farm_out/%u/suli/pid_train_%A_%a.out
#SBATCH --error=/farm_out/%u/suli/pid_train_%A_%a.err
#SBATCH --requeue

# Account: verified via `sacct --format=Account -X | sort -u`. Cluster default
# partition is `production` (see `sinfo -s`), so no --partition directive is
# needed for batch submissions. If batch submissions later require a specific
# partition, add e.g.:
#   #SBATCH --partition=production
#SBATCH --account=clas12

# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Positional argument ───────────────────────────────────────────────────────
SAMPLE="${1:?usage: $0 <mc|data>}"

if [ "${SAMPLE}" != "mc" ] && [ "${SAMPLE}" != "data" ]; then
    echo "ERROR: SAMPLE must be 'mc' or 'data', got: ${SAMPLE}"
    exit 1
fi

# ── SLURM environment check ───────────────────────────────────────────────────
# Use a soft default of 0 so the script can be sourced interactively for smoke
# testing without SLURM_ARRAY_TASK_ID being set.  Under a real sbatch array
# submission this variable is always set by the scheduler.
ARRAY_IDX="${SLURM_ARRAY_TASK_ID:-0}"

# ── Resolve repo root ─────────────────────────────────────────────────────────
# Script lives at: $REPO_ROOT/suli2026_pid/slurm/_pid_training_array.sh
# REPO_ROOT is the sibling-parent directory that contains both suli2026_pid/ and
# clas12_analysis_software/ as siblings.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# File list was written by submit_*.sh.
# Prefer the FILE_LIST path the submitter passed via --export, falling back
# to auto-detection from script location for interactive testing.
LIST="${FILE_LIST:-${SCRIPT_DIR}/_${SAMPLE}_file_list.txt}"
if [ ! -f "${LIST}" ]; then
    echo "ERROR: File list not found: ${LIST}"
    echo "       Run submit_${SAMPLE}.sh first."
    exit 1
fi

# ── Module environment ────────────────────────────────────────────────────────
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

# QADB is required by the data groovy; the MC groovy does not use it.
if [ "${SAMPLE}" = "data" ]; then
    module load qadb/3.3.0
fi

# JVM scratch — required to avoid java.util.logging.LogManager init failure
# (see cooper_day1_and_week1.md, JVM scratch section).
# The tmpfs/auto module (loaded as a dependency of clas12) sets TMPDIR to a
# writable location on JLab compute nodes (typically /u/home/$USER/tmpfs).
# Defer to that; fall back to /tmp if for some reason TMPDIR is unset.
# Do NOT mkdir here — tmpfs/auto handles creation on compute nodes.
export _JAVA_OPTIONS="-Djava.io.tmpdir=${TMPDIR:-/tmp}"

# ── Resolve this task's input HIPO file ──────────────────────────────────────
# SLURM_ARRAY_TASK_ID is 0-indexed; sed line numbers are 1-based.
LINE_NUM=$((ARRAY_IDX + 1))
INPUT_HIPO=$(sed -n "${LINE_NUM}p" "${LIST}")

if [ -z "${INPUT_HIPO}" ]; then
    echo "ERROR: No file at index ${ARRAY_IDX} in ${LIST} (line ${LINE_NUM})"
    exit 1
fi
if [ ! -f "${INPUT_HIPO}" ]; then
    echo "ERROR: Input HIPO file does not exist (may not be staged from tape): ${INPUT_HIPO}"
    echo "       For /cache/ files, check staging with: jstat ${INPUT_HIPO}"
    exit 1
fi

OUTPUT_STEM="$(basename "${INPUT_HIPO}" .hipo)"
echo "Task ${ARRAY_IDX}: processing ${OUTPUT_STEM}"

# ── Per-task scratch directory ────────────────────────────────────────────────
# All intermediates (.txt, scratch .root) live here and are deleted on EXIT.
# Only the final .root is moved to /volatile/ before cleanup.
#
# Scratch base: /volatile/ is JLab's batch-processing scratch tier (parallel
# filesystem, large quota, fast I/O). DO NOT use:
#   - /scratch/$USER     (doesn't exist on compute nodes)
#   - /scratch/slurm/... (project-quota-limited, ~MB scale)
#   - $TMPDIR / ~/tmpfs  (home quota, ~25 GB)
# Fallback to /tmp/$USER for interactive (non-SLURM) testing.
if [ -d "/volatile/clas12/${USER}" ]; then
    SCRATCH_BASE="/volatile/clas12/${USER}/SULI/scratch"
else
    SCRATCH_BASE="/tmp/${USER}"
fi
mkdir -p "${SCRATCH_BASE}"
SCRATCH="${SCRATCH_BASE}/pid_train_${SLURM_ARRAY_TASK_ID:-local}"

# Defensive: clear stale scratch from previous failed runs whose trap-on-EXIT
# cleanup didn't fire (e.g., script killed externally).
rm -rf "${SCRATCH}"
mkdir -p "${SCRATCH}"
trap 'echo "Cleaning scratch: ${SCRATCH}"; rm -rf "${SCRATCH}"' EXIT

# ── Resolve shared framework path ────────────────────────────────────────────
# Prefer the FRAMEWORK path the submitter passed via --export, falling back
# to deriving from script location for interactive testing.
FRAMEWORK="${FRAMEWORK:-${REPO_ROOT}/clas12_analysis_software}"
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

# ── Per-task framework copy to avoid run-groovy jar-swap race ────────────────
# coatjava/bin/run-groovy unconditionally does on every invocation:
#   rm coatjava/lib/services/processing_classes.jar
#   cp processing_classes/dist/processing_classes.jar coatjava/lib/services/
# With up to 50 array tasks running concurrently from a shared framework
# checkout these two ops race, producing a momentarily absent jar and
# ClassNotFoundException failures across the entire job (confirmed: job 5185666,
# 300/300 tasks failed in ~7 s each).
# Fix: each task rsyncs the framework (~75 MB) into its own private scratch
# directory and runs run-groovy from there.  No two tasks share the jar file.
# Overhead: ~16 s per task on /volatile/; acceptable at 5–10 min per task.
TASK_FRAMEWORK="${SCRATCH}/framework"
echo "Rsyncing framework to per-task scratch: ${TASK_FRAMEWORK}"
rsync -a --exclude='.git' "${FRAMEWORK}/" "${TASK_FRAMEWORK}/"
echo "Framework rsync done."

# ── Symlink the one HIPO file into a flat per-task input directory ────────────
# The groovy script scans args[0] recursively for *.hipo (spec §1.2).  We give
# it a directory containing exactly one symlink so it processes exactly one file.
INPUT_DIR="${SCRATCH}/input"
mkdir -p "${INPUT_DIR}"
ln -sf "${INPUT_HIPO}" "${INPUT_DIR}/${OUTPUT_STEM}.hipo"

# ── Intermediate and final output paths ──────────────────────────────────────
OUTPUT_TXT="${SCRATCH}/${OUTPUT_STEM}.txt"
OUTPUT_ROOT="${SCRATCH}/${OUTPUT_STEM}.root"
FINAL_DIR="/volatile/clas12/${USER}/SULI/${SAMPLE}_v01"
FINAL_ROOT="${FINAL_DIR}/${OUTPUT_STEM}.root"

# Output directory must exist (created by submit_*.sh, but guard here too)
mkdir -p "${FINAL_DIR}"

if [ "${SAMPLE}" = "mc" ]; then
    GROOVY="${TASK_FRAMEWORK}/processing_scripts/processing_mc_pid_training.groovy"
    # runnum_override=11 forces MC mode (bypasses QA; see spec §1.1 and §1.8)
    RUNNUM_OVERRIDE="11"
    SCRIPT_INDEX=7
    IS_MC=1
elif [ "${SAMPLE}" = "data" ]; then
    GROOVY="${TASK_FRAMEWORK}/processing_scripts/processing_data_pid_training.groovy"
    # No runnum_override for data: per-run lookup drives beam energy and QADB
    RUNNUM_OVERRIDE=""
    SCRIPT_INDEX=8
    IS_MC=0
fi
BEAM_E="10.6041"   # MC: used verbatim when runnum=11; data: fallback for unknown runs

# ── Verify the converter binary was pre-compiled at submit time ───────────────
CONVERTER="${FRAMEWORK}/processing_scripts/convert_txt_to_root"
if [ ! -x "${CONVERTER}" ]; then
    echo "ERROR: Converter binary not found or not executable: ${CONVERTER}"
    echo "       It should have been compiled by submit_${SAMPLE}.sh."
    echo "       Re-run the submit script to rebuild it."
    exit 1
fi

# ── Run groovy ───────────────────────────────────────────────────────────────
# Bypass processing.csh entirely (see spec §1.5):
#   - processing.csh would run `git pull` and recompile the converter per task.
#   - We call coatjava/bin/run-groovy directly with the project classpath.
#
# Groovy CLI (spec §1.1):
#   <hipo_dir> <output_basename_with_.txt> [n_files] [beam_E] [runnum_override]
# We pass:
#   hipo_dir  = INPUT_DIR (per-task scratch with one symlinked HIPO)
#   output    = OUTPUT_TXT (full path including .txt — groovy does NOT add it)
#   n_files   = 1 (process exactly the one file in INPUT_DIR)
#   beam_E    = BEAM_E
#   runnum    = RUNNUM_OVERRIDE (11 for MC; omitted entirely for data)
#
echo "--- groovy start: $(date) ---"
cd "${TASK_FRAMEWORK}"
# For data, RUNNUM_OVERRIDE is empty; the unquoted expansion produces no extra
# argument, so the groovy receives exactly 4 positional args (no override).
# shellcheck disable=SC2086
"${TASK_FRAMEWORK}/coatjava/bin/run-groovy" \
    "${GROOVY}" \
    "${INPUT_DIR}" \
    "${OUTPUT_TXT}" \
    1 \
    "${BEAM_E}" \
    ${RUNNUM_OVERRIDE}
echo "--- groovy end: $(date) ---"

# Sanity: groovy must have produced a non-empty .txt
if [ ! -s "${OUTPUT_TXT}" ]; then
    echo "ERROR: Groovy produced an empty or missing output: ${OUTPUT_TXT}"
    exit 1
fi

# ── Convert txt → ROOT ────────────────────────────────────────────────────────
# Converter signature (convert_txt_to_root.cpp:160–175):
#   <input_txt> <output_root> <script_index> <is_mc>
#   script_index = 7 for MC (processing_mc_pid_training.groovy)
#   script_index = 8 for data (processing_data_pid_training.groovy)
echo "--- converter start: $(date) ---"
"${CONVERTER}" \
    "${OUTPUT_TXT}" \
    "${OUTPUT_ROOT}" \
    "${SCRIPT_INDEX}" \
    "${IS_MC}"
echo "--- converter end: $(date) ---"

if [ ! -s "${OUTPUT_ROOT}" ]; then
    echo "ERROR: Converter produced an empty or missing ROOT file: ${OUTPUT_ROOT}"
    exit 1
fi

# ── Move final ROOT to /volatile/ ────────────────────────────────────────────
# Only the .root is moved.  The .txt is deleted by the EXIT trap.
# mv is used (not cp) because SCRATCH and FINAL_DIR are both on /volatile/ —
# same filesystem → instant inode rename.  If they ever land on different
# filesystems mv falls back to copy+delete, so this is safe either way.
mv "${OUTPUT_ROOT}" "${FINAL_ROOT}"

ROOT_SIZE=$(du -sh "${FINAL_ROOT}" | cut -f1)
echo "DONE  task=${ARRAY_IDX}  stem=${OUTPUT_STEM}  size=${ROOT_SIZE}"
echo "      output=${FINAL_ROOT}"
