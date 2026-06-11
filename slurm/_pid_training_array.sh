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

# Account / partition: if SLURM rejects with "no default account" or
# "no valid partition", uncomment and set these:
#   #SBATCH --account=<your_account>
#   #SBATCH --partition=<partition_name>
# Find your account with: sacctmgr show user $USER
# Find available partitions with: sinfo

# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Positional argument ───────────────────────────────────────────────────────
SAMPLE="${1:?usage: $0 <mc|data>}"

if [ "${SAMPLE}" != "mc" ] && [ "${SAMPLE}" != "data" ]; then
    echo "ERROR: SAMPLE must be 'mc' or 'data', got: ${SAMPLE}"
    exit 1
fi

# ── SLURM environment check ───────────────────────────────────────────────────
ARRAY_IDX="${SLURM_ARRAY_TASK_ID:?must be run as a slurm array task}"

# ── Resolve repo root ─────────────────────────────────────────────────────────
# Script lives at: $REPO_ROOT/suli2026_pid/slurm/_pid_training_array.sh
# REPO_ROOT is the ~/CLAS/SULI/ parent that contains both suli2026_pid/ and
# clas12_analysis_software/ as siblings.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# File list was written by submit_*.sh
LIST="${SCRIPT_DIR}/_${SAMPLE}_file_list.txt"
if [ ! -f "${LIST}" ]; then
    echo "ERROR: File list not found: ${LIST}"
    echo "       Run submit_${SAMPLE}.sh first."
    exit 1
fi

# ── Module environment ────────────────────────────────────────────────────────
module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module load clas12

# QADB is required by the data groovy; the MC groovy does not use it.
if [ "${SAMPLE}" = "data" ]; then
    module load qadb/3.3.0
fi

# JVM scratch — required to avoid java.util.logging.LogManager init failure
# (see cooper_day1_and_week1.md, JVM scratch section)
mkdir -p "/scratch/${USER}/tmpfs"
export _JAVA_OPTIONS="-Djava.io.tmpdir=/scratch/${USER}/tmpfs"

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
# Only the final .root is copied to /volatile/ before cleanup.
SCRATCH="/scratch/${USER}/pid_train_${SLURM_JOB_ID}_${ARRAY_IDX}"
mkdir -p "${SCRATCH}"
trap 'echo "Cleaning scratch: ${SCRATCH}"; rm -rf "${SCRATCH}"' EXIT

# ── Symlink the one HIPO file into a flat per-task input directory ────────────
# The groovy script scans args[0] recursively for *.hipo (spec §1.2).  We give
# it a directory containing exactly one symlink so it processes exactly one file.
INPUT_DIR="${SCRATCH}/input"
mkdir -p "${INPUT_DIR}"
ln -sf "${INPUT_HIPO}" "${INPUT_DIR}/"

# ── Intermediate and final output paths ──────────────────────────────────────
OUTPUT_TXT="${SCRATCH}/${OUTPUT_STEM}.txt"
OUTPUT_ROOT="${SCRATCH}/${OUTPUT_STEM}.root"
FINAL_DIR="/volatile/clas12/${USER}/SULI/${SAMPLE}_v01"
FINAL_ROOT="${FINAL_DIR}/${OUTPUT_STEM}.root"

# Output directory must exist (created by submit_*.sh, but guard here too)
mkdir -p "${FINAL_DIR}"

# ── Pick groovy script and converter args based on sample type ────────────────
FRAMEWORK="${REPO_ROOT}/clas12_analysis_software"

if [ "${SAMPLE}" = "mc" ]; then
    GROOVY="${FRAMEWORK}/processing_scripts/processing_mc_pid_training.groovy"
    # runnum_override=11 forces MC mode (bypasses QA; see spec §1.1 and §1.8)
    RUNNUM_OVERRIDE="11"
    SCRIPT_INDEX=7
    IS_MC=1
elif [ "${SAMPLE}" = "data" ]; then
    GROOVY="${FRAMEWORK}/processing_scripts/processing_data_pid_training.groovy"
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

# ── NOTE ON run-groovy JAR-SWAP RACE ─────────────────────────────────────────
# coatjava/bin/run-groovy lines 3–4 do:
#   rm coatjava/lib/services/processing_classes.jar
#   cp processing_classes/dist/processing_classes.jar coatjava/lib/services/
# With many array tasks running concurrently on the SAME framework checkout,
# these two operations race and can produce a momentarily absent jar, causing
# ClassNotFoundException failures in some tasks.
#
# For this first production run this risk is accepted — tasks are requeueable
# and resubmit_failed.sh handles recovery.  If ClassNotFound errors appear in
# the .err logs, the fix is to rsync clas12_analysis_software/ into a
# per-submission directory before submitting and point GROOVY/CONVERTER at the
# copy (see design spec §7.5 for the rsync pattern).

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
cd "${FRAMEWORK}"
# For data, RUNNUM_OVERRIDE is empty; the unquoted expansion produces no extra
# argument, so the groovy receives exactly 4 positional args (no override).
# shellcheck disable=SC2086
"${FRAMEWORK}/coatjava/bin/run-groovy" \
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

# ── Copy final ROOT to /volatile/ ────────────────────────────────────────────
# Only the .root is copied.  The .txt and scratch .root are deleted by the EXIT
# trap, so /volatile/ never receives the .txt and scratch never accumulates.
cp "${OUTPUT_ROOT}" "${FINAL_ROOT}"

ROOT_SIZE=$(du -sh "${FINAL_ROOT}" | cut -f1)
echo "DONE  task=${ARRAY_IDX}  stem=${OUTPUT_STEM}  size=${ROOT_SIZE}"
echo "      output=${FINAL_ROOT}"
