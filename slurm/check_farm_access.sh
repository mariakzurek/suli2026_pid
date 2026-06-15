#!/bin/bash
# =============================================================================
# check_farm_access.sh — 5-check preflight for ifarm BDT training
#
# Usage (from the suli2026_pid/ repo root on ifarm):
#   ./slurm/check_farm_access.sh
#
# Runs five checks needed before submitting the BDT training job.  Each check
# prints PASS, WARN, or FAIL with a → Fix: hint.  Exits 0 only if no FAIL.
# Run this once before the first sbatch submission; address any FAIL lines.
# =============================================================================

set -uo pipefail

FAIL=0
WARN=0
PASS=0
RESULTS=()

_pass() { echo "  ✓ PASS: $*"; PASS=$((PASS + 1)); RESULTS+=("PASS: $*"); }
_warn() { echo "  ⚠ WARN: $*"; WARN=$((WARN + 1)); RESULTS+=("WARN: $*"); }
_fail() { echo "  ✗ FAIL: $*"; FAIL=$((FAIL + 1)); RESULTS+=("FAIL: $*"); }

echo "========================================"
echo " check_farm_access.sh — ifarm preflight"
echo "========================================"
echo ""

# ── Check 1: hostname matches ifarm* ─────────────────────────────────────────
echo "[1/5] Hostname check ..."
HOST="$(hostname -s 2>/dev/null || hostname)"
if [[ "${HOST}" == ifarm* ]]; then
    _pass "Running on ifarm: ${HOST}"
else
    _warn "Not on ifarm (hostname: ${HOST}).  Most checks may fail or be irrelevant."
    echo "       → Fix: ssh <user>@ifarm.jlab.org then rerun."
fi
echo ""

# ── Check 2: module load clas12 succeeds ─────────────────────────────────────
echo "[2/5] module load clas12 ..."
export TMPDIR=/tmp   # Required: SLURM/modules lockfile issue (see _pid_training_array.sh)
if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck source=/dev/null
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        # shellcheck source=/dev/null
        source /usr/share/Modules/init/bash
    else
        _fail "Cannot locate modules init script (tried /etc/profile.d/modules.sh, /usr/share/Modules/init/bash)."
        echo "       → Fix: source the appropriate modules init manually or contact system admin."
        echo ""
    fi
fi

if command -v module >/dev/null 2>&1; then
    module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles 2>/dev/null
    module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles 2>/dev/null
    module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles 2>/dev/null
    if module load clas12 2>/dev/null; then
        _pass "module load clas12 succeeded."
    else
        _fail "module load clas12 failed."
        echo "       → Fix: check that CVMFS is mounted (/cvmfs/) and that the"
        echo "              module paths above are correct for this machine."
    fi
else
    _fail "module command still not available after sourcing init scripts."
    echo "       → Fix: see cooper_day1_and_week1.md §4 for environment setup."
fi
echo ""

# ── Check 3: conda activate suli2026_pid succeeds ────────────────────────────
echo "[3/5] conda activate suli2026_pid ..."
CONDA_ACTIVATED=0

# Probe the three known hook locations (no ~/.bashrc sourcing).
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
            _pass "conda activate suli2026_pid succeeded (hook: ${CONDA_HOOK})."
            CONDA_ACTIVATED=1
            break
        fi
    fi
done

if [ "${CONDA_ACTIVATED}" -eq 0 ]; then
    _fail "conda activate suli2026_pid failed (or conda not found)."
    echo "       → Fix: see cooper_day1_and_week1.md §4f for environment setup."
    echo "              Checked hooks at:"
    echo "                /work/clas12/${USER}/miniconda3/etc/profile.d/conda.sh"
    echo "                \${HOME}/miniconda3/etc/profile.d/conda.sh"
    echo "                /apps/anaconda3/etc/profile.d/conda.sh"
fi
echo ""

# ── Check 4: /volatile/ and /farm_out/$USER/suli/ writable ───────────────────
echo "[4/5] Scratch and log directory access ..."
VOL_DIR="/volatile/clas12/${USER}"
FARM_DIR="/farm_out/${USER}/suli"

if [ -d "${VOL_DIR}" ]; then
    if [ -w "${VOL_DIR}" ]; then
        _pass "/volatile/clas12/${USER}/ exists and is writable."
    else
        _fail "/volatile/clas12/${USER}/ exists but is not writable."
        echo "       → Fix: check volatile quota; contact system admin if full."
    fi
else
    _fail "/volatile/clas12/${USER}/ does not exist."
    echo "       → Fix: on ifarm, /volatile/clas12/\$USER/ is your batch scratch area."
    echo "              It should be created automatically; contact system admin."
fi

if mkdir -p "${FARM_DIR}" 2>/dev/null && [ -w "${FARM_DIR}" ]; then
    _pass "/farm_out/${USER}/suli/ exists and is writable (created if missing)."
else
    _fail "Cannot create or write to /farm_out/${USER}/suli/."
    echo "       → Fix: /farm_out/\$USER/ must be your SLURM log directory;"
    echo "              contact system admin if it is missing or permission-denied."
fi
echo ""

# ── Check 5: sbatch --test-only accepts a realistic submission ────────────────
echo "[5/5] sbatch --test-only ..."

PREFLIGHT_SCRIPT="$(mktemp /tmp/preflight_XXXXXX.sh)"
cat > "${PREFLIGHT_SCRIPT}" << 'EOSJOB'
#!/bin/bash
#SBATCH --job-name=preflight_check
#SBATCH --account=clas12
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/farm_out/%u/suli/preflight_%j.out
#SBATCH --error=/farm_out/%u/suli/preflight_%j.err
echo "preflight job"
EOSJOB
chmod +x "${PREFLIGHT_SCRIPT}"

if command -v sbatch >/dev/null 2>&1; then
    if sbatch --test-only "${PREFLIGHT_SCRIPT}" 2>&1 | grep -qiE "submission test successful|would be submitted"; then
        _pass "sbatch --test-only: submission accepted by scheduler."
    else
        TEST_OUTPUT="$(sbatch --test-only "${PREFLIGHT_SCRIPT}" 2>&1)"
        if echo "${TEST_OUTPUT}" | grep -qiE "error|invalid|rejected|fail"; then
            _fail "sbatch --test-only: scheduler rejected the submission."
            echo "       → Fix: check account (sacctmgr show user \$USER)."
            echo "              Scheduler output: ${TEST_OUTPUT}"
        else
            # Some SLURM versions don't print "successful" — treat non-error as OK.
            _pass "sbatch --test-only: no error returned (output: ${TEST_OUTPUT})."
        fi
    fi
else
    _warn "sbatch not found — not on a submission node (expected on ifarm login nodes)."
    echo "       → Fix: run from ifarm.jlab.org (ssh, not a compute node)."
fi
rm -f "${PREFLIGHT_SCRIPT}"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================"
echo " Summary"
echo "========================================"
for r in "${RESULTS[@]}"; do echo "  ${r}"; done
echo ""
echo "  PASS: ${PASS}  WARN: ${WARN}  FAIL: ${FAIL}"
echo ""

if [ "${FAIL}" -gt 0 ]; then
    echo "  ✗ Not ready — address the FAIL items above before submitting."
    exit 1
else
    echo "  ✓ All checks passed (${WARN} warning(s)).  Safe to submit."
    exit 0
fi
