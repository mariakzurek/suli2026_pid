# Cooper — Day 1 and Week 1 Onboarding

**Project:** ML kaon PID for CLAS12 SIDIS `ep → e' p K+ X`
**PI:** Maria Zurek (ANL)
**Read alongside:** `cooper_10week_plan.md` — the full 10-week schedule lives there; this
document covers Day 1 orientation only and does not repeat the schedule.

---

## 1. Welcome

The project you are stepping into has been scoped carefully, but the actual work is yours.
Nobody has run this pipeline on this channel yet. The questions you will answer: does the
ML classifier beat the standard cut, by how much, and where, are questions the group
genuinely needs answered to do the physics. This is not a pre-cooked tutorial that checks
boxes and produces a known result.

---

## 2. Project in Plain English

### 2.1 The problem

CLAS12's Event Builder (EB) assigns particle IDs using the time-of-flight system (FTOF).
It works well at low momentum, where pions and kaons have clearly different velocities
at a given momentum. Above roughly 2–2.5 GeV/c, the velocities converge and the FTOF
cannot reliably separate pi+ from K+. The EB's refinement tool is a variable called
`chi2pid`: a signed number quantifying how well the measured beta matches the kaon
hypothesis. The standard production cut on this variable is **momentum-dependent** — at low
momentum (p < 2 GeV/c) it's a symmetric ±3σ window around the kaon-hypothesis mean, but
above 2 GeV/c the window contracts asymmetrically to handle the worsening pion/kaon
separation. The Python reference implementation lives in `scripts/baseline_chi2pid.py` in
the analysis repo. The simpler form `|chi2pid| < 3` is sometimes used as a loose cut for
orientation but is not the production baseline.

The problem is that even with this cut, pions leak into the kaon sample — at high momentum,
pion contamination in the EB-identified K+ sample reaches 30–50%. The numbers are known from two sources: simulation with MC truth
labels, and the RICH detector (a Cherenkov ring imager installed in one sector of the
Forward Detector), which provides independent PID above ~1.75 GeV/c. Connor Pecar's prior
work on a dihadron channel (`ep → e' K+ pi- X`) documented all of this carefully. His
analysis note's Section 8 (limitations) is the clearest statement of why the problem is
not fully solved yet; `connor_method_summary.md` in the notes directory distills the
relevant parts.

### 2.2 What ML can do about it

The EB's `chi2pid` uses essentially one number from the FTOF. But each track leaves a
richer footprint: the full FTOF hit pattern across two panel layers, calorimeter energy
deposits at three depth layers (PCAL, ECin, ECout), and the Cherenkov signal from the HTCC
(which can veto pions in some momentum ranges). An ML classifier — a gradient-boosted decision tree (BDT), or a small neural net as a
second model family — can look at all of these simultaneously and learn discriminants
that `chi2pid` alone misses. The training data is `clasdis` Monte Carlo,
where every reconstructed track has a geometrically-matched MC truth label, so we know
which tracks are real K+ and which are pi+ wearing a K+ mask. The classifier learns to
separate them without ever using the truth label at inference time.

### 2.3 What you own

End to end: the training data pipeline (running the Groovy script on the clasdis files,
converting to ROOT, loading into Python), the model training and comparison (BDT first,
then a small neural net as a second model family), the validation against MC truth, the
cross-check against actual RGA pass-2 data using the Simone method (`ep → e h+ (n)`
neutron-tagged) plus RICH where it has coverage, and the headline number — "at this kaon
efficiency, the ML reduces pion contamination from X% to Y% in the `ep → e' p K+ X`
analysis." The final products are a calibrated model file, a code repository that
reproduces it from scratch, a poster, and a written report. The 10-week plan
(`cooper_10week_plan.md`) has the full schedule with tasks, done-whens, and fallback
options.

---

## 3. Day-1 Logistics

### 3.2 Access checklist

Work through this list on Day 1. Flag anything that isn't working immediately — delays in
ifarm access compound into lost days.

- [ ] SSH to ifarm: `ssh <username>@ifarm.jlab.org`  
- [ ] `hipo-utils` is in `PATH` after loading the module (check with `which hipo-utils`)
- [ ] Jupyter accessible on ifarm or locally — whichever you prefer for Python work, https://jupyterhub.jlab.org
- [ ] GitHub: confirm you can read and push to the framework fork  
      `https://github.com/mariakzurek/clas12_analysis_software` (branch: `suli_kaon_pid`)  
      and to the analysis repo `https://github.com/mariakzurek/suli2026_pid` (branch: `main`)
- [ ] Slack / email channel for the group  

### 3.3 Important paths

**Two repos for this project:**

- **Framework repo** (forked from Timothy Hayward's analysis software — Groovy/Java/C++
  pipeline that produces ntuples from HIPO files):
  - GitHub: `https://github.com/mariakzurek/clas12_analysis_software`
  - SSH clone: `git@github.com:mariakzurek/clas12_analysis_software.git`
  - Working branch: `suli_kaon_pid` (off `rich_studies`)
  - Local path (on ifarm): `/work/clas12/<username>/SULI/clas12_analysis_software/`
    (NOT `/home/` — see Section 4b for clone instructions)

- **Analysis repo** (Maria's, brand new — for all Python notebooks, ML training, plots,
  slurm scripts):
  - GitHub: `https://github.com/mariakzurek/suli2026_pid`
  - SSH clone: `git@github.com:mariakzurek/suli2026_pid.git`
  - Single branch: `main`
  - Local path (on ifarm): `/work/clas12/<username>/SULI/suli2026_pid/`
  - He commits his work here, NOT in the framework repo.

| What | Path |
|---|---|
| clasdis MC files | `/cache/clas12/rg-a/production/montecarlo/clasdis_pass2/fa18_inb/clasdis_rga_fa18_inb_45nA_10604MeV-0355.hipo - 0672.hipo` |
| RGA pass-2 data (needed Week 7+) | `[CONFIRM WITH MARIA — can wait until Week 6]` |
| Notes directory | `notes/` |


---

## 4. Cheatsheets

These are reference sections, not tutorials. Read them once now to know what's here, then
come back when you need them.

---

### 4a. Python for this project

For Week 1, you don't need to install anything — use JLab JupyterHub (see Section 4f).
The default Python kernel on JupyterHub has `numpy`, `pandas`, `matplotlib`,
`scikit-learn`, and `uproot` pre-installed. The only package missing is `lightgbm`,
which you install once in a notebook cell with `!pip install --user lightgbm`.

For Week 2 and beyond — when you start running batch jobs or want a reproducible Python
environment for slurm submissions — set up a conda environment from
`suli2026_pid/environment.yml`. See Section 4f for that step (not needed Week 1).

**Reading a ROOT TTree into a DataFrame:**

```python
import uproot
import pandas as pd

tree = uproot.open("pid_training_one_file.root:PhysicsEvents")
df = tree.arrays(library="pd")   # returns a pandas DataFrame
print(df.shape)                  # (n_rows, 57)
print(df.dtypes)
print(df.head())
```

**First look at the data:**

```python
print(df.describe())                        # per-column stats
print(df["mc_matching_pid"].value_counts()) # how many pi+, K+, p, unmatched?
print((df["mc_matching_pid"] == -9999).sum())  # unmatched rows to drop
```

**Selecting truth classes:**

```python
kaons  = df[df["mc_matching_pid"] == 321]
pions  = df[df["mc_matching_pid"] == 211]
protons = df[df["mc_matching_pid"] == 2212]
```

**Minimal BDT pipeline (Week 4):**

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

feature_cols = [
    "beta", "chi2pid",
    "ftof_energy_1A", "ftof_energy_1B", "ftof_time_1A", "ftof_time_1B",
    "ftof_path_1A", "ftof_path_1B",
    "ecin_energy", "ecout_energy", "ecin_time", "ecout_time",
    "ecin_path", "ecout_path",
]
# Binary label: 1 = true K+, 0 = true pi+
# (You'll want to filter to only pi+ and K+ rows before training)
# Replace -9999 sentinels with NaN — LightGBM handles NaN natively.
X = df[feature_cols].replace(-9999, float("nan")).values
y = (df["mc_matching_pid"] == 321).astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
bdt = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                          random_state=42)
bdt.fit(X_tr, y_tr)

probs = bdt.predict_proba(X_te)[:, 1]
print(roc_auc_score(y_te, probs))
print(classification_report(y_te, bdt.predict(X_te)))
```

**Matplotlib pattern (one clear plot, save it):**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(pions["beta"],  bins=100, range=(0.7, 1.1), histtype="step", label="pi+", density=True)
ax.hist(kaons["beta"],  bins=100, range=(0.7, 1.1), histtype="step", label="K+",  density=True)
ax.set_xlabel("beta"); ax.set_ylabel("Density"); ax.legend()
fig.tight_layout()
fig.savefig("figures/beta_pi_vs_K.png", dpi=150)
plt.close(fig)
```

Always call `plt.close(fig)` after saving — on ifarm without a display, leaving figures
open wastes memory.

**Useful pandas one-liners:**

```python
df.loc[df["sector"] == 4, "beta"].describe()  # sector-4 tracks only
df.groupby("mc_matching_pid")["beta"].mean()   # mean beta per truth class
df["ftof_energy_1A"].value_counts(dropna=False, bins=10)  # quick distribution check
df.isna().sum()          # count NaN per column (uproot converts -9999 to NaN in some modes)
(df["beta"] == -9999).sum()  # if -9999 was not converted: count missing
```

---

### 4b. Git workflow for this project

#### First-time setup: GitHub account + SSH key

Do this before you try to clone anything. It is a one-time procedure per machine (ifarm
and/or your laptop). If you have already done it on a given machine, skip to "Clone both
repos" below.

**1. GitHub account**

If you do not already have one, create a free account at [github.com](https://github.com)
(takes about 2 minutes). Then send your GitHub username to Maria. Maria will add you as a
collaborator on both repos:

- `mariakzurek/clas12_analysis_software`
- `mariakzurek/suli2026_pid`

You will receive email invitations to both repositories. **Accept both invitations** before
trying to push anything — without accepting, your pushes will be rejected even if the clone
succeeds.

**2. Generate an SSH key on the machine you will use**

SSH keys let you push to GitHub without typing a password every time. You need one key per
machine (one for ifarm, one for your laptop if you work there too).

```bash
# Check if a key already exists:
ls ~/.ssh/id_ed25519.pub  # or id_rsa.pub for older keys

# If not, generate one (use your email):
ssh-keygen -t ed25519 -C "his@email.example"
# Press Enter at all prompts (use defaults, optional passphrase)

# Print the public key — copy this output:
cat ~/.ssh/id_ed25519.pub
```

The output of `cat` will be a single long line starting with `ssh-ed25519`. Copy the
entire line.

**3. Add the key to GitHub**

Go to [github.com](https://github.com) → click your profile icon (top right) → **Settings** →
**SSH and GPG keys** (left sidebar) → **New SSH key**. Paste the public key into the
"Key" field. Give it a descriptive title — something like `ifarm` or `laptop` — so you
can tell your keys apart if you add more later. Click **Add SSH key**.

To confirm it worked, run:

```bash
ssh -T git@github.com
```

You should see a message like `Hi <username>! You've successfully authenticated`. If you
see `Permission denied (publickey)`, the key was not added correctly — check that you
pasted the full line and saved it.

---

There are two repos. Clone both on your first day on ifarm.

**Important — clone into `/work/`, not `/home/`.** The framework repo contains a
committed Python virtualenv (`python_env/`, ~327 MB) that you do not need. Use
sparse-checkout to skip it — this drops the working tree from ~330 MB to ~10 MB and
avoids filling your quota. The clone commands below assume your `/work/` allocation path
is `/work/clas12/<username>/` (confirm the exact path with `ls /work/clas12/` first —
your username is your CUE username). Your `/home/` quota on ifarm is small;
the `/work/` filesystem is persistent and has a real quota for project files.

> Why `python_env/`? Someone accidentally committed a virtualenv to the upstream
> framework repo. We skip it; the Python environment for this project is built fresh
> from `suli2026_pid/environment.yml` (see Section 4f).

```bash
# Always work under /work/, not /home/ — your /home/ quota is small.
cd /work/clas12/<username>/      # confirm this path with `ls /work/clas12/` first
mkdir -p SULI && cd SULI

# 1. Framework repo — sparse-checkout to skip the 327 MB python_env directory
git clone --no-checkout -b suli_kaon_pid \
    git@github.com:mariakzurek/clas12_analysis_software.git
cd clas12_analysis_software
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/python_env'
git checkout suli_kaon_pid
cd ..

# 2. Analysis repo — small, plain clone
git clone git@github.com:mariakzurek/suli2026_pid.git
```

If `git clone` fails with "No space left on device", check your quotas: `quota -s` for
`/home/`, `df -h /work/clas12/` for `/work/`. The clone should be on `/work/`, not
`/home/`.

**Framework repo (`clas12_analysis_software`) rules:**

The rule: **never commit directly to `suli_kaon_pid`**. Create a feature branch off it,
do your work there, push, and ask Maria to review before merge.

```bash
cd clas12_analysis_software

# Check current state
git status
git log --oneline -10

# Always pull the base branch before starting new work
git pull origin suli_kaon_pid

# Create a branch for a specific task (name it descriptively)
git checkout -b cooper-week1-env-check

# After editing / adding files
git add path/to/changed/file
git commit -m "week1: describe what you did"
git push origin cooper-week1-env-check
```

Then open a pull request on GitHub targeting `suli_kaon_pid` and let Maria know.

**Analysis repo (`suli2026_pid`) rules:**

This is where all your Python notebooks, plots, ML scripts, slurm submission scripts,
and writeup material go. Commit directly to `main` for day-to-day work (it's your repo).

```bash
cd suli2026_pid

git pull origin main

# Add your work — notebooks, figures, scripts
git add notebooks/week1_diagnostics.ipynb figures/beta_vs_p_truth_classes.png
git commit -m "week1: beta vs p diagnostic plots, three truth classes"
git push origin main
```

**Fixing a merge conflict (when it happens):**

1. `git pull origin <branch>` — Git marks conflicting sections with `<<<<<<< HEAD`,
   `=======`, `>>>>>>> <branch>`.
2. Open the file, pick the correct version (or combine them), remove the markers.
3. `git add <file> && git commit` — Git notices the conflict is resolved.

If it looks complicated, ask Maria before guessing. One bad merge is more expensive than
a 10-minute question.

---

### 4c. ifarm + Slurm

**Interactive session for testing (up to 1 hour):**

```bash
srun --pty --partition=ifarm -t 1:00:00 --mem=8G bash
module load clas12
```

**Check queue and cancel:**

```bash
squeue -u $USER          # see your jobs
scancel <JOBID>          # cancel a job
```

**Output goes to:**
- `/volatile/clas12/<username>/` — scratch, deleted after ~2 weeks. Use for intermediate
  files while processing. **Do not put final outputs here.**
- `/work/clas12/<username>/` — persistent working storage. Put ROOT files and model
  checkpoints here.

**Batch submission — running the Groovy pipeline on one HIPO file:**

```bash
#!/bin/bash
#SBATCH --job-name=pid_train_test
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=/volatile/clas12/<username>/logs/pid_test_%j.out

module load clas12
cd /work/clas12/<username>/SULI/clas12_analysis_software

./processing.csh \
    processing_scripts/processing_mc_pid_training.groovy \
    /path/to/one/hipo/file/directory/ \
    /volatile/clas12/<username>/pid_training_test \
    1 10.6041 11
```

Submit with `sbatch submit.sh`. The `1` means process only 1 HIPO file. The `11`
forces `runnum=11` (MC mode, QA always passes).

**Job array for many files (Week 3):**

```bash
#SBATCH --array=0-49   # 50 files, 0-indexed
```

With the array index used to select a slice of the file list. Maria will help set that up
in Week 3.

---

### 4d. ROOT ↔ Python bridge

**Read a TTree into pandas (the main workflow):**

```python
import uproot

# Open and inspect available trees/branches
f = uproot.open("pid_training_one_file.root")
print(f.keys())                          # lists TTrees in the file
tree = f["PhysicsEvents"]
print(tree.keys())                       # lists branch names

# Load into a DataFrame (all branches)
df = tree.arrays(library="pd")

# Load only specific branches (faster for large files)
cols = ["pid", "p", "theta", "beta", "mc_matching_pid"]
df = tree.arrays(cols, library="pd")
```

**Chunk iteration for files too large to fit in memory:**

```python
for batch in tree.iterate(step_size="100 MB", library="pd"):
    # process batch
    pass
```

**Write back to ROOT (if ever needed):**

```python
with uproot.recreate("output.root") as out_file:
    out_file["results"] = {"score": scores_array, "pid": pid_array}
```

You will mostly be reading, not writing, ROOT files. The main output format for model
checkpoints and results is Python (joblib / CSV), not ROOT.

---

### 4e. The Timothy Hayward Framework in One Page

The `clas12_analysis_software` repo processes HIPO files through a five-stage pipeline:

```
HIPO file
    → processing.csh          (tcsh dispatcher, compiles converter, sets environment)
    → processing_*.groovy     (Groovy event loop — reads banks, applies cuts, writes txt)
    → Java fitter layer       (GenericKinematicFitter subclass: applies PID cuts, corrections)
    → Java analyzer layer     (TwoParticles / ThreeParticles: computes kinematics, RICH lookup)
    → space-separated .txt    (one row per track or per event, ~57-98 columns)
    → convert_txt_to_root.cpp (C++ program: reads txt, creates TTree "PhysicsEvents" in .root)
```

**The three fitter classes** (all in `processing_classes/src/extended_kinematic_fitters/`):

| Fitter | What it does | When to use |
|---|---|---|
| `event_builder_fitter` | Passes every `REC::Particle` row through unchanged. No cuts. | RICH studies (preserves index alignment) |
| `analysis_fitter` | Applies refined per-PID fiducial + vertex cuts, electron momentum corrections. | Production SIDIS analyses |
| `monte_carlo_fitter` | Reads `MC::Particle` (truth), not `REC::Particle`. | MC-truth-level event reconstruction |

**`processing_mc_pid_training.groovy` specifically** — this is your main script:

- Uses `analysis_fitter` to check the trigger electron (row 0) passes full electron cuts.
- Then loops directly on `REC::Particle` rows (not through the TwoParticles analyzer),
  which avoids the `getIndex()` index-alignment bug described in `framework_analysis.md §2.5`.
- For each FD π±/K± track that passes vertex + DC-fiducial cuts, reads detector responses
  directly from HIPO banks: `REC::Scintillator` (FTOF), `REC::Calorimeter` (ECAL/PCAL),
  `REC::Cherenkov` (HTCC), `RICH::Particle`.
- Geometrically matches each track to `MC::Lund` truth (|Δφ|<6°, |Δθ|<2°, best match wins).
- Writes **57 columns** per track row. Column map printed at the end of every run.

**How to invoke it:**

```bash
./processing.csh \
    processing_scripts/processing_mc_pid_training.groovy \
    <hipo_directory> \
    <output_basename> \
    [n_files] [beam_E_GeV] [runnum_override]
```

Example (one file, clasdis MC):

```bash
./processing.csh \
    processing_scripts/processing_mc_pid_training.groovy \
    /path/to/clasdis/fa18_inb/ \
    /volatile/clas12/<username>/pid_training_test \
    1 10.6041 11
```

This produces `pid_training_test.txt` (space-separated) and then automatically runs
`convert_txt_to_root` to produce `pid_training_test.root` with a TTree named
`PhysicsEvents`.

**The 57 columns, by group:**

| Columns | Group |
|---|---|
| 1–12 | Event-level: `runnum, evnum, helicity, e_p, e_theta, e_phi, vz_e, Q2, W, x, y, nu` |
| 13–19 | Per-track kinematics: `pid, p, theta, phi, vz, sector, status` |
| 20–34 | ML features (`beta`, `chi2pid`, `nphe_htcc`, FTOF panels 1A/1B {energy, time, path}, ECAL ECin/ECout {energy, time, path}) |
| 35–40 | PCAL + FTOF layer 2 (Connor dropped; Cooper re-evaluates): `pcal_energy, pcal_time, pcal_path, ftof_energy_2, ftof_time_2, ftof_path_2` |
| 41–54 | RICH (cross-check only, NOT training features): `rich_emilay, rich_emico, rich_emqua, rich_best_PID, rich_RQ, rich_ReQ, rich_el_logl, rich_pi_logl, rich_k_logl, rich_pr_logl, rich_best_ch, rich_best_c2, rich_best_RL, rich_best_ntot` |
| 55–57 | MC truth: `mc_matching_pid, mc_parent_pid, mc_match_quality` |

**Missing-value sentinel:** `-9999` means the variable was absent for that track (no RICH
hit, no PCAL deposit, no MC match, etc.). You will need to handle these in the ML
pipeline. Missing values in PCAL and FTOF layer 2 are physically informative — a kaon
that misses the PCAL is different from a pion that misses it.

**For deeper framework questions:** `framework_analysis.md` in the notes directory is a
complete deep-read of the repository. The most relevant sections:
- §2.5 — the `getIndex()` index-alignment bug and why the new script avoids it
- §5b — deep read of `processing_calibration.groovy` (the template the new script was built from)
- §8.1 — the open question about RICH bank presence in MC HIPO files (which you will
  answer on Day 1/2)

---

### 4f. Jupyter notebooks on JLab JupyterHub

A Jupyter notebook is a web-based interactive Python environment. You write code in
cells, run them one at a time, and the output — numbers, tables, plots — appears
inline directly below the cell. It is the right tool for exploring data and making
plots, and it is what we will use for almost all Python work in this project.

**URL and login.** Go to `https://jupyterhub.jlab.org` and sign in with the same CUE
credentials you use for ifarm SSH (the username and password you got from JLab IT). After
login, JupyterHub will ask you to choose a server profile. The default profile usually
works fine. Click "Start" and wait 30–60 seconds for the server to spawn — it only takes this long the first time.

**What you see after login.** A file browser opens on the left side showing your home
directory on ifarm (`/home/<username>/`). Navigate to wherever you cloned `suli2026_pid`
— on ifarm that will be
`/work/clas12/<username>/SULI/suli2026_pid/notebooks/` (not a `~/` path; see Section 4b).
To create a new notebook, click
**"New"** → **"Notebook"** and pick a **Python 3** kernel. The environment already has
`uproot`, `pandas`, `numpy`, `matplotlib`, and `scikit-learn` available by default
because the JupyterHub server mounts the CLAS12 Python stack. If `lightgbm` is missing,
install it in a notebook cell:

```python
!pip install --user lightgbm
```

You only need to run that cell once; it installs into your user directory and persists
across sessions.

**Basic notebook usage — minimum to be productive.**

- **Running cells:** Type code in a cell and press `Shift+Enter` to run it and move to
  the next cell, or `Ctrl+Enter` to run it and stay in the same cell. Use `Shift+Enter`
  for normal flow; use `Ctrl+Enter` when you want to re-run a single cell repeatedly.

- **Markdown cells:** Change the cell type dropdown from **"Code"** to **"Markdown"** to
  write formatted notes or section headings between code blocks. Run a Markdown cell the
  same way (`Shift+Enter`) to render it. This is useful for recording what a block of
  code is doing or noting a result.

- **Saving and checkpointing:** `Ctrl+S` saves the notebook manually. JupyterHub also
  auto-saves every few minutes. The file is saved as a `.ipynb` file in whatever
  directory you opened it from — which, if you started from inside the `suli2026_pid`
  repo, means it is already in the right place to commit to Git. Commit notebooks to the
  analysis repo the same way you commit any other file (`git add`, `git commit`,
  `git push`).

**One practical note.** The JupyterHub server runs on ifarm compute resources and has
direct access to `/work/clas12/<username>/` and `/volatile/clas12/<username>/` — you
can open ROOT files from those paths in a notebook cell the same way you would from
an interactive shell. You do not need to copy files to your laptop.

**Optional: conda environment for batch jobs (Week 2+)**

When you start submitting batch jobs to slurm — which won't have JupyterHub's
pre-installed kernel — you'll want a reproducible Python environment. The `suli2026_pid`
repo has an `environment.yml` for exactly this.

**Before creating the env, redirect conda's package cache to `/work/`.** Conda's default
is `~/.conda/pkgs/`, which lives on your small `/home/` quota and will fill up after a
single env install. Do this ONCE per ifarm account:

```bash
# Create /work/-based directories for conda
mkdir -p /work/clas12/<username>/conda/pkgs
mkdir -p /work/clas12/<username>/conda/envs

# Tell conda to use them
conda config --add pkgs_dirs /work/clas12/<username>/conda/pkgs
conda config --add envs_dirs /work/clas12/<username>/conda/envs

# Verify
conda config --show pkgs_dirs envs_dirs
```

Then create and activate the env:

```bash
cd /work/clas12/<username>/SULI/suli2026_pid
conda env create -f environment.yml
conda activate suli2026_pid
```

Confirm conda is available first: `which conda` (or try `module load anaconda` if it's
not in your PATH). The env contains numpy, pandas, matplotlib, scikit-learn, uproot,
awkward, lightgbm, xgboost — runtime libraries only, no jupyterlab (use JLab JupyterHub
for notebooks). **You do not need this in Week 1** — only set it up when you start
writing slurm submission scripts (~Week 2-3).

If you hit 'No space left on device' during install, your `~/.conda/pkgs/` is full from
a previous failed attempt — clean it with `rm -rf ~/.conda/pkgs/*` and rerun after
confirming the `pkgs_dirs` redirect above is in place.

---

## 5. Week 1 Task List

Cooper arrives Tuesday. Tuesday and Wednesday are ANL lab orientation (badging, IT
account setup, building access, general onboarding). Project work does not start until
orientation clears, which in practice means Wed/Thu is the first real project day —
and even Thursday may be a half day if orientation overflows. The four tasks below are
the Week-1 project work and will likely span Thursday through the following Monday or
Tuesday (rolling into calendar Week 2). Work through them in priority order. Each task
has a "done when" — don't call a task done until the criterion is met.

---

### Task 1 — Read docs + meet Maria (highest priority, ~half day; do this on your first project day after orientation)

**What:** Read this document and `cooper_10week_plan.md` front to back. Then meet Maria.

**What to cover in the meeting:**
- Where are the clasdis MC files on ifarm?
- Is your JLab IT account fully provisioned? Can you SSH to ifarm?
- High-level walk through the project goals and the 10-week plan.
- Maria's expectations for the week and the best channel for daily questions (Slack).

**Done when:** You have read both documents, the ifarm path to clasdis is confirmed with
Maria, and you have exchanged contact details.

---

### Task 2 — ifarm environment check (~2 hours; do once Task 1 is done)

**What:** SSH to ifarm, load the CLAS12 module, find the clasdis files at the path Maria
gave you, and run `hipo-utils -dump` on one file to inspect the bank structure.

```bash
ssh <username>@ifarm.jlab.org
module load clas12
hipo-utils -dump /path/to/one/clasdis/file.hipo
```

**This answers a real open research question.** Check if `RICH::Particle` is populated in the clasdis MC
HIPO files. If it is empty, the RICH cross-check columns in your ntuple will all be
`-9999` and some downstream plans will need to adjust. If it is populated, great.

**Done when:** You can paste a snippet of the `hipo-utils -dump` output showing which
banks are present — specifically whether `RICH::Particle` appears and whether it has
rows. Share the snippet with Maria via Slack.

---

### Task 3 — Clone repos; run both Groovy scripts on one file; open output in Python (~half day; second project day)

**What:** Clone both repos. Run `processing_calibration.groovy` on ONE small clasdis
file end-to-end as a build-chain validation. If that succeeds, run
`processing_mc_pid_training.groovy` on the same file and load the result into Python.

**Step 1 — clone:**

Follow the sparse-checkout instructions in Section 4b ("Clone both repos") — clone into
`/work/clas12/<username>/SULI/`, not your `/home/`. The framework repo requires
sparse-checkout to skip `python_env/` (~327 MB); the analysis repo is a plain clone.

**Step 2 — calibration script (build-chain smoke test):**

```bash
cd clas12_analysis_software
./processing.csh \
    processing_scripts/processing_calibration.groovy \
    /path/to/clasdis/one_file_directory/ \
    /volatile/clas12/<username>/calib_test \
    1

ls -lh /volatile/clas12/<username>/calib_test.root
```

If this fails, stop and debug the build environment before going further. Common
failures: `module load clas12/pro` not done, classpath wrong, Java version mismatch.
Check `processing.csh` stderr carefully.

**Step 3 — training script:**

```bash
./processing.csh \
    processing_scripts/processing_mc_pid_training.groovy \
    /path/to/clasdis/one_file_directory/ \
    /volatile/clas12/<username>/pid_training_test \
    1 10.6041 11
```

The script prints the column index map at the end. Note any columns that look suspicious
(all zeros, all -9999).

**Step 4 — open in Python:**

```python
import uproot, pandas as pd

df = uproot.open("/volatile/clas12/<username>/pid_training_test.root:PhysicsEvents") \
           .arrays(library="pd")

print(df.shape)                              # should be (N_tracks, 57)
print(df.head())
print(df.describe())
print(df["mc_matching_pid"].value_counts())  # key: how many K+, pi+, p, unmatched?
```

If the ROOT file is empty (zero rows), check the txt file first:
`wc -l /volatile/clas12/<username>/pid_training_test.txt`. If the txt has rows but the
ROOT file is empty, the issue is in `convert_txt_to_root`. If both are empty, check
whether the HIPO file had events passing the electron cut.

**Done when:** You have a DataFrame on screen with `df.shape[1] == 57` and
`df["mc_matching_pid"].value_counts()` showing at least some rows with PID 321 (K+) and
211 (pi+).

---

### Task 4 — Diagnostic plots + brief summary to Maria (~half day; third project day or rolling into Week 2)

**What:** Make 2–3 diagnostic plots from the DataFrame you produced in Task 3. At
minimum:

1. **beta vs p colored by truth class** — the first time you see the training problem
   as a picture. Three overlaid histograms or a 2D scatter, one color per MC truth class
   {pi+, K+, p}.

```python
import matplotlib.pyplot as plt
import numpy as np

matched = df[df["mc_matching_pid"] != -9999].copy()

fig, ax = plt.subplots(figsize=(8, 6))
for pid_val, label, color in [(211, "pi+", "steelblue"),
                               (321, "K+",  "tomato"),
                               (2212, "p",   "forestgreen")]:
    subset = matched[matched["mc_matching_pid"] == pid_val]
    ax.hist2d(subset["p"], subset["beta"],
              bins=[80, 80], range=[[0.5, 5.0], [0.6, 1.1]],
              alpha=0.6, label=label, cmap="Blues" if pid_val==211 else
                                          "Reds" if pid_val==321 else "Greens")
ax.set_xlabel("p (GeV/c)")
ax.set_ylabel("beta")
ax.set_title("beta vs p by MC truth class")
fig.tight_layout()
fig.savefig("figures/beta_vs_p_truth_classes.png", dpi=150)
plt.close(fig)
```

2. **chi2pid distribution per truth class** — this is the baseline discriminant; the ML
   has to beat it.

```python
fig, ax = plt.subplots(figsize=(7, 5))
for pid_val, label in [(211, "pi+"), (321, "K+"), (2212, "p")]:
    subset = matched[matched["mc_matching_pid"] == pid_val]
    ax.hist(subset["chi2pid"].clip(-10, 10), bins=100, histtype="step",
            label=label, density=True)
ax.set_xlabel("chi2pid"); ax.set_ylabel("Density"); ax.legend()
fig.tight_layout()
fig.savefig("figures/chi2pid_by_truth_class.png", dpi=150)
plt.close(fig)
```

Save both PNGs to `figures/` and commit to the analysis repo. Additional plots are
welcome but not required this week — they move to Week 2.

**Week 1 summary:** Send Maria a brief Slack message or email with: the two plots
attached, what you found in the `hipo-utils` dump (RICH bank populated or not?), and
any questions that came up during Tasks 1–3. No separate written document is required
for Week 1. The formal written summary moves to the end of Week 2.

**Done when:** At least 2 PNGs saved and committed to the analysis repo, and a brief
Slack/email with the plots sent to Maria.

---

## 6. Reading List

Read these in the order listed, at the times listed. Do not try to read all of them in
Week 1 — most of this is spread across 10 weeks.

### 1. scikit-learn user guide — specific sections (2–3 hours total, spread across Week 1–2)

Go to [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
and read:

| Section | What | When | Time budget |
|---|---|---|---|
| 1.10 Decision Trees | How a single tree works; the splitting criterion | Week 1 | 30 min |
| 1.11 Ensemble methods | Skim; focus on the Gradient Boosting subsection (primary model). The Random Forests subsection is prior-art context. | Week 2 | 45 min |
| 3.1 Cross-validation | The train/val/test split rationale; why not to touch the test set | Week 2 | 30 min |
| 3.3 Metrics | Classification report, ROC, AUC, confusion matrix | Week 2 | 30 min |
| Common pitfalls | Short section on what goes wrong. Read this. | Week 1 | 20 min |

The pitfalls section is easy to skip and frequently saves someone's project. Read it.

### 2. Connor Pecar's analysis note — prior-work reference

File: [Note](https://www.jlab.org/Hall-B/shifts/admin/paper_reviews/2026/cpecar_anaNote_BSA_dihadWithKaons_V3.pdf-1043422-2026-02-11-v6.pdf)

Connor Pecar's CLAS12 analysis note on dihadron BSA with kaons. Section IV is relevant
prior work on ML-based PID refinement. Useful reference for technique ideas — NOT a
recipe to copy. Our channel is different, our MC release is different, and the feature
set should be driven by what the data support, not by replication.

### 3. Optional — scikit-learn calibration (30 min, Week 2)

The scikit-learn documentation on `CalibratedClassifierCV` and the reliability diagram
example. Relevant to Week 6 but useful to skim early so Week 6 isn't a surprise.

---

## 7. How to Ask for Help

**Maria:** Python + ML questions, physics interpretation, "is this result sensible",
anything about the project direction. Don't wait 2 hours guessing at a Python error
before asking. The 30-minute rule: if you've tried something twice and looked online
for 30 minutes without a clear answer, ask on Slack.

**Stack Overflow / scikit-learn docs:** Pandas, numpy, matplotlib, sklearn API questions.
These are well-documented and Stack Overflow coverage is good.

**ChatGPT / LLMs (use cautiously, as a learning tool not an answer machine).**
Large language models can be useful for: explaining a concept you read but didn't follow,
generating a first-pass code template you then rewrite, debugging a specific error message,
or rubber-ducking — articulating a problem out loud often surfaces its own solution.
They are dangerous for: physics interpretation (frequently confidently wrong about CLAS12
specifics, ML statistics, and analysis conventions), choosing analysis methods
(it will recommend whatever is most common on the internet, which is rarely what your
analysis needs), and producing code you don't understand. The rule: never paste LLM
output into your code or notes without understanding every line. If you can't explain
*why* the code does what it does, you don't understand it and you will not be able to
defend it in a group meeting. Treat ChatGPT like a knowledgeable but unreliable colleague
who has never actually run a CLAS12 analysis — useful for general questions, untrustworthy
for specifics.

Later in the summer (probably after Week 3, once you have a feel for the project), we may
set you up with OpenCode using the Argonne API, which has tighter integration with code
editing. That's a separate conversation.

**Two-attempts rule:** Try it yourself. If your first attempt doesn't work, try a
different approach. If the second attempt doesn't work within 30 minutes, ask rather than
guess further. 

---

## 8. End Note

You don't need to master all of this in Week 1. The plan is 10 weeks, and Week 1 is
deliberatly narrow: get the environment working, get your first plots, and look at the
data once before any model. That's the entire goal.

The first model is Week 4. The first headline number is Week 5. The data-driven
validation is Weeks 8–9. There is time.

If something in the environment is broken on Day 1, fix it before doing anything else. A
working ifarm setup is the foundation of every other task. If it takes two days to get
access sorted, that is not a crisis — the 10-week plan has fallback options written in for
exactly this situation (see the Week 1 risk register in `cooper_10week_plan.md`).

Good luck. The problem is real, the data exists, and the project has a clear path from
where you start to a result you can put on a poster. The rest is work.

---

*Document written: May 2026. Written for Cooper, SULI student, summer 2026.*

