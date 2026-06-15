# Cooper — Week 4 Walkthrough: Training Your First BDT

The official Week-4 done-when criteria live in `~/CLAS/SULI/suli2026_pid/notes/cooper_10week_plan.md`; this walkthrough is how you actually get there. Everything you need to run the work is in this document.

---

**One-time conda env setup.**
The `suli2026_pid` conda env does not include `pyarrow` (which the training scripts need
to read and write parquet files). Run this once on ifarm:

```bash
conda activate suli2026_pid
conda install -c conda-forge pyarrow
```

Verify it worked:

```bash
python -c "import pyarrow; print(pyarrow.__version__)"
```

If that prints a version number, you're set.

---

## 1. What you are doing this week

The Week 4 goal is to train a binary classifier that, given an event-builder-identified
K+ track, answers the question: "is this a real kaon, or a pion that fooled the
event builder?" You will then show that your classifier beats the chi2pid baseline at that
task, measured per `(p, θ)` bin on a held-out test set.

This is the simplest version of the project's headline question, and it is the right place
to start. The chi2pid variable is essentially one number — a pull of the measured beta
against the kaon hypothesis. Your BDT will look at a dozen or more detector observables
simultaneously: FTOF energies and times across both panel layers, ECAL inner and outer
deposits, possibly HTCC hits — everything the track left behind. The classifier learns
combinations of those signals that separate kaons from pions better than chi2pid alone.
Everything in Weeks 5 through 9 builds on the pipeline you establish now.

Why binary K-vs-π first, not K-vs-π-vs-proton? Because binary is simpler to debug. If
something is wrong — a data leak, a feature that should not be there, a calibration
failure — it shows up in a binary model as a clearly wrong ROC curve or a reliability
diagram that looks nothing like the diagonal. Adding a third class multiplies the ways
things can go wrong before you have any intuition for what "right" looks like. You will
add protons in v2, after you have a working binary model and a per-bin contamination map
showing you exactly where protons are contributing. That decision is yours to make at the
end of Week 4 (Section 6 below explains how).

**Your Week 4 tasks, in order:**

1. Produce the file-level train/val/test split and commit it.
2. Pick the upper momentum bound for the binary model from your existing plots.
3. Run `build_dataset.py` to produce the three parquet files.
4. Run `train_bdt.py` to fit and calibrate the BDT. Inspect the outputs.
5. Run `evaluate.py` to get the per-bin comparison against chi2pid. Inspect the plots.
6. Decide whether protons enter v2 and document the decision.

---

## 2. The five things to understand before you write any code

Read this section before opening any training script. These concepts come up everywhere
in the pipeline, and understanding them in advance will make every error message and plot
you see make sense immediately.

---

### 2.1 What a boosted decision tree actually is

Start with a single decision tree. A decision tree is just a cascade of if-else rules on
feature values:

```
if beta < 0.91:
    if ftof_energy_1A < 3.5:
        predict PION
    else:
        predict KAON
else:
    if chi2pid < 0.8:
        predict KAON
    else:
        predict PION
```

Each split point is chosen to maximize the separation between the two classes at that
node. A shallow tree (two or three levels deep) is weak — it can only draw a few
boundaries in feature space and misclassifies a lot of examples. But if you grow the
tree very deep — say, 20 levels — it memorizes the training data: every training example
falls into its own leaf with a single label, giving you perfect training accuracy and
terrible accuracy on anything new. That is overfitting.

Boosting solves this differently. Instead of building one deep tree, you build a sequence
of shallow trees, where each tree focuses on correcting the mistakes of all the trees
before it. Concretely:

```
Tree 1:  fit the labels directly.
         Get predictions → some are right, some wrong.

Tree 2:  fit the *residuals* (errors) of Tree 1.
         Add Tree 2's predictions to Tree 1's.

Tree 3:  fit the residuals of Trees 1+2.
         Add to the running sum.

...

Final score: sum of all tree predictions, squeezed through a sigmoid
             so it lives in [0, 1].
```

Each individual tree is intentionally weak (shallow, maybe 4–6 levels). The ensemble is
strong because each tree focuses on what the previous ones got wrong. This is the BDT:
Boosted Decision Tree.

You are using LightGBM, which is a specific, fast implementation of this algorithm.
One internal difference worth knowing: LightGBM grows trees leaf-wise (it always splits
the leaf with the highest potential gain, wherever that leaf is) rather than level-wise
(filling out one full level before going deeper). This makes it faster and usually more
accurate, but it means `max_depth` and `num_leaves` interact in a non-obvious way. For
Week 4 you will use the defaults from the plan (`max_depth=6`), so this is just
background knowledge — you do not need to tune it yet.

---

### 2.2 The hyperparameters you will see and what they control

You will set these when you call `LGBMClassifier(...)`. Here is what each one does.

**`n_estimators=200`** — the number of trees in the ensemble. More trees means more
capacity and more training time. If you make it too large the model starts memorizing
noise and val loss diverges from train loss. If you make it too small the model
underfits and the ROC curve will be flat in the low-false-positive region.

**`learning_rate=0.05`** — how much each new tree is allowed to move the ensemble's
prediction. Smaller values mean each tree contributes less, so you need more trees to
reach the same total fit. But smaller learning rates generalize better, because the
corrections are more conservative and the model does not chase noise. This is the most
important hyperparameter to understand intuitively.

The learning_rate ↔ n_estimators tradeoff looks like this: if you set `learning_rate=0.5`
you can get to the same training loss with 20 trees, but the model often generalizes
worse because each tree is doing too much work. If you set `learning_rate=0.01` you need
800+ trees to converge, but the result is often slightly more accurate on new data. The
pair `(n_estimators=200, learning_rate=0.05)` is a reliable middle ground for a dataset
this size. You are not tuning these in Week 4. Week 7 is for tuning.

**`max_depth=6`** — the maximum depth of each individual tree. Six levels means at most
64 leaves per tree, which is plenty to capture feature interactions without severe
overfitting on a million-row dataset.

**`min_child_samples=20`** — the minimum number of training examples that must fall into
a leaf for it to be created. This is LightGBM's default and you are not changing it.
If you made it very small (say, 1), the model would create leaves for single outlier
points and overfit badly. If you made it very large, it would undersplit and miss
fine-grained boundaries.

**`objective='binary'`** — tells LightGBM you are doing binary classification and to
optimize log-loss (cross-entropy). The output before the sigmoid is a raw "logit"; after
the sigmoid it is a score in [0, 1].

**`random_state=42`** — a fixed seed for reproducibility. Use 42 everywhere in this
project. That way, if you or Maria re-run the exact same code, you get the exact same
model.

---

### 2.3 The LGBMClassifier API in five lines

Here is the exact code you will use. Read each line carefully.

```python
import lightgbm as lgb

clf = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    random_state=42, objective='binary'
)

clf.fit(X_train, y_train, sample_weight=w_train)

scores = clf.predict_proba(X_test)[:, 1]   # probability of class 1 (kaon)

importances = clf.feature_importances_     # which features mattered most
```

Line by line:

`clf = lgb.LGBMClassifier(...)` — creates the model object with those hyperparameters.
Nothing is trained yet; this just configures what training will do.

`clf.fit(X_train, y_train, sample_weight=w_train)` — this is where training actually
happens. `X_train` is a 2D array of shape `(n_samples, n_features)` — your feature
matrix. `y_train` is a 1D integer array where 1 = true kaon, 0 = true pion.
`sample_weight` is optional (see Section 2.4 below). For Week 4 v1, pass
`sample_weight=None` or just omit it.

`clf.predict_proba(X_test)[:, 1]` — this is the most common source of confusion for
people new to sklearn-style classifiers. `predict_proba` returns an array of shape
`(n_samples, 2)`. Column 0 is the probability that the track is class 0 (pion). Column
1 is the probability that the track is class 1 (kaon). You almost always want column 1.
The `[:, 1]` selects that column for all rows.

`clf.feature_importances_` — a 1D array, one number per feature, measuring how much each
feature contributed to the model's splits (by gain: the average improvement in loss when
the feature was used). High importance means the classifier found that feature useful.
Low importance means it was mostly ignored. You will plot these — they tell you whether
the classifier learned something physically reasonable.

---

### 2.4 What `sample_weight` does and why you might want it

Every training example contributes equally to the loss by default. But your training MC
might not have the same `(p, θ)` distribution as real data. For example, if your MC
generates three times as many low-momentum pions as you would see in the actual RGA
data, the classifier learns that low-momentum pion tracks are more common than they
really are. When you apply it to data, it will be miscalibrated in exactly those regions.

`sample_weight` is the lever that fixes this. If a track in the MC comes from a
`(p, θ)` bin that is over-represented compared to data, you give it a weight less than 1.
If it comes from an under-represented bin, you give it a weight greater than 1. The
effect is that the classifier's loss function sees a training distribution that matches
the data distribution, even if the raw MC does not.

The prior analysis used this exact technique: a 15×15 grid in `(p, θ)` where each MC track's weight
equals `f_data(p, θ) / f_MC(p, θ)` — the ratio of the data density to the MC density in
that bin. You will likely do the same thing, but only after your Week-3 audit tells you
whether the `(p, θ)` distributions actually disagree.

For Week 4 v1, run unweighted first. The unweighted model already tells you whether the
features are informative. If the `(p, θ)` distributions agree well (which you can check
from your Week-3 audit metrics), the weights would be close to 1 anyway and the
unweighted model is fine. If they disagree significantly, you add the reweighting in
Week 5. The `--reweight-map` argument to `train_bdt.py` exists for exactly this — omit
it for v1, add it for v2.

---

### 2.5 What probability calibration is and why raw scores are not probabilities

A BDT score of 0.8 does NOT mean "there is an 80% chance this track is a kaon." It means
"this track looks more kaon-like than a track with score 0.5." The score is a
monotonically increasing indicator of kaon-ness, but the absolute values are not
calibrated to match actual fractions. This distinction matters for physics: if you want
to say "tracks above this threshold have at most 10% pion contamination," you need the
score to actually be a probability. Without calibration, you can only make relative
statements.

A reliability diagram makes the miscalibration visible. You bin all test tracks by their
BDT score — say, ten bins from [0.0, 0.1), [0.1, 0.2), … [0.9, 1.0]. For each bin you
compute the fraction of tracks that are truly kaons. If the classifier were perfectly
calibrated, the fraction in the [0.6, 0.7) bin would be between 60% and 70%. A typical
uncalibrated BDT looks S-shaped or has an exaggerated spread: the fraction in the
[0.6, 0.7) bin might be 85%, while the fraction in the [0.3, 0.4) bin might be 5%. The
predicted scores are spread wider than the actual probabilities.

Calibration fixes this. You fit a small logistic regression (one sigmoid function, two
parameters) that maps raw BDT scores to actual probabilities. This is called Platt
scaling. In code:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(
    estimator=clf,      # already-trained BDT
    method='sigmoid',   # Platt scaling
    cv='prefit',        # clf is already fitted; just fit the sigmoid
)
calibrated_clf.fit(X_cal, y_cal)   # X_cal, y_cal: the calibration slice
```

The critical rule: the calibration set must be held out from training. You carve 20% of
the training rows for calibration before you fit the BDT. The BDT trains on the other
80%. Then you fit the calibrator on the 20% slice. The validation and test sets are never
touched during calibration. `train_bdt.py` handles this split automatically via
`--calibration-frac 0.2` — you do not need to do it manually. But you do need to inspect
the reliability diagram it produces. Before calibration the diagram will probably be
somewhat S-shaped. After calibration it should sit much closer to the diagonal.

LightGBM probabilities tend to be better calibrated than random forests out of the box,
so the improvement may be modest. That is fine — the calibration step is still worth
doing, because "close to calibrated" is not "calibrated."

---

## 3. Your first task: producing the file split

Before anything else, you need to split your MC ROOT files into train, validation, and
test sets. This sounds trivial — why not just split the events randomly? The reason is
that events from the same ROOT file share run conditions, beam current, and detector
state. A model trained on events from the same files it will be evaluated on can
memorize file-level patterns — specific noise levels, occasional weird run conditions,
detector dead-time artifacts — that have nothing to do with physics. Your test numbers
would look better than they really are because the model memorized the test files.

The fix is to split by file, not by event. All events in a given file go to one of
{train, val, test}, never to two. That way, when you evaluate on the test set, the model
has genuinely never seen those files.

**The wrinkle: `mc_v01/` has bimodal file sizes.** Most files are ~247 MB, but some are
~488 MB — roughly 2× larger, which means roughly 2× as many events. A naive split by
file count can produce a badly unbalanced event-count split: if the large files happen
to cluster into one set, that set will have disproportionately more training events even
if the file counts look right. You need to balance approximate event counts, not file
counts.

File size in bytes is a good proxy for event count — a ~488 MB file really does have
roughly twice the events of a ~247 MB one, without you having to open any ROOT files.
The approach: get the byte size of each file, then use greedy bin-packing to assign files
to train/val/test so that the total bytes in each set is as close to the 70/15/15 target
as possible. Sort files by size descending, then for each file assign it to whichever
split is currently furthest below its byte budget. This is a standard greedy
approximation to the bin-packing problem and is typically within 1% of the target ratio.

Run this from the repo root (`~/CLAS/SULI/suli2026_pid/`), in a notebook cell or an
interactive Python session on ifarm:

```python
import os, random
from pathlib import Path

mc_dir = Path("/volatile/clas12/zurek/SULI/mc_v01")
files = sorted(mc_dir.glob("*.root"))
sizes = {f.name: f.stat().st_size for f in files}

# Shuffle (fixed seed for reproducibility), then sort by size descending
random.seed(42)
shuffled = list(sizes.items())
random.shuffle(shuffled)
shuffled.sort(key=lambda x: -x[1])

# Greedy bin-packing into three buckets with target ratios 0.7 / 0.15 / 0.15
total = sum(sizes.values())
targets = {"train": 0.70 * total, "val": 0.15 * total, "test": 0.15 * total}
buckets = {"train": [], "val": [], "test": []}
loads   = {"train": 0,   "val": 0,   "test": 0}

for name, size in shuffled:
    # Assign to whichever split is most "under target" relative to its budget
    deficit = {k: targets[k] - loads[k] for k in buckets}
    pick = max(deficit, key=deficit.get)
    buckets[pick].append(name)
    loads[pick] += size

for split in ("train", "val", "test"):
    frac = loads[split] / total
    print(f"{split}: {len(buckets[split])} files, {loads[split]/1e9:.1f} GB, {frac:.3f} of total")
    Path("slurm").mkdir(exist_ok=True)
    with open(f"slurm/{split}_files.txt", "w") as f:
        f.write("\n".join(sorted(buckets[split])) + "\n")
```

The three files land in `slurm/train_files.txt`, `slurm/val_files.txt`, and
`slurm/test_files.txt`. The printed fractions should read roughly `0.700 / 0.150 / 0.150`
— if any split is off by more than a few percent, something went wrong (probably the size
distribution is more extreme than expected; ping Maria).

**Verify before committing.** Check that the total file count adds up, and confirm the
size fractions look right:

```bash
wc -l slurm/train_files.txt slurm/val_files.txt slurm/test_files.txt
ls /volatile/clas12/zurek/SULI/mc_v01/*.root | wc -l
```

The line counts must sum to the total ROOT file count. The size fractions printed by the
script are your event-count verification — that is the check that matters here.

Then commit all three files to git:

```bash
git add slurm/train_files.txt slurm/val_files.txt slurm/test_files.txt
git commit -m "week4: size-weighted file-level 70/15/15 train/val/test split, seed=42"
git push origin main
```

Commit them before running anything else. The split files are a contract: the entire rest
of the Week 4 analysis depends on which files are in which set. If you lose them and
re-run the splitter, any subsequent training run becomes incomparable to the first.

**Done when:** three text files exist under `slurm/`, their line counts sum to the total
ROOT file count, and they are committed and pushed.

---

## 4. The momentum range decision: yours to make

The most important physics decision in Week 4 is the upper momentum bound for the binary
K/π model. You will pass this as `--p-max <value>` to `build_dataset.py`.

Here is why this matters. At high momentum, kaons and pions have nearly the same velocity
at a given momentum — that is the core of the problem you are solving. But "nearly the
same" eventually becomes "indistinguishable." Above some momentum the K and π β-bands
completely overlap, chi2pid loses all discrimination, and your detector features carry
essentially no information about which species the track is. If you train your BDT to
classify tracks in that region, you are asking it to learn from pure noise. The trees
will find spurious correlations in the training data that do not generalize, and —
crucially — those spurious fits will degrade performance in the momentum bins where the
classifier *can* learn something useful, because the model wastes capacity trying to
classify what is unclassifiable.

The practical consequence: a model trained all the way to 5 GeV will likely perform
worse in the 2–3 GeV range than a model trained only up to 3 GeV, because the
unclassifiable high-p tracks pollute the gradient.

**How to pick your cutoff.** Look at your existing β-vs-p plot colored by truth class
(you made this in Week 1, Task 4). Find the momentum where the K+ and π+ β-bands merge
— where you can no longer see two separate bands. Also look at your per-bin contamination
matrix from Week 2: find the bins where baseline contamination hits 40–50%. And look
at your Week-3 audit drift metrics at high p — if feature distributions start becoming
noisy or failing the drift thresholds above some momentum, that confirms the classifier
will struggle there too. The momentum where you see all three signals converge is roughly
your cutoff.

Order-of-magnitude guidance: you are looking for somewhere in the range 2.5–3.5 GeV.
The prior analysis used `1.0 < p < 3.0 GeV` as the training range for the K+ classifier,
which gives you a sanity-check anchor. Your data and MC are different, and your
β-vs-p plots may show the bands merging at a slightly different momentum. Use your own
plots; do not just copy that number.

Once you have chosen a value, document it. Create a file
`~/CLAS/SULI/notes/cooper_week4_decisions.md` with at least:

```markdown
# Cooper — Week 4 Decisions

## Momentum cutoff for binary K/π model

Chosen: --p-max X.X GeV

Rationale: [one paragraph pointing to contamination plots va momentum
where separation collapses]

Supporting plot: contamination plots and maps vs momentum and theta
```

This matters because anyone reading your results later needs to know why the training
range stops where it does. This decision file is where your reasoning lives.

**Done when:** you have chosen a value, written a one-paragraph justification in
`cooper_week4_decisions.md`, and can point to the plot that supports it.

---

## 5. The three scripts, one at a time

Three scripts live under `scripts/training/`. You interact with them from the command
line. You do not need to read their internals to use them. Here is what each does, how
to invoke it, what it writes, and what to look for afterward.

---

### 5.1 `build_dataset.py` — ROOT files → parquet trio + manifest

**What it does.** Reads every ROOT file listed in your three split text files, selects
EB-identified K+ tracks, filters to your chosen momentum range, applies your audit's
KEEP feature list, assigns binary labels (1 = true K+, 0 = true π+), and writes three
parquet files: `train.parquet`, `val.parquet`, and `test.parquet`. It also writes a
`manifest.json` — your build's provenance record.

**What the parquet files contain.** Every row is one EB-identified K+ track. The columns
are always in this order (the scripts select by name, not position, so order doesn't
matter to you — but it helps to know what's there):

- `p`, `theta`, `phi`, `vz`, `sector` — track kinematics and sector (`float32` /
  `int32`). Always present.
- `chi2pid` — the experiment's standard kaon pull variable. Required by `evaluate.py` for
  the baseline comparison.
- `<feature columns>` — whatever features your audit marked KEEP, in the order they
  appear in `scripts/training/feature_list.txt`. Sentineled values are NaN.
- `pid` — always 321 (EB called this a K+). `int32`.
- `mc_matching_pid` — the true PDG code from the MC match: 321 for real kaons, 211 for
  pions, 2212 for protons, −9999 for unmatched tracks.
- `label` — the classification target. In `train.parquet` and `val.parquet`: `1` for true
  K+ (`mc_matching_pid == 321`), `0` for true π+ (`mc_matching_pid == 211`), stored as
  non-nullable `int8`. In `test.parquet`: same encoding but stored as nullable `Int8`;
  protons and unmatched tracks have `label = NaN`. That is intentional — `evaluate.py`
  uses `label.notna()` rows for the K/π efficiency and contamination metrics, and
  `mc_matching_pid == 2212` rows for the proton contamination map, all from the same file.

**What the manifest contains.** After the build, open `manifest.json` in a notebook to
sanity-check the run:

```python
import json
with open("/volatile/clas12/zurek/SULI/dataset_v01/manifest.json") as f:
    m = json.load(f)
print(m["p_max"])            # the --p-max you passed
print(m["train"]["n_rows"])  # row counts per split
print(m["feature_list"])     # list of feature column names that went in
print(m["git_sha"])          # repo state at build time
```

The manifest also records `features_file_sha256` (so you can detect if
`feature_list.txt` changed between builds), `missing_fraction` (fraction of split-file
stems that had no matching ROOT file), and `build_timestamp`. It is your paper trail if
you ever wonder "which features did model v01 actually train on?"

**The command you will type:**

```bash
conda activate suli2026_pid
cd ~/CLAS/SULI/suli2026_pid

python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/zurek/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /volatile/clas12/zurek/SULI/dataset_v01 \
    --features-file scripts/training/feature_list.txt \
    --p-max 3.0
```

Replace `3.0` with whatever you chose in Section 4. The full set of optional flags:
`--max-files N` (only read N files — use `--max-files 2` for a smoke test before
processing the full dataset), `--allow-missing-files` (continue even if some split-file
stems have no matching ROOT file; normally the script refuses if more than 5% are
missing), `--overwrite` (re-run if the output directory already exists).

**What to look at after.** The script prints a summary table at the end. Check:

- Row counts look plausible: the training parquet should be the largest; val and test
  should each be roughly half as large (70/15/15 file split).
- The truth breakdown in the manifest: you should see both kaons (label=1) and pions
  (label=0) in each split. If one class is missing from val or test, something went
  wrong with the split files.
- Open `manifest.json` as described above and confirm `p_max`, `n_rows`, and
  `feature_list` look right.

**Common failure: missing files.** If `build_dataset.py` reports `missing_fraction > 5%`
it will refuse to build and tell you to use `--allow-missing-files`. This happens when
ROOT file basenames in your split text files do not match what is actually in `mc_v01/`.
Diagnose with:

```bash
diff <(sort -u slurm/train_files.txt) \
     <(ls /volatile/clas12/zurek/SULI/mc_v01/ | sed 's/\.root$//' | sort -u)
```

Any lines in the diff are files that exist in one place but not the other.

---

### 5.2 `train_bdt.py` — fit the BDT and calibrate it

**What it does.** Loads `train.parquet` and `val.parquet` from your dataset directory.
Carves 20% of the training rows as a calibration slice (stratified by label). Trains the
LightGBM BDT on the remaining 80%. Fits a Platt calibrator on the 20% slice. Writes
everything you need: `model.joblib` (the calibrated model object), a
`training_summary.csv` with AUC and Brier score on both train and val (before and after
calibration), a `reliability_diagram.png`, a `roc_val.png`, and a
`feature_importance.png`. It never touches `test.parquet` — that file is for
`evaluate.py` only.

**The command:**

```bash
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/zurek/SULI/dataset_v01 \
    --outdir /volatile/clas12/zurek/SULI/model_v01
```

Omit `--reweight-map` for v1. Default hyperparameters: `--n-estimators 200`,
`--learning-rate 0.05`, `--max-depth 6`, `--calibration-frac 0.2`, `--seed 42`. The
only optional flag you will realistically use is `--reweight-map PATH` (points to a
`.npz` file containing `p_edges`, `theta_edges`, and `weights` — a Week 5 deliverable;
leave it out now). Pass `--overwrite` if you want to re-run and replace existing outputs.

**What the script writes.** All outputs land in `--outdir`:

- `model.joblib` — the calibrated model object. This is the only file `evaluate.py`
  needs; it wraps both the BDT and the Platt calibrator.
- `training_summary.csv` — one row with AUC and Brier score on train and val, before
  and after calibration.
- `reliability_diagram.png` — two panels: raw BDT score vs. actual kaon fraction, then
  calibrated score vs. actual kaon fraction.
- `roc_val.png` — ROC curve on the validation set.
- `feature_importance.png` and `feature_importance.csv` — top 15 features by gain.

**What to look at after.**

First, `training_summary.csv`: val AUC should be appreciably above 0.5 (random
guessing = 0.5). A good BDT on this task typically lands above 0.85. If val AUC is
close to train AUC, the model is not badly overfitting. If val AUC is much lower than
train AUC, something is wrong — either the model is too complex for the data you gave
it, or there is a data quality issue.

Second, `reliability_diagram.png`. Before calibration the curve will probably deviate
somewhat from the diagonal. After calibration it should be much closer. If the
post-calibration curve is still badly off the diagonal, ping Maria — that is unusual.

Third, `feature_importance.png`. You should expect beta and chi2pid to appear near the
top. If a feature that has no physical reason to separate kaons from pions appears at
the top (e.g., sector number, run number, phi), that is a red flag — it probably means
the feature is picking up a confounding variable or there is a data quality issue. Share
this plot with Maria.

**Common failure: class imbalance printing a warning.** LightGBM will print a warning if
one class is much rarer than the other. This is expected if the training set has many
more pions than kaons. It does not mean the training failed. The calibration step
partially compensates for class imbalance. If the imbalance is very extreme (say, 20:1)
and the feature importance plot shows the model learned nothing, discuss adding
`class_weight='balanced'` with Maria.

---

### 5.3 `evaluate.py` — per-bin comparison against chi2pid

**What it does.** Loads your `model.joblib` and `test.parquet`. For each `(p, θ)` bin,
it sweeps the BDT score threshold from 0 to 1 and computes kaon efficiency and pion
contamination at each threshold. It also computes the same numbers for the chi2pid
baseline (using the `passes_kplus_chi2pid_cut` function from `scripts/baseline_chi2pid.py`
— the same cut the experiment uses in production). It produces two headline plots.

**The command:**

```bash
python scripts/training/evaluate.py \
    --model /volatile/clas12/zurek/SULI/model_v01/model.joblib \
    --dataset-dir /volatile/clas12/zurek/SULI/dataset_v01 \
    --outdir /volatile/clas12/zurek/SULI/eval_v01
```

The full set of optional flags: `--p-edges E1 E2 E3 ...` and `--theta-edges E1 E2 ...`
to override the default bin boundaries (defaults are set to reasonable values for the
`ep → e′ p K+ X` analysis — leave them alone for Week 4). `--threshold-grid LOW HIGH N`
controls the sweep (default is 0 to 1 in 200 steps). Pass `--overwrite` to replace
existing outputs.

**What to look at after.**

`contam_vs_ptheta_baseline_vs_bdt.png` — this is the Week 4 headline plot. It shows a
2D map of `(p, θ)` bins, with color encoding the pion contamination, as two panels
side-by-side: baseline (chi2pid) and BDT at matched kaon efficiency. If the BDT helps,
the right panel should be systematically lighter (lower contamination) than the left.
Bins with too few test events (fewer than 50 tracks) are shown in gray.

`comparison_summary.csv` — a table with one row per `(p, θ)` bin. The columns you care
about most:

- `eff_K_baseline` — kaon efficiency at the standard chi2pid cut.
- `C_pi_baseline` — pion contamination at that same cut.
- `eff_K_bdt` — kaon efficiency at the BDT threshold that matches `eff_K_baseline`.
- `C_pi_bdt_at_matched_eff` — your headline result: pion contamination at that matched
  efficiency. If the BDT is working, this should be smaller than `C_pi_baseline` in
  most bins.

Efficiency is defined as `N(selected ∧ true K+) / N(true K+)`; contamination is
`N(selected ∧ true π+) / N(selected, label not NaN)`.

`cp_to_K_map.png` — the proton contamination map. This shows `C^{p→K}`, the fraction of
tracks in each bin that are true protons but were selected as kaons. This is your input
for the Phase 4 decision (Section 6 below).

**Common failure: empty bins producing NaN.** If your test set is small (which it might
be if you used `--max-files 2` for a smoke test), some high-p bins may have zero kaon
tracks. The script should handle this gracefully and mark those bins gray. If it crashes
instead of showing gray bins, that is a bug to report.

---

## 6. Where to run things (notebook, srun, sbatch)

You have three options for where to run these scripts, and picking the wrong one wastes
time. Here is the rule:

**Use JupyterHub for inspection.** Open the dataset manifest, load one column of the
parquet and plot it, eyeball the feature importance CSV — anything where you are reading
small amounts of data and looking at plots. The JupyterHub default kernel is fine for
this. Do not run `build_dataset.py` or `train_bdt.py` from JupyterHub — those scripts
need the conda environment and meaningful RAM, and JupyterHub's kernel is not the
`suli2026_pid` conda env.

**Use `srun --pty` for your first run of each script.** An `srun` interactive shell gives
you a real compute node with a live terminal. If the script crashes, you see the full
Python traceback in real time. Run it there first. Here is the command to get that shell:

```bash
srun --pty --account=clas12 --time=2:00:00 --mem=16G --cpus-per-task=8 bash
```

Once the shell opens, activate your environment and run:

```bash
source /etc/profile.d/modules.sh
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module load clas12
conda activate suli2026_pid
cd ~/CLAS/SULI/suli2026_pid
```

Then run whichever script you are testing. The binary K/π BDT on a momentum-capped
dataset (roughly 1M rows × 15 features) trains in a few minutes on 8 CPU cores. It will
finish well within the 2-hour time limit.

**Use `sbatch` via `submit_training_bdt.sh` for the canonical run.** Once you have run
`train_bdt.py` interactively and verified the outputs look right, do one final clean run
as a batch job. This is the run that produces the numbers you put in your writeup — it
is logged, reproducible, and not tied to your SSH session. Submit it with:

```bash
./slurm/submit_training_bdt.sh \
    --dataset-dir /volatile/clas12/zurek/SULI/dataset_v01 \
    --model-dir /volatile/clas12/zurek/SULI/model_v01_canonical
```

This returns a job ID. Watch the queue with `squeue -u $USER`. While it runs, you can
tail the log:

```bash
tail -f /farm_out/zurek/suli/training_bdt_<jobid>.out
```

If you need to cancel: `scancel <jobid>`. When the job finishes, check the exit status:

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

`State=COMPLETED` and `ExitCode=0:0` means it finished cleanly.

**Can you train without sbatch?** Yes, for Week 4. The binary K/π model on a few-GeV
dataset is small enough that `srun --pty` is entirely sufficient. Use `sbatch` for the
final canonical run only — once the pipeline is working end-to-end and you want the
logged, reproducible version.

---

### If your sbatch job dies

The first thing to do is check what the scheduler recorded:

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Read the `State` column:

- **`COMPLETED` / `ExitCode=0:0`** — finished cleanly. If the outputs look wrong,
  the bug is in the Python code, not the job itself.
- **`OUT_OF_MEMORY`** — the job exceeded `--mem`. Resubmit with a larger allocation,
  e.g., `--mem 32G` instead of `--mem 16G`. Check `MaxRSS` in the `sacct` output to
  see how much RAM the job actually used.
- **`TIMEOUT`** — the job hit its `--time` wall. Resubmit with a longer limit. The
  binary K/π training on a few-GeV dataset should finish in under 30 minutes on 8
  cores; if it is timing out at 2 hours, something is probably stuck in I/O (check
  whether `/volatile` is slow) or the dataset is much larger than expected.
- **`FAILED` / non-zero `ExitCode`** — Python crashed. The traceback is in the `.err`
  log:
  ```bash
  cat /farm_out/zurek/suli/training_bdt_<jobid>.err
  ```
  The traceback is near the end. Reproduce the failure interactively in an `srun --pty`
  shell (see above) so you can iterate without waiting for the queue.

To cancel a running job: `scancel <jobid>`. To cancel all your jobs: `scancel -u $USER`.

---

## 7. The Phase 4 decision: do protons enter v2?

After you have run `evaluate.py`, look at `cp_to_K_map.png`. This shows the fraction of
BDT-selected tracks in each `(p, θ)` bin that are actually protons. You need to make a
judgment call: is the proton contamination large enough to matter?

The prior analysis found `C^{p→K} < 1%` across the board in the dihadron analysis and treated it as
negligible. Your channel is different (exclusive `ep → e′ p K+ X`), your MC is
different, and your momentum range may differ too — so check your own map rather than
assuming you get the same answer.

A practical threshold: if `C^{p→K}` is above roughly 2–3% in more than a few bins,
particularly in high-p bins where the physics analysis is sensitive, you should add
protons to v2 as a third class. If it is everywhere small, binary K/π is sufficient and
you document that protons are negligible.

Write your conclusion in `notes/cooper_week4_decisions.md` under a "Protons in v2"
heading:

- State the observed range of `C^{p→K}` across bins.
- State whether you are recommending 3-class (K vs π vs proton) or a second binary
  (K vs proton) for v2.
- Point to `cp_to_K_map.png` as the supporting plot.

This decision directly affects Week 5 scoping. Ping Maria once you have a draft so she
can review before you commit to either direction.

---

## 8. How your approach compares to prior work

The prior analysis on this channel (summarized in `~/CLAS/SULI/notes/connor_method_summary.md`)
solved a closely related but distinct problem: K+ identification for a dihadron SIDIS
analysis (`ep → e′ h₁ h₂ X`) rather than your exclusive `ep → e′ p K+ X` channel.
Understanding where you match and where you diverge will help you calibrate your results.

**Where you match the prior method:**

- Global classifier, per-bin evaluation. You train one model across the full `(p, θ)`
  range and then evaluate it bin-by-bin. The prior method did the same — one RF per
  (charge × torus) configuration, evaluated in 16 `(p, θ)` bins.
- Per-track features only. No event-level inputs (no Q², no missing mass, no partner
  hadron kinematics) go into the classifier. Mixing per-track PID features with
  event-level kinematics would couple the PID score to the physics observable you are
  trying to measure.
- `(p, θ)` reweighting concept. The `sample_weight` approach in Week 5 replicates
  Eq. 16 of that prior note (a bin-by-bin density ratio of MC to data, 15×15 grid).

**Where you diverge from the prior method:**

- Algorithm. The prior method used a Random Forest with sklearn defaults (100 trees, fully grown,
  no tuning). You are using LightGBM with explicit hyperparameter choices. BDTs are
  generally faster and often more accurate than untuned random forests at this scale.
- Calibration. The prior analysis applied no probability calibration to the `P_RF` score — it was used
  as a hard cut without treating it as a probability. You are applying Platt scaling
  and producing a reliability diagram. This is one of the explicit advances in your
  work.
- Momentum range. The prior method used `1.0 < p < 3.0 GeV` as its training range. Your `--p-max`
  choice (Section 4) may differ depending on your β-vs-p plots.
- Test set. The prior analysis did not have a strictly held-out test set — the cut optimization used
  the validation MC, so the final performance numbers came from data already seen during tuning.
  You have a proper three-way split with a test set that is not touched until final
  evaluation.
- No calibration plot in the prior work. That analysis note has no reliability diagram. Yours
  will have one.

---

## 9. Common mistakes to avoid

**Evaluating on data you trained on.** If you accidentally load `train.parquet` in
`evaluate.py` instead of `test.parquet`, your numbers will look excellent and be
meaningless — the model has memorized the training labels. `evaluate.py` takes
`--dataset-dir` and loads `test.parquet` from there; double-check the path is your
output directory, not your raw MC directory.

**Picking the momentum cutoff after you have seen test-set results.** If you run
`evaluate.py`, see that the contamination map looks bad at p > 2.8 GeV, and then decide
to set `--p-max 2.8`, you have used the test set to make a training decision. That is
look-ahead bias. You will get unrealistically good numbers in the bins you effectively
removed from evaluation. Pick `--p-max` from your β-vs-p and contamination plots (Weeks
1–2 results) before you ever run `evaluate.py`.

**Tuning hyperparameters in Week 4.** The defaults (`n_estimators=200`,
`learning_rate=0.05`, `max_depth=6`) are fine. If your val AUC looks modest and you are
tempted to try 500 trees or a deeper model, resist. Every time you try a new set of
hyperparameters and look at the val AUC, you are using the validation set to make a
decision — and if you try 20 combinations, you have effectively overfit to the validation
set. Hyperparameter search belongs in Week 7, on the validation set, with the test set
locked away.

**Adding protons before you have a working binary model.** Build the simplest version
that can possibly work, confirm it works, and then extend. A broken three-class model is
harder to debug than a broken binary model, because there are more things that can go
wrong.

**Fitting the calibrator on val or test.** The calibration set must be carved from the
training data. `train_bdt.py` handles this (`--calibration-frac 0.2` carves 20% of
train for calibration). The val and test sets are never seen by the calibrator. If you
fit calibration on val, your calibration metrics will be artificially good and the
reliability diagram will not reflect real-world performance.

**Trusting raw BDT scores as probabilities.** Before calibration, a score of 0.8 is not
"80% kaon." Use the calibrated model object (`model.joblib`) in any downstream work.
It wraps both the BDT and the calibrator.

**Running `train_bdt.py` in JupyterHub and leaving it overnight.** JupyterHub kernels
can be killed if idle or if the hub restarts. Any job that takes more than 10–15 minutes
belongs in an `srun --pty` shell or as an `sbatch` job, where it is managed by the
scheduler and has a proper log file.

**Committing after training, not before.** The split files must be committed before you
train, not after. If you train first and then commit, you cannot be sure the committed
files match what was actually used — a notebook cell might have regenerated them.
Commit, push, then train.

---

## 10. When you are stuck

**Something crashed in a batch job.** Check the log:

```bash
cat /farm_out/zurek/suli/training_bdt_<jobid>.out
cat /farm_out/zurek/suli/training_bdt_<jobid>.err
```

The Python traceback is near the end of the `.err` file. Once you have the traceback,
reproduce the failure interactively in an `srun --pty` shell so you can iterate quickly.

**An API error in a training script.** Run the script with `--help` to see all accepted
arguments and their types. Most argument errors are a mismatch between what you passed
and what the script expects — e.g., a string where it wants a float, or a path that
does not exist yet.

**A scikit-learn or LightGBM API question.** The authoritative references:

- LightGBM Python API:
  `https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html`
- sklearn CalibratedClassifierCV:
  `https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html`
- sklearn train_test_split (if you need to split manually):
  `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html`
- sklearn calibration user guide (with reliability diagram example):
  `https://scikit-learn.org/stable/modules/calibration.html`

**A physics decision is non-obvious.** Ping Maria. This covers: the momentum cutoff
choice (if your β-vs-p plots are ambiguous), whether to add protons to v2 (if the
contamination map is borderline), and whether to apply reweighting in v1 (if the audit
showed large drift in `(p, θ)` distributions). Maria should review the proton decision
output before you commit to a direction.

**Slurm or conda is behaving unexpectedly.** Run the preflight check:

```bash
bash slurm/check_farm_access.sh
```

It runs five checks and prints a `→ Fix:` hint for each failure. If all checks pass but
the job still fails, the issue is almost certainly in the Python code itself, not in the
farm setup — reproduce it in an `srun` shell.

---

*Document written: June 2026. Written for Cooper, SULI student, summer 2026.*
*An implementer-facing spec with the full internal design lives at `~/CLAS/SULI/notes/week4_training_examples_plan.md` if you ever want to read it — but you do not need to.*
