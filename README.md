# SULI 2026 — ML PID for CLAS12

ML-based particle identification for the `ep → e' p K+ X` SIDIS channel using CLAS12 RGA pass-2 data.

**Student:** Cooper Bell (SULI 2026)
**PI:** Maria Zurek (ANL)


## Introduction
In semi inclusive deep inelastic scattering experiments, the study of hadrons give insight into the structure of the proton. Of particular interest is kaons, which give access to the strange quark. At high momentum ranges, the signitures of Kaons and Pions become similar, resulting in lower purity of K+ samples. In order to improve the purity, a machine learning classifier (gradient boosted decsion tree (BDT)) was developed, with the goal of having lower contamination, whilst maintaining efficiency. The model was trained using Monte Carlo hipo files from Clas-Dis. The model was tested against a baseline method using cuts on chi2pid. Contamination results were validated using two methods: contamination using a Ring Image Cherenkov Detector (RICH) as a truth, and using an exclusive epi(N) reaction to find pione->kaon misidentification, which was used to estimate contamination. This repo also contains a small MLP neural network model which was found to perform similar to the BDT model.

## What's in this repo

- `notebooks/` — Jupyter notebooks for data exploration, model training, evaluation
- `scripts/` — standalone Python scripts (training pipeline, evaluation, plot generation)
- `figures/` — generated plots and figures (not source files — those live in the notebooks/scripts)
- `slurm/` — ifarm batch submission scriptss
- `notes/` — project notes, weekly summaries, intermediate writeups

## Companion repo

The CLAS12 framework and data-production scripts (groovy + Java + C++ converter) live in:
https://github.com/mariakzurek/clas12_analysis_software (branch `suli_kaon_pid`, off `rich_studies`)

That repo runs on ifarm to produce the training/analysis ntuples that this repo consumes.

## Quick start

**For interactive notebook work** (recommended for Week 1):
1. Log in to JLab JupyterHub at `https://jupyterhub.jlab.org` with your CUE credentials
2. Navigate to `/work/clas12/<username>/SULI/suli2026_pid/notebooks/`
3. Open or create a notebook with the Python 3 kernel
4. The default kernel has numpy, pandas, matplotlib, scikit-learn, uproot pre-installed
5. For `lightgbm`: run `!pip install --user lightgbm` once in a notebook cellz

## Conda Envroment Set Up
The following section contains instructions for setting up the conda enviroment. Type the Following commands
- mkdir -p /work/clas12/<username>/conda/pkgs
  mkdir -p /work/clas12/<username>/conda/envs
- conda config --add pkgs_dirs /work/clas12/<username>/conda/pkgs
  conda config --add envs_dirs /work/clas12/<username>/conda/envs
- cd /work/clas12/<username>/SULI/suli2026_pid
- conda env create -f environment.yml
- conda activate suli2026_pid
Confirm with "which conda",
Anytime the project is opened and a python script needs run, you must type conda activate suli2026_pid first.

## The Pipeline
This section will outline the general pipeline, and contain steps on how to produce a full model with commands

Pipeline Summary
- (Prequisite) Create or Access MC hipo files
- Create ntuple root files from hipos
- build dataset parquet files: train, val, test
- train BDT model from feature list of varaibles (audit MC-Experimental agreement first)
- evaluate
- optimize FOM

# Pipeline Instructions

Creating ntuples

1. Use the Clas12Analyizer to process MC hipos into usable root files
run this command in outer-most directory within the Clas12 Analizyer:
  './processing_scripts/processing.csh \
    processing_scripts/processing_mc_pid_training.groovy \
    [Hipo file Directory Path] \
    [Output Path] \
    [number of hipo files] 10.6041 11'
  
  10.6041 is the beam energy, 11 is the runnum for MC. 

2. Use the notebook script fileMixer.ipynb (replacing the string in mc_dir with your path), to generate the file splitting for the parquette building (output in slurm/)

The following Steps can be found in Scripts/training/

3. Run the following command to build the parquets

   'python scripts/training/build_dataset.py --mc-dir [Mc hipo path] --split-dir slurm --outdir [Output Path]'

   This makes the train, val, and test parquettes

5. Run the following locally, or alternativy submitt training as a Slurm Job

   Locally:

    'python train_bdt.py  --dataset-dir [output directory from step 3] --model-dir [Model output location] --features-file [feature file txt with variable names]'

   Slurm Job (from outer directory in repo):
    './slurm/submit_training_bdt.sh --dataset-dir [output directory from step 3] --model-dir [Model output location] --features-file [feature file txt with variable names]'

ROC curve and feature importance plots are generated alongside the model file.

5. Run The following to Evaluate the model
   'python scripts/training/evaluate.py --model [path of model.joblib] --dataset-dir [path to step 3 output] --outdir [output location path]'

Generates contaminations, efficiencies using initial model, has fixed efficiency, chi2pid (baseline method) matched efficiency plots.

# Optimizing FOM for the BDT
To optmimize the Feature of Merit (FOM) of the BDT model, see under scripts common_functions.py

In a script or notebook, load in your dataset. Then import common_functions.py 
Run the following functions
- load_model_and_data(model_path, df) #This returns the model, and df associated, must use val parquet to be loaded in
  
- get_feature_names(model_path) #returns list of features the model was trained on



- optimizeFOM(model_df, theta_bin_edges, p_bin_edges, outputCSV_path, deviation) # This function will create a CSV with the score thresholds that maximize FOM. Deviation is a bias, which will pick the highest threshold within a deviation range from the max FOM. common_functions also contains utilities for the creation of the bin edge lists, see comments in script. This also returns results_df which contains the same data as the CSV


# Applying Optimized BDT

There are two avenues for applying the BDT to data, the first is running this command:

- 'python scripts/apply_bdt.py --model [path to model joblib] --input-dir [path to data] --output-dir [output path to bdt scored data]  --threshold-csv [path to optimized threshold csv]'

or the corresponding slurm job

'./slurm/submit_apply_bdt.sh --model [path to model joblib] --input-dir [path to data] --output-dir [output path to bdt scored data]  --threshold-csv [path to optimized threshold csv]'


The second method is done within the code:

-First follow instructions from the Optimizing FOM for the BDT section to load the model, running the FOM optimization if not done already.

- run apply_model_to_df(model, df, feature names) to return a df with the "score" colum added.

- run apply_optimized_bdt_cut(df, threshold_df=results_df) if optimizing for the first time or apply_optimized_bdt_cut(df, CSVPath="Insert Path") if using a pre-existing optimization. This code will generate a boolean mask.
 

**For batch jobs** (Week 2+):
```bash
# Set up the conda env (one-time, see notes/cooper_day1_and_week1.md Section 4f for full instructions)
conda env create -f environment.yml
conda activate suli2026_pid
```

The conda env contains numpy, pandas, matplotlib, scikit-learn, uproot, awkward, lightgbm, xgboost — runtime libraries only. No jupyterlab (use JupyterHub for notebooks).

See `notes/cooper_day1_and_week1.md` for the full onboarding doc.

## Project plan from Summer 2026

See `notes/` directory.
