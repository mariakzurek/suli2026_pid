# MLP training run provenance
Generated: 2026-07-14T20:33:37.736681Z

## Dataset
- Source: `/work/clas12/CooperBe/MLStuff/dataset_v03`
- Features (8): beta, ftof_energy_1B, ftof_time_1B, ftof_path_1B, chi2pid, ecin_path, ecin_energy, ecin_time

## Hyperparameters
- `hidden_layer_sizes`: (64, 64)
- `activation`: relu
- `solver`: adam
- `alpha`: 0.0001
- `batch_size`: 512
- `learning_rate_init`: 0.001
- `max_iter`: 200
- `early_stopping`: True
- `validation_fraction`: 0.1
- `n_iter_no_change`: 10
- `random_state`: 42
- `verbose`: True
- calibration_frac: 0.2

## Metrics (validation set)
- auc_train_uncal: 0.90447
- auc_val_uncal: 0.90443
- auc_train_cal: 0.90447
- auc_val_cal: 0.90443
- brier_val_uncal: 0.12028
- brier_val_cal: 0.12179
- logloss_val_uncal: 0.37634
- logloss_val_cal: 0.38607
- n_train: 21302845
- n_fit: 17042276
- n_cal: 4260569
- n_val: 4544388
- k_frac_train: 0.62283
- k_frac_val: 0.62280

## Outputs
- `model.joblib` — wrapper dict {"model": calibrated MLPClassifier, "features": feature list}
- `training_summary.csv` — AUC, Brier score, and log-loss before/after calibration
- `reliability_diagram.png` — reliability diagram on the validation set
- `roc_val.png` — ROC curve on the validation set
