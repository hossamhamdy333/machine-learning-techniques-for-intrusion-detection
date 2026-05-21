# Notebook 04 — CatBoost

**File:** `catboost.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/catboost)
**Plots:** [`/plots/catboost/`](../../plots/catboost/)

---

## Overview

CatBoost with GPU acceleration. Uses symmetric trees (oblivious decision trees) and ordered boosting, which reduce overfitting on imbalanced data compared to standard gradient boosting. Training uses a `Pool` object with explicit sample weights.

Early stopping is handled natively via `od_type='Iter'` and `od_wait` on a validation pool — no manual loop required.

---

## Hyperparameters

### Model 1 — Binary

```python
{
    'iterations'          : 1500,
    'depth'               : 6,
    'learning_rate'       : 0.05,
    'l2_leaf_reg'         : 5.0,
    'random_strength'     : 0.1,
    'bagging_temperature' : 0.5,
    'border_count'        : 128,
    'task_type'           : 'GPU',
    'loss_function'       : 'Logloss',
    'eval_metric'         : 'AUC',
    'od_type'             : 'Iter',
    'od_wait'             : 50,
}
```

### Model 2 — Full Multiclass (21 classes)

```python
{
    'iterations'          : 1500,
    'depth'               : 9,
    'learning_rate'       : 0.03,
    'l2_leaf_reg'         : 15.0,
    'random_strength'     : 2.0,
    'bagging_temperature' : 1.0,
    'border_count'        : 128,
    'task_type'           : 'GPU',
    'loss_function'       : 'MultiClass',
    'eval_metric'         : 'MultiClass',
    'od_type'             : 'Iter',
    'od_wait'             : 50,
}
```

### Model 3 — Attack-Only Multiclass (20 classes)

```python
{
    'iterations'          : 1200,
    'depth'               : 7,
    'learning_rate'       : 0.04,
    'l2_leaf_reg'         : 10.0,
    'random_strength'     : 1.5,
    'bagging_temperature' : 0.8,
    'border_count'        : 128,
    'task_type'           : 'GPU',
    'loss_function'       : 'MultiClass',
    'eval_metric'         : 'MultiClass',
    'od_type'             : 'Iter',
    'od_wait'             : 50,
}
```

---

## Notes on CatBoost vs XGBoost for this dataset

- CatBoost's ordered boosting prevents target leakage during training, which matters for rare classes
- `l2_leaf_reg` replaces XGBoost's `min_child_weight` for regularisation
- `bagging_temperature` controls Bayesian bootstrap intensity (higher = more random)
- Feature importance from CatBoost uses `get_feature_importance()` which returns PredictionValuesChange scores

---

## Evaluation plots

Each model produces:
- Learning curve with best iteration marker
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Feature importance bar chart (top 20, engineered features highlighted)
- Comparison bar chart vs published papers on NF-UQ-NIDS-v2
