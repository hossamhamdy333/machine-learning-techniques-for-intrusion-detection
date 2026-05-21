# Notebook 03 — XGBoost

**File:** `xgboost.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/xgboost)
**Plots:** [`/plots/xgboost/`](../../plots/xgboost/)

---

## Overview

XGBoost with GPU acceleration (`tree_method='hist', device='cuda'`). Serves as the primary tree-based baseline. Fast to train, strong out-of-the-box, and highly interpretable via feature gain importance.

Class imbalance is handled with log-smoothed sample weights passed to the `sample_weight` parameter.

---

## Hyperparameters

### Model 1 — Binary

```python
{
    'max_depth'        : 6,
    'learning_rate'    : 0.05,
    'n_estimators'     : 1000,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'min_child_weight' : 5,
    'tree_method'      : 'hist',
    'device'           : 'cuda',
    'eval_metric'      : ['logloss', 'auc'],
    'early_stopping_rounds': 50,
}
```

### Model 2 — Full Multiclass (21 classes)

```python
{
    'max_depth'        : 8,
    'learning_rate'    : 0.03,
    'n_estimators'     : 1500,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.6,
    'min_child_weight' : 10,
    'tree_method'      : 'hist',
    'device'           : 'cuda',
    'eval_metric'      : ['mlogloss'],
    'early_stopping_rounds': 50,
}
```

### Model 3 — Attack-Only Multiclass (20 classes)

```python
{
    'max_depth'        : 8,
    'learning_rate'    : 0.03,
    'n_estimators'     : 1500,
    'subsample'        : 0.75,
    'colsample_bytree' : 0.7,
    'min_child_weight' : 10,
    'tree_method'      : 'hist',
    'device'           : 'cuda',
    'eval_metric'      : ['mlogloss'],
    'early_stopping_rounds': 50,
}
```

---

## Evaluation plots

Each model produces:
- Learning curve (train loss + val loss)
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Feature importance bar chart (top 20, engineered features highlighted in red)
- Comparison bar chart vs published papers on NF-UQ-NIDS-v2
