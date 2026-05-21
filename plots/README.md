# Plots

All evaluation visualisations produced by the model notebooks. Each subfolder corresponds to one notebook and is populated when that notebook is run on Kaggle.

## Contents per method

| Subfolder | Notebook |
|-----------|---------|
| `eda/` | Notebook 01 — EDA |
| `preprocessing/` | Notebook 02 — Preprocessing |
| `xgboost/` | Notebook 03 — XGBoost |
| `catboost/` | Notebook 04 — CatBoost |
| `tabnet/` | Notebook 05 — TabNet |
| `residual-mlp/` | Notebook 06 — Residual MLP |

## Plot types (model notebooks)

Each model notebook saves the following plots:

**Binary model (Model 1)**
- `bin_learning_curve.png` — train loss + val AUC per epoch/iteration
- `bin_confusion_matrix.png` — raw counts + normalised heatmap side by side
- `bin_roc_curve.png` — ROC curve with AUC score
- `bin_pr_curve.png` — Precision-Recall curve with AP score
- `bin_feature_importance.png` — top 20 features, engineered highlighted in red

**Full multiclass (Model 2)**
- `multi_learning_curve.png`
- `multi_confusion_matrix.png`
- `multi_f1_per_class.png` — horizontal bar chart sorted by F1
- `multi_pr_heatmap.png` — Precision/Recall heatmap across 21 classes
- `multi_feature_importance.png`

**Attack-only multiclass (Model 3)**
- `att_learning_curve.png`
- `att_confusion_matrix.png`
- `att_f1_per_class.png`
- `att_pr_heatmap.png`
- `att_feature_importance.png`

**Model-specific**
- `tabnet_attention_mean.png` — mean CLS/step attention bar chart
- `tabnet_attention_per_class.png` — per-class attention heatmap (TabNet)
- `resmlp_attention_mean.png` — mean gradient importance bar chart
- `resmlp_attention_per_class.png` — per-class gradient importance heatmap
- `papers_comparison.png` — your models vs published work bar chart

## Downloading from Kaggle

At the end of each notebook, all plots are zipped and available as a notebook output:

```python
shutil.make_archive('xgboost_plots', 'zip', plots_dir)
```

Download the zip from the Kaggle notebook output panel and extract into the corresponding subfolder here.
