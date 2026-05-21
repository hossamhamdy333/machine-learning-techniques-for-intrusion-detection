# Notebook 05 — TabNet

**File:** `tabular-network.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/tabular-network)
**Plots:** [`/plots/tabnet/`](../../plots/tabnet/)

---

## Overview

TabNet (Arik & Pfister, 2021) is an attention-based deep learning architecture designed specifically for tabular data. Unlike standard MLPs, TabNet selects a sparse subset of features at each decision step using a sequential attention mechanism, making it inherently interpretable.

The key advantage over tree models shown in this notebook: **the attention masks reveal which features each attack type focuses on**, producing a per-class feature importance heatmap that trees cannot replicate.

Because the full 13.7M training rows exceed Kaggle's memory budget, the training set is subsampled to ~5.2M rows by capping the three largest classes (Benign, DDoS, DoS at 1.2M each; Recon, Scanning, Mirai at 400k each) while keeping all rare class samples intact.

---

## Hyperparameters (shared across all three models)

```python
TABNET_PARAMS = {
    'n_d'               : 16,    # width of decision step output
    'n_a'               : 16,    # width of attention embedding
    'n_steps'           : 3,     # number of sequential decision steps
    'gamma'             : 1.3,   # attention sparsity regularisation
    'n_shared'          : 2,     # shared layers across steps
    'lambda_sparse'     : 1e-4,  # sparsity loss weight
    'optimizer_fn'      : Adam,
    'optimizer_params'  : {'lr': 2e-3},
    'scheduler_fn'      : StepLR,
    'scheduler_params'  : {'step_size': 10, 'gamma': 0.9},
    'mask_type'         : 'sparsemax',
    'device_name'       : 'cuda',
    'seed'              : 42,
    'verbose'           : 1,
}
```

**Fit parameters:**

| Param | Binary | Multiclass | Attack-Only |
|-------|--------|-----------|-------------|
| max_epochs | 100 | 100 | 100 |
| patience | 15 | 15 | 15 |
| batch_size | 16384 | 16384 | 16384 |
| virtual_batch_size | 512 | 512 | 512 |

---

## Memory management

TabNet's `predict()` is called in chunks of 131,072 rows to avoid OOM on the 4.5M-row test set. Training data is deleted and CUDA cache cleared between models.

---

## Unique visualisation: attention mask analysis

After all three models are trained, `tabnet_multi` is reloaded and `explain()` is called on a 2,000-sample subset of the attack-only test set. This produces:

1. **Mean attention bar chart** — average attention weight per feature across all samples
2. **Per-class attention heatmap** — mean attention per feature grouped by attack class, showing that TabNet learns distinct feature subsets for different attack types

---

## Evaluation plots

Each model produces:
- Learning curve (train loss + val AUC/accuracy)
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Feature importance (TabNet `feature_importances_`, top 20)
- Attention mask visualisation (mean + per-class heatmap)
- Comparison bar chart vs published papers
