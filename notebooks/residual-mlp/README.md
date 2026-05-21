# Notebook 06 — Residual MLP

**File:** `residual-mlp.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/residual-mlp)
**Plots:** [`/plots/residual-mlp/`](../../plots/residual-mlp/)

---

## Overview

A deep MLP with residual connections, BatchNorm, and GELU activations — trained from scratch using PyTorch. Designed as a compute-efficient alternative to the FT-Transformer that still achieves competitive accuracy on large tabular datasets.

The architecture follows a simple principle: stack residual blocks wide rather than deep, let BatchNorm handle internal covariate shift, and use AMP fp16 to double throughput on the T4 GPU.

**All three models finish training in under 40 minutes on Kaggle free T4.**

---

## Architecture

```
Input (42 features)
    │
    └── BatchNorm1d(42)
         │
         └── ResBlock × 4
              ├── Linear(in → 512) → BN → GELU → Dropout(0.15)
              ├── Linear(512 → 512) → BN → GELU → Dropout(0.15)
              └── skip: Linear(in → 512) [only on first block]
         │
         └── Linear(512 → n_classes)
```

Total parameters: ~1.5M (binary) / ~1.6M (21-class)

---

## Hyperparameters

```python
RESMLP_PARAMS = {
    'hidden_dim'   : 512,
    'n_blocks'     : 4,
    'dropout'      : 0.15,
    'lr'           : 3e-4,
    'weight_decay' : 1e-4,
}

FIT_PARAMS = {
    'max_epochs'  : 20,
    'patience'    : 5,
    'batch_size'  : 8192,
    'num_workers' : 2,
}
```

**Training details:**
- Optimiser: AdamW
- LR schedule: Linear warmup (3 epochs) → cosine annealing to 0
- Gradient clipping: max norm 1.0
- Mixed precision: `torch.cuda.amp.GradScaler` (fp16)
- Early stopping: patience on primary validation metric

---

## Feature importance

TabNet has built-in attention masks. This model uses **gradient-based feature importance** instead: backpropagates through the model on a stored training sample and takes the mean absolute gradient per input dimension. The resulting importance scores are plotted identically to the other notebooks for direct comparison.

The `explain(X)` method returns per-sample gradient magnitudes with the same shape as TabNet's `explain()` output, enabling the same per-class heatmap visualisation.

---

## Why Residual MLP over a plain MLP?

- Skip connections allow gradients to flow cleanly through depth without vanishing
- BatchNorm before each activation stabilises training on the heavily skewed, multi-scale NetFlow features
- With 5M training rows, the model learns quickly — depth adds more than width here
- The residual structure makes the model robust to the extreme class imbalance via the log-smoothed sample weights, because rare class gradients are not washed out

---

## Evaluation plots

Each model produces:
- Learning curve (train loss + val AUC/accuracy/balanced accuracy)
- Confusion matrix (raw count + normalised)
- ROC curve with AUC *(binary model)*
- Precision-Recall curve with AP score *(binary model)*
- Per-class F1 bar chart *(multiclass models)*
- Per-class Precision/Recall heatmap *(multiclass models)*
- Gradient feature importance bar chart (top 20, engineered features highlighted)
- Per-class gradient importance heatmap (same interface as TabNet attention)
- Comparison bar chart vs published papers
