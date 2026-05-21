# Machine Learning Techniques for Intrusion Detection

A systematic comparison of six machine learning methods on the [NF-UQ-NIDS-v2](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) network intrusion detection dataset — one of the largest publicly available NetFlow-based IDS benchmarks, containing roughly 76 million records across 21 traffic classes.

Each method trains three models that can be stacked into a two-stage detection pipeline:

| Model | Task | Classes |
|-------|------|---------|
| Model 1 | Binary | Benign vs Attack |
| Model 2 | Full Multiclass | 21 classes (Benign + 20 attack types) |
| Model 3 | Attack-Only Multiclass | 20 attack types (used as second stage after Model 1) |

---

## Repository Structure

```
├── notebooks/
│   ├── eda/                    # 01 — Exploratory Data Analysis
│   ├── preprocessing/          # 02 — Feature engineering & preprocessing
│   ├── xgboost/                # 03 — XGBoost models
│   ├── catboost/               # 04 — CatBoost models
│   ├── tabnet/                 # 05 — TabNet (attention-based tabular network)
│   ├── residual-mlp/           # 06 — Residual MLP (deep learning)
│
├── plots/
│   ├── eda/                    # EDA visualisations
│   ├── preprocessing/          # Preprocessing visualisations
│   ├── xgboost/                # XGBoost evaluation plots
│   ├── catboost/               # CatBoost evaluation plots
│   ├── tabnet/                 # TabNet evaluation plots
│   ├── residual-mlp/           # Residual MLP evaluation plots
│
└── assets/                     # Images used in this README
```

---

## Dataset

**NF-UQ-NIDS-v2** — Network Flow University of Queensland NIDS Dataset v2

- **Source:** [Kaggle — aryashah2k](https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset)
- **Size:** ~76 million NetFlow records, 42 features (after dropping non-predictive IPs)
- **Format:** CSV, features are all numerical (float32)
- **Classes:** 21 total — 1 Benign + 20 named attack types

The dataset is not included in this repository. Download it from Kaggle and place it at the path expected by the notebooks (`/kaggle/input/datasets/...`).

### Class Distribution (training split, after stratified subsampling)

| Class | Count | Class | Count |
|-------|-------|-------|-------|
| Benign | 1,200,000 | DDoS | 1,200,000 |
| DoS | 1,200,000 | Reconnaissance | 400,000 |
| Scanning | 400,000 | Mirai | 400,000 |
| Password | 214,708 | Infiltration | 133,775 |
| MITM | 24,497 | Ransomware | 22,939 |
| XSS | 11,759 | Vulnerability Scanner | 5,951 |
| Backdoor | 4,151 | Bot | 3,496 |
| Injection | 1,933 | DDoS-SlowRate | 1,486 |
| Analysis | 621 | Fuzzers | 422 |
| Worms | 229 | Shellcode | 227 |
| DoS-SlowRate | 133 | — | — |

Severe class imbalance (3 dominant classes, 8 rare classes under 5k samples) makes this a challenging real-world benchmark.

---

## Notebooks

All notebooks run on **Kaggle free tier (T4 GPU, 15 GB VRAM, 30 GB RAM)**. Each is self-contained and loads artefacts produced by the preprocessing notebook.

### 01 — Exploratory Data Analysis
`notebooks/eda/`

Chunked stratified sampling (33% of 76M rows), memory-efficient loading with dtype casting, full statistical profiling, skewness/kurtosis analysis, correlation matrix, multicollinearity detection, mutual information feature ranking, PCA separability check, and class distribution visualisations.

### 02 — Preprocessing & Feature Engineering
`notebooks/preprocessing/`

Median imputation, 12 domain-specific engineered features (flow ratios, protocol flags, TTL range), multicollinearity-driven feature pruning, log1p transforms for high-skew features, RobustScaler normalisation, stratified 60/20/20 train/val/test split, log-smoothed class weights, and mutual information feature selection. All artefacts (scaler, encoder, arrays) are saved for downstream notebooks.

**Engineered features include:**
- `bytes_per_pkt_in / out` — payload size per packet direction
- `upload_ratio` — upload vs total byte share
- `byte_asymmetry / pkt_asymmetry` — directional flow imbalance
- `is_http_https / is_ssh_telnet / is_dns` — protocol flags
- `ttl_range` — TTL spread (proxy for path diversity)
- `syn_only_flag` — SYN-without-ACK (scan indicator)
- `retransmit_pkt_ratio` — retransmission rate

### 03 — XGBoost
`notebooks/xgboost/`

GPU-accelerated gradient boosted trees. Strong baseline — fast to train, interpretable feature importance, no normalisation required (used here with pre-scaled data for a fair comparison). Handles class imbalance via log-smoothed sample weights.

| Model | Key params |
|-------|-----------|
| Binary | depth=6, lr=0.05, 1000 trees |
| Full multiclass | depth=8, lr=0.03, 1500 trees |
| Attack-only | depth=8, lr=0.03, 1500 trees |

### 04 — CatBoost
`notebooks/catboost/`

Yandex's gradient boosting with symmetric trees and built-in overfitting detection via early stopping on a validation pool. Tends to outperform XGBoost on imbalanced data due to ordered boosting.

| Model | Key params |
|-------|-----------|
| Binary | depth=6, lr=0.05, 1500 iterations |
| Full multiclass | depth=9, lr=0.03, 1500 iterations |
| Attack-only | depth=7, lr=0.04, 1200 iterations |

### 05 — TabNet
`notebooks/tabnet/`

Attention-based tabular deep learning model. Uses sequential attention to select relevant features at each decision step, producing interpretable attention masks. The unique visualisation in this notebook shows which features each attack type attends to most.

| Param | Value |
|-------|-------|
| n_d / n_a | 16 |
| n_steps | 3 |
| gamma | 1.3 |
| lambda_sparse | 1e-4 |

### 06 — Residual MLP
`notebooks/residual-mlp/`


| Param | Value |
|-------|-------|
| Hidden dim | 512 |
| Residual blocks | 4 |
| Dropout | 0.15 |
| Batch size | 8192 |
| Optimiser | AdamW, lr=3e-4 |


Feature Tokenisation Transformer — each numerical feature is embedded as a token and processed by standard Transformer encoder blocks with CLS attention pooling. The attention weights serve as feature importance scores per sample and per class.

---

## Evaluation

Every model is evaluated on the held-out test set (20% of data, ~4.5M rows) with the same metrics across all methods:

- **Accuracy** and **Macro F1** (primary)
- **Balanced Accuracy** (accounts for class imbalance)
- **Per-class Precision, Recall, F1** (classification report)
- **ROC-AUC** and **Precision-Recall curve** (binary models)
- **Normalised confusion matrix** (full heatmap)
- **Feature importance** (model-specific: gain for trees, attention masks for TabNet, gradient magnitude for MLP/Transformer)
- **Learning curves** (train loss + val metric per epoch/round)

### Comparison with Published Work on NF-UQ-NIDS-v2

The notebooks include a side-by-side bar chart comparing results against 10 published papers on the same dataset:

| Paper | Method | Year |
|-------|--------|------|
| Sarhan et al. | Extra Trees | 2022 |
| Wijethilaka et al. | Random Forest | 2025 |
| Mouatassim et al. | XGBoost | 2024 |
| Bouzaachane et al. | Random Forest | 2025 |
| Gu et al. | CNN | 2024 |
| Krupski et al. | MLP | 2024 |
| Komisarek et al. | Various | 2021 |
| Gouda et al. | Transformer | 2024 |
| Adeniyi et al. | LSTM | 2024 |
| Lanvin et al. | GNN | 2023 |

---

## How to Run

All notebooks are designed to run on **Kaggle** with the following setup:

1. Upload the NF-UQ-NIDS-v2 dataset to your Kaggle account as a dataset input.
2. Run the notebooks **in order** — each one after preprocessing loads artefacts from the preprocessing notebook output.
3. Enable GPU accelerator (T4 × 1) in notebook settings.
4. Expected runtimes on Kaggle free T4:

| Notebook | Approx. runtime |
|----------|----------------|
| EDA | ~15 min |
| Preprocessing | ~20 min |
| XGBoost | ~45 min |
| CatBoost | ~50 min |
| TabNet | ~90 min |
| Residual MLP | ~35 min |

---

## Dependencies

Each notebook installs its own dependencies via pip. Common versions used:

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib
seaborn
joblib
psutil
xgboost          (notebook 03)
catboost         (notebook 04)
pytorch-tabnet   (notebook 05)
torch==2.3.0     (notebooks 05, 06, 07)
```

---

## Author

**Hossam Hamdy**
Kaggle: [hossamhamdyfakry](https://www.kaggle.com/hossamhamdyfakry)

---

## License

This project is released under the MIT License. The NF-UQ-NIDS-v2 dataset is subject to its own terms — see the [original dataset page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) for details.
