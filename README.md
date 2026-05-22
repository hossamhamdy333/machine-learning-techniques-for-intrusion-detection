# Machine Learning Techniques for Intrusion Detection

A large-scale comparative study of modern machine learning and deep learning approaches for **network intrusion detection** using the **NF-UQ-NIDS-v2** benchmark dataset.

The project evaluates multiple architectures under the same preprocessing pipeline, train/validation/test splits, and evaluation metrics to provide a fair comparison between:

* Gradient boosted decision trees
* Attention-based tabular neural networks
* Deep residual multilayer perceptrons
* Classical statistical preprocessing pipelines

The repository is designed for reproducible experimentation on the **Kaggle free tier (T4 GPU)** while still scaling to tens of millions of network flow records.

---

## Project Goals

## This repository focuses on:


 Scalable IDS Pipelines: Building high-throughput data processing networks capable of parsing and normalizing massive NetFlow/IPFIX datasets without packet loss. 
 
 Dual-Architecture Benchmarking: Executing continuous, rigorous comparisons between traditional tree-based classifiers (`XGBoost`, `CatBoost`) and specialized tabular deep learning paradigms (`PyTorch`, `TabNet`).
 
 Imbalance Robustness: Evaluating, engineering, and stabilizing model loss gradients under the severe class imbalances (often >99.9% benign) inherent to real-world network traffic.
 
 Explainable AI (XAI): Transitioning from black-box heuristics to trusted automation by computing global and real-time local `SHAP` feature attribution vectors for security analysts.
 
 Research Verification: Benchmarking architecture performance, False Positive Rates (FPR), and Macro F1-Scores directly against published academic baselines on the **NF-UQ-NIDS-v2** multi-source dataset.
 
Real-World Deployment: Bridging the gap between Jupyter Notebook prototyping and enterprise infrastructure using distributed stream processing (`Kafka`/`Spark`), line-rate inference engines (`ONNX Runtime`), and structural `SIEM` alerting layers.

---

## Dataset

### NF-UQ-NIDS-v2

The experiments use the publicly available:

**Network Flow University of Queensland Intrusion Detection System Dataset v2 (NF-UQ-NIDS-v2)**

### Dataset Characteristics

| Property       | Value                         |
| -------------- | ----------------------------- |
| Records        | ~76 million                   |
| Features       | 42 numerical NetFlow features |
| Classes        | 21 total classes              |
| Benign classes | 1                             |
| Attack classes | 20                            |
| Format         | CSV                           |
| Feature type   | Continuous numerical          |
| Domain         | Network intrusion detection   |

### Source

* Kaggle: [https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset](https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset)
* Original dataset page: [https://staff.itee.uq.edu.au/marius/NIDS_datasets/](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)

### Challenges of the Dataset

NF-UQ-NIDS-v2 is significantly harder than many small IDS benchmarks because it contains:

* Extremely imbalanced classes
* Rare attack categories with only hundreds of samples
* Large-scale traffic distributions
* Highly skewed NetFlow statistics
* Strong feature correlations
* Non-linear decision boundaries

Several attack classes contain fewer than 1,000 samples, making macro-level evaluation metrics critical.

---

## Repository Structure

```text
├── notebooks/
│   ├── eda/
│   ├── preprocessing/
│   ├── xgboost/
│   ├── catboost/
│   ├── tabnet/
│   └── residual-mlp/
│
├── plots/
│   ├── eda/
│   ├── preprocessing/
│   ├── xgboost/
│   ├── catboost/
│   ├── tabnet/
│   └── residual-mlp/
│
├── LICENSE
└── README.md
```

---

# Pipeline Overview

Each modelling notebook trains three IDS variants:

| Model   | Objective                      | Output            |
| ------- | ------------------------------ | ----------------- |
| Model 1 | Binary detection               | Benign vs Attack  |
| Model 2 | Full multiclass classification | 21 classes        |
| Model 3 | Attack-only classification     | 20 attack classes |

This design enables deployment as a two-stage IDS pipeline:

1. Detect malicious traffic
2. Identify the attack category

---

# Notebook Details

## 01 — Exploratory Data Analysis

**Directory:** `notebooks/eda/`

### Main Objectives

* Understand class imbalance
* Analyse feature distributions
* Detect multicollinearity
* Identify skewed variables
* Evaluate feature separability
* Discover informative features

### Key Techniques

* Chunked CSV loading
* Stratified sampling
* Correlation analysis
* Mutual information ranking
* PCA visualisation
* Skewness and kurtosis analysis
* Duplicate detection
* Missing value analysis

### Important Findings

* The dataset is highly imbalanced
* Three dominant classes represent most traffic
* Multiple features are strongly skewed
* PCA shows weak linear separability
* Several highly correlated features exist
* Engineered ratio features improve separability

### Outputs

* Correlation heatmaps
* Distribution plots
* PCA projections
* Mutual information rankings
* Class imbalance charts

---

## 02 — Preprocessing & Feature Engineering

**Directory:** `notebooks/preprocessing/`

### Main Objectives

* Prepare data for all downstream models
* Create reusable preprocessing artefacts
* Reduce redundancy
* Improve feature quality
* Handle imbalance safely

### Preprocessing Pipeline

| Step                | Description                |
| ------------------- | -------------------------- |
| Missing handling    | Median imputation          |
| Scaling             | RobustScaler               |
| Encoding            | LabelEncoder               |
| Feature selection   | Mutual information         |
| Correlation pruning | Remove redundant variables |
| Transformations     | log1p for skewed variables |
| Splitting           | Stratified train/val/test  |

### Engineered Features

The notebook introduces domain-informed networking features such as:

* Packet asymmetry
* Byte asymmetry
* Upload ratio
* Protocol flags
* SYN-only detection
* Retransmission ratio
* TTL spread
* Flow duration metrics

### Saved Artefacts

The notebook exports:

* NumPy arrays
* Label encoders
* Class weights
* Scalers
* Imputers
* Feature rankings
* Attack-only splits

These artefacts are reused by every model notebook.

---

## 03 — XGBoost

**Directory:** `notebooks/xgboost/`

### Overview

XGBoost serves as the primary tree-based baseline using GPU acceleration.

### Why XGBoost?

* Extremely fast training
* Strong tabular performance
* Stable optimisation
* Good handling of heterogeneous distributions
* High interpretability via gain importance

### Training Characteristics

| Property                 | Value               |
| ------------------------ | ------------------- |
| Backend                  | GPU histogram trees |
| Early stopping           | Yes                 |
| Feature scaling required | No                  |
| Interpretability         | High                |
| Memory efficiency        | Strong              |

### Strengths Observed

* Excellent binary detection performance
* Fast convergence
* Strong robustness to noisy features
* Reliable performance on dominant attack classes

### Weaknesses Observed

* Lower recall on ultra-rare attacks
* Less expressive than attention-based networks
* Feature interactions are implicit rather than explicit

---

## 04 — CatBoost

**Directory:** `notebooks/catboost/`

### Overview

CatBoost uses ordered boosting and symmetric decision trees to reduce overfitting and improve stability on imbalanced datasets.

### Why CatBoost?

* Better handling of imbalance
* Strong regularisation
* Robust validation behaviour
* Reduced target leakage
* Stable multiclass optimisation

### Training Characteristics

| Property             | Value            |
| -------------------- | ---------------- |
| Backend              | GPU              |
| Tree type            | Symmetric trees  |
| Overfitting detector | Native           |
| Validation strategy  | Ordered boosting |
| Interpretability     | High             |

### Strengths Observed

* Strong multiclass performance
* Better rare-class stability than XGBoost
* Smooth optimisation curves
* Consistent macro-F1 behaviour

### Weaknesses Observed

* Slightly slower training than XGBoost
* Higher memory usage during training

---

## 05 — TabNet

**Directory:** `notebooks/tabnet/`

### Overview

TabNet is an attention-based architecture designed specifically for tabular data.

Instead of using all features equally, TabNet learns sparse feature selection masks at each decision step.

### Why TabNet?

* Interpretable attention masks
* Dynamic feature selection
* Better modelling of complex feature interactions
* Neural network flexibility for tabular data

### Architecture Characteristics

| Component                | Value |
| ------------------------ | ----- |
| Decision dimension       | 16    |
| Attention dimension      | 16    |
| Decision steps           | 3     |
| Sparse attention         | Yes   |
| Mixed precision training | Yes   |

### Unique Contributions

The notebook includes:

* Per-class attention heatmaps
* Mean feature attention analysis
* Sparse feature selection visualisations
* Attack-specific feature focus analysis

### Strengths Observed

* Excellent interpretability
* Better representation learning for rare attacks
* Learns attack-specific feature subsets

### Weaknesses Observed

* Longest training time
* More sensitive to hyperparameters
* Higher GPU memory consumption

---

## 06 — Residual MLP

**Directory:** `notebooks/residual-mlp/`

### Overview

A deep residual multilayer perceptron implemented in PyTorch using:

* Residual connections
* Batch normalisation
* GELU activations
* Mixed precision training
* AdamW optimisation

### Why Residual MLP?

The architecture provides a simpler and faster alternative to Transformer-style tabular networks while maintaining strong predictive performance.

### Architecture Characteristics

| Component        | Value |
| ---------------- | ----- |
| Hidden dimension | 512   |
| Residual blocks  | 4     |
| Activation       | GELU  |
| Dropout          | 0.15  |
| Optimiser        | AdamW |
| Batch size       | 8192  |

### Strengths Observed

* Fast neural-network training
* Stable optimisation
* Strong macro-F1 performance
* Better scalability than attention-heavy architectures

### Weaknesses Observed

* Less interpretable than TabNet
* Gradient importance is noisier than tree importance

---

# Model Performance Comparison

## Measured Performance Results

> The following results are taken directly from the executed notebook outputs on the NF-UQ-NIDS-v2 dataset using the shared preprocessing pipeline.

## Model 1 — Binary Classification (Benign vs Attack)

| Model        | Accuracy | F1-Score | Weighted-F1 |
| ------------ | -------- | -------- | ----------- |
| XGBoost      | 0.9909   | 0.9897   | 0.9909      |
| CatBoost     | 0.9901   | 0.9888   | 0.9901      |
| TabNet       | 0.9831   | 0.9809   | 0.9831      |
| Residual MLP | 0.9813   | 0.9786   | 0.9812      |

### Binary Classification Observations

* All models exceed 98% accuracy
* XGBoost achieves the strongest overall binary detection performance
* Tree-based models converge faster and generalise better on dominant traffic distributions
* Neural architectures remain competitive while requiring significantly longer training

---

## Model 2 — Full Multiclass Classification (21 Classes)

| Model        | Accuracy | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1 |
| ------------ | -------- | ---------------------- | ------------------- | --------------- |
| XGBoost      | 0.9823   | 0.9825                 | 0.9823              | 0.9820          |
| CatBoost     | 0.9770   | 0.9774                 | 0.9770              | 0.9762          |
| TabNet       | 0.9615   | 0.9633                 | 0.9615              | 0.9627          |
| Residual MLP | 0.9659   | 0.9675                 | 0.9659              | 0.9669          |

### Full Multiclass Observations

The large gap between Weighted-F1 and F1-Score reflects the extreme imbalance of NF-UQ-NIDS-v2.

Dominant classes:

* DDoS
* DoS
* scanning
* xss
* Reconnaissance

contain hundreds of thousands to millions of samples.

Rare classes such as:

* Worms
* Shellcode
* Analysis
* Theft

contain fewer than 200 samples and heavily reduce F1-Score.

XGBoost provides the best overall multiclass performance and strongest rare-class stability among all evaluated models.

---

## Model 3 — Attack-Only Multiclass Classification (20 Attack Classes)

| Model        | Accuracy | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1 |
| ------------ | -------- | ---------------------- | ------------------- | --------------- |
| XGBoost      | 0.9872   | 0.9873                 | 0.9872              | 0.9872          |
| CatBoost     | 0.9816   | 0.9819                 | 0.9816              | 0.9815          |
| Residual MLP | 0.9791   | 0.9793                 | 0.9791              | 0.9792          |
| TabNet       | 0.9733   | 0.9742                 | 0.9733              | 0.9739          |

### Attack-Only Classification Observations

Removing benign traffic significantly improves F1-Score across all architectures.

This indicates that:

* benign-vs-attack separation is relatively easy
* distinguishing between attack families is substantially harder
* rare attack categories dominate F1-Score behaviour

The attack-only setup provides a more realistic benchmark for advanced IDS research.

---

## Rare-Class Behaviour

The following classes remain consistently difficult across all models:

| Class     | Main Challenge                     |
| --------- | ---------------------------------- |
| Worms     | Extremely low support (21 samples) |
| Analysis  | Severe overlap with other traffic  |
| Theft     | Very small support                 |
| mitm      | Weak statistical separability      |
| Shellcode | Small sample size                  |

Examples observed directly from notebook outputs:

* TabNet failed to correctly identify Worms samples in several runs
* Residual MLP struggled with Analysis precision despite strong recall
* CatBoost improved stability on Worms and Shellcode
* XGBoost produced the strongest overall F1-Score values

---

## Training & Runtime Comparison

| Model        | Approx Runtime | GPU Usage | Notes                  |
| ------------ | -------------- | --------- | ---------------------- |
| XGBoost      | ~45 min        | Moderate  | Fastest convergence    |
| CatBoost     | ~50 min        | Moderate  | Most stable training   |
| TabNet       | ~90 min        | High      | Most expensive model   |
| Residual MLP | ~35 min        | Moderate  | Best neural efficiency |

---

### Performance Summary

| Model        | Binary Detection | Full Multiclass | Attack-Only | Overall Trend              |
| ------------ | ---------------- | --------------- | ----------- | -------------------------- |
| XGBoost      | Best             | Best            | Best        | Strongest overall baseline |
| CatBoost     | Very Strong      | Strong          | Strong      | Stable under imbalance     |
| TabNet       | Strong           | Moderate        | Strong      | Best interpretability      |
| Residual MLP | Strong           | Strong          | Strong      | Best neural efficiency     |

## Overall Behaviour

The experiments show clear differences between classical boosting methods and deep learning approaches.

| Model        | Training Speed | Inference Speed | Rare Attack Performance | Interpretability | Memory Usage |
| ------------ | -------------- | --------------- | ----------------------- | ---------------- | ------------ |
| XGBoost      | Very Fast      | Very Fast       | Moderate                | High             | Moderate     |
| CatBoost     | Fast           | Fast            | Strong                  | High             | Moderate     |
| TabNet       | Slow           | Moderate        | Strong                  | Very High        | High         |
| Residual MLP | Fast           | Very Fast       | Strong                  | Moderate         | Moderate     |

---

## Binary Detection Performance

### General Findings

All models achieve very high binary detection capability because separating benign from malicious traffic is easier than distinguishing attack categories.

Observed behaviour:

* Tree models converge fastest
* Neural networks require longer warmup
* CatBoost produces the most stable validation curves
* Residual MLP achieves strong throughput-performance balance

---

## Full Multiclass Performance

### Most Difficult Classes

The following attack types are consistently difficult across all models:

* Worms
* Shellcode
* Analysis
* Fuzzers
* DDoS-SlowRate

Main reasons:

* Very small sample counts
* Overlapping traffic signatures
* Similar flow statistics
* High variance distributions

### Most Separable Classes

The easiest classes include:

* DDoS
* DoS
* Reconnaissance
* Scanning

These attacks exhibit stronger statistical patterns in:

* Packet counts
* Throughput
* TTL statistics
* Directional asymmetry

---

## Rare-Class Behaviour

### Best Rare-Class Stability

| Model        | Rare-Class Behaviour           |
| ------------ | ------------------------------ |
| CatBoost     | Most stable                    |
| TabNet       | Strong representation learning |
| Residual MLP | Competitive with weighting     |
| XGBoost      | Sensitive to imbalance         |

The combination of:

* log-smoothed weights
* engineered features
* balanced sampling
* robust preprocessing

substantially improves macro-level performance.

---

# Evaluation Metrics

Every notebook evaluates models using:

| Metric            | Purpose                      |
| ----------------- | ---------------------------- |
| Accuracy          | Overall correctness          |
| Weighted Avg F1   | Class-balanced evaluation    |
| Balanced Accuracy | Imbalance-aware accuracy     |
| ROC-AUC           | Binary separability          |
| PR-AUC            | Precision-recall behaviour   |
| Precision         | False positive control       |
| Recall            | Attack detection sensitivity |
| Confusion Matrix  | Per-class analysis           |

---

# Visualisations Included

Each modelling notebook generates:

* Learning curves
* ROC curves
* Precision-recall curves
* Normalised confusion matrices
* Per-class F1 charts
* Precision/recall heatmaps
* Feature importance rankings
* Published-paper comparison charts

Additional model-specific visualisations:

| Model        | Unique Visualisation              |
| ------------ | --------------------------------- |
| TabNet       | Attention masks                   |
| Residual MLP | Gradient importance maps          |
| XGBoost      | Gain importance                   |
| CatBoost     | PredictionValuesChange importance |

---

# Comparison with Published Research

The repository includes comparisons against previously published work on NF-UQ-NIDS-v2.

Compared approaches include:

* Random Forest
* CNNs
* LSTMs
* Transformers
* GNNs
* Extra Trees
* Hybrid architectures

The goal is not only to maximise accuracy, but also to compare:

* scalability
* training efficiency
* interpretability
* robustness under imbalance
* deployment practicality

---

# Hardware & Runtime

All notebooks are designed for:

| Resource | Configuration |
| -------- | ------------- |
| Platform | Kaggle        |
| GPU      | NVIDIA T4     |
| VRAM     | 15 GB         |
| RAM      | 30 GB         |

### Approximate Runtime

| Notebook      | Runtime |
| ------------- | ------- |
| EDA           | ~15 min |
| Preprocessing | ~20 min |
| XGBoost       | ~45 min |
| CatBoost      | ~50 min |
| TabNet        | ~90 min |
| Residual MLP  | ~35 min |

---

# Dependencies

```text
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
matplotlib
seaborn
joblib
psutil
xgboost
catboost
pytorch-tabnet
torch==2.3.0
```

---

# How to Run

1. Upload NF-UQ-NIDS-v2 to Kaggle
2. Enable GPU acceleration
3. Run notebooks in order:

```text
01 → EDA
02 → Preprocessing
03 → XGBoost
04 → CatBoost
05 → TabNet
06 → Residual MLP
```

4. Download generated plots from the corresponding `/plots/` directory.

---

# Key Takeaways

* Gradient boosting remains extremely competitive for IDS tasks
* CatBoost provides the best balance between stability and performance
* TabNet offers the best interpretability among neural models
* Residual MLP achieves strong performance with significantly lower complexity
* Proper preprocessing and imbalance handling matter as much as architecture choice
* Rare-class evaluation is essential for realistic IDS benchmarking

---

# Author

**Hossam Hamdy**

Gmail:
[Email Me](mailto:hossam3759180@gmail.com)

Kaggle:
[https://www.kaggle.com/hossamhamdyfakry](https://www.kaggle.com/hossamhamdyfakry)
