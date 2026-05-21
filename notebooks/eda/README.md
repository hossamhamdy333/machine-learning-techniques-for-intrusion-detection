# Notebook 01 — Exploratory Data Analysis

**File:** `1-ids-eda.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/1-ids-eda)
**Plots:** [`/plots/eda/`](../../plots/eda/)

---

## What this notebook does

The raw NF-UQ-NIDS-v2 CSV is ~76 million rows and does not fit in memory at once. This notebook reads it in chunks of 500,000 rows, applies stratified sampling (33% per class per chunk), and builds a representative in-memory subset for analysis.

### Steps

1. **Environment check** — GPU, CPU, RAM, Python and library versions
2. **Chunked loading** — 500k-row chunks, deduplication per chunk, 33% stratified sample
3. **Data quality**
   - Shape, dtypes, missing value counts
   - Infinity value detection and replacement
   - Corrupted label check (malformed binary/multiclass targets)
   - Duplicate row count
   - Constant feature detection
4. **Class distribution**
   - Binary (Benign vs Attack) — bar + pie chart
   - Multiclass (21 classes) — horizontal bar chart with counts and percentages
   - Rare class report (classes with fewer than 1,000 samples)
5. **Statistical profiling**
   - `.describe()` — mean, std, min/max, percentiles
   - Skewness and kurtosis per feature
   - Features flagged for log transform (|skew| > 1.0)
6. **Visualisations**
   - Feature histograms (original scale)
   - Feature histograms (log1p scale for high-skew features)
   - Skewness and kurtosis bar charts
   - Correlation matrix (50k-row sample, full 42-feature heatmap)
7. **Multicollinearity analysis**
   - Pairs with Pearson correlation > 0.95
   - Weakest-of-pair dropped based on target correlation
8. **Mutual information** — MI scores vs binary target, top features ranked
9. **Feature distributions by attack type** — violin/strip plots for top 6 MI features
10. **PCA** — 2-component projection, Benign vs Attack, variance explained

### Outputs saved

All plots are saved to `/kaggle/working/plots_eda/` and zipped for download.

---

## Key findings

- **76.4M total records**, reduced to ~25M after 33% stratified sampling
- **3 dominant classes** (Benign, DDoS, DoS) account for ~85% of traffic
- **8 rare classes** with fewer than 5,000 samples — significant class imbalance
- **All features are numerical** — no categorical columns, no text
- **~35% of features have |skew| > 1.0** and benefit from log1p transform
- **Multiple highly correlated pairs** identified (e.g. MAX_IP_PKT_LEN / LONGEST_FLOW_PKT)
- **PCA explains only 27.3% variance in 2 components** — high intrinsic dimensionality, non-linear separability
- **Top MI features:** `LONGEST_FLOW_PKT`, `MAX_IP_PKT_LEN`, `IN_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`, `MIN_TTL`, `MAX_TTL`
