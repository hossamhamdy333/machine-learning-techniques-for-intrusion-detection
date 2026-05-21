# Notebook 02 — Preprocessing & Feature Engineering

**File:** `2-ids-preprocessing.ipynb`
**Kaggle:** [Run on Kaggle](https://www.kaggle.com/hossamhamdyfakry/2-ids-preprocessing)
**Plots:** [`/plots/preprocessing/`](../../plots/preprocessing/)

---

## What this notebook does

Transforms the raw sampled data into clean, scaled, split arrays ready for model training. All artefacts (scaler, encoder, arrays) are saved and reused by every downstream model notebook.

### Steps

1. **Chunked loading** — same pipeline as EDA (33% stratified sample, deduplication)
2. **Data cleaning**
   - Replace ±Inf with NaN
   - Median imputation (fitted on full sample, saved as `imputer.pkl`)
3. **Feature engineering** — 12 new domain-specific features added on top of the 39 original features

| Feature | Description |
|---------|-------------|
| `bytes_per_pkt_in` | Avg payload size per inbound packet — small = flood/scan |
| `bytes_per_pkt_out` | Avg payload size per outbound packet |
| `upload_ratio` | Outbound bytes / total bytes — high in exfiltration |
| `byte_asymmetry` | Absolute difference between in/out bytes |
| `pkt_asymmetry` | Absolute difference between in/out packets |
| `is_http_https` | Flag: destination port 80 or 443 |
| `is_ssh_telnet` | Flag: destination port 22 or 23 |
| `is_dns` | Flag: destination port 53 |
| `ttl_range` | MAX_TTL − MIN_TTL — path diversity proxy |
| `syn_only_flag` | TCP_FLAGS == 0x02 — SYN without ACK (scan indicator) |
| `retransmit_pkt_ratio` | Retransmitted packets / total packets |
| `flow_duration_ms` | Flow duration in milliseconds |

4. **Multicollinearity pruning** — 9 features dropped (highest-correlated pairs where the weaker member has lower MI with target):
   `MAX_IP_PKT_LEN`, `ICMP_IPV4_TYPE`, `MAX_TTL`, `CLIENT_TCP_FLAGS`, `LONGEST_FLOW_PKT`, `RETRANSMITTED_OUT_PKTS`, `retransmit_pkt_ratio`, `NUM_PKTS_1024_TO_4096_BYTES`, `SHORTEST_FLOW_PKT`

5. **Label encoding** — `LabelEncoder` fitted on 21 attack class names → saved as `label_encoder.pkl`

6. **Log1p transform** — applied in-place to all features with |skew| > 1.0

7. **Train / Val / Test split** — 60 / 20 / 20, stratified on multiclass target to guarantee rare class representation in every split

8. **RobustScaler** — fitted on training data only, applied to val and test (prevents data leakage)

9. **Class weights** — `compute_class_weight('balanced')` on training labels, saved as `class_weights.pkl`; log-smoothed version also saved for use with neural network models

10. **Mutual information** — computed on 200k training samples, column order saved as `mi_selected_cols.pkl` (feature name list in MI rank order)

11. **Attack-only split** — separate arrays for Model 3 (rows where binary label = 1), saved as `X_train_att.npy` etc.

### Saved artefacts

| File | Contents |
|------|----------|
| `imputer.pkl` | Fitted `SimpleImputer` (median) |
| `scaler.pkl` | Fitted `RobustScaler` |
| `label_encoder.pkl` | Fitted `LabelEncoder` (21 classes) |
| `class_weights.pkl` | Balanced class weight dict (multiclass) |
| `class_weight_att_dict.pkl` | Balanced class weight dict (attack-only) |
| `mi_selected_cols.pkl` | Feature name list in MI rank order |
| `X_train.npy` | Training features (float32) |
| `X_val.npy` | Validation features |
| `X_test.npy` | Test features |
| `y_train_bin.npy` | Binary training labels |
| `y_val_bin.npy` | Binary validation labels |
| `y_test_bin.npy` | Binary test labels |
| `y_train_multi.npy` | Multiclass training labels (encoded) |
| `y_val_multi.npy` | Multiclass validation labels |
| `y_test_multi.npy` | Multiclass test labels |
| `X_train_att.npy` | Attack-only training features |
| `X_val_att.npy` | Attack-only validation features |
| `X_test_att.npy` | Attack-only test features |
| `y_train_att_multi.npy` | Attack-only multiclass training labels |
| `y_val_att_multi.npy` | Attack-only multiclass validation labels |
| `y_test_att_multi.npy` | Attack-only multiclass test labels |

---

## Split sizes

| Split | Rows | Features |
|-------|------|---------|
| Train | ~13.7M → subsampled to ~5.2M in model notebooks | 42 |
| Val | ~4.6M | 42 |
| Test | ~4.6M | 42 |
