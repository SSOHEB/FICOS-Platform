# 25-Point Leakage & Empirical Integrity Audit Checklist

This machine-auditable checklist establishes the verification standards required for every training run, feature calculation, and validation step in the Google Colab benchmark.

---

### Part I: Temporal Splitting & Index Integrity
- [x] **Audit Check 1 (Strict Chronological Ordering)**: The dataset index is strictly sorted by ascending calendar date. `df['date'].is_monotonic_increasing == True`.
- [x] **Audit Check 2 (Duplicate Date Prohibition)**: Zero duplicate dates exist in the primary temporal index. `df['date'].duplicated().sum() == 0`.
- [x] **Audit Check 3 (Strict Timestamp Disjointness)**: No overlap between splits:
  - `TRAIN`: $\text{date} < \text{2023-01-01}$ ($N=1699$)
  - `DEV/VAL`: $\text{2023-01-01} \le \text{date} < \text{2025-01-01}$ ($N=480$)
  - `LOCKED TEST`: $\text{date} \ge \text{2025-01-01}$ ($N=402$)
- [x] **Audit Check 4 (Zero Random Permutation)**: Random train/test splits (`train_test_split(shuffle=True)`) are strictly prohibited.
- [x] **Audit Check 5 (Walk-Forward Isolation)**: All cross-validation folds exist entirely inside $\text{date} < \text{2025-01-01}$.

---

### Part II: Feature Engineering & Timestamp Alignment
- [x] **Audit Check 6 (No Negative Shifts)**: No feature uses negative `.shift(-k)` shifts. Any negative shift in a feature column triggers an immediate fatal assertion error.
- [x] **Audit Check 7 (Strictly Backward-Looking Rolling Windows)**: Every rolling window metric (`rolling_mean`, `rolling_std`, `rolling_min`, `rolling_max`) uses `center=False` and includes observations strictly through date $t$.
- [x] **Audit Check 8 (Target Column Cleanliness)**: All `target_*` and `dir_*` columns from raw files are stripped from the candidate feature matrix $X$ prior to model ingestion.
- [x] **Audit Check 9 (Lagged Weather & Cyclone Inputs)**: Port weather and cyclone risk features reflect information reported prior to prediction day $t$.
- [x] **Audit Check 10 (Lagged News & Sentiment Signals)**: GDELT event indices and tone features are lagged by at least 1 day ($t-1$) to eliminate intra-day release lookahead.
- [x] **Audit Check 11 (Volatility Normalizer Alignment)**: For Target D ($\Delta / \sigma$), the denominator rolling standard deviation $\sigma_{30d}$ is computed using historical daily changes up to date $t$. Zero future volatility lookahead is permitted.

---

### Part III: Preprocessing & Transformer Pipeline Isolation
- [x] **Audit Check 12 (Training-Only Imputation)**: Missing value imputation statistics (medians) are calculated exclusively on the training fold slice and applied forward to validation/test.
- [x] **Audit Check 13 (Training-Only Feature Scaling)**: `StandardScaler.fit()` is called only on the training fold data. Validation and test sets are transformed via `.transform()`.
- [x] **Audit Check 14 (Training-Only Feature Selection)**: Feature selection algorithms (`SelectKBest`, `f_regression`, `f_classif`) are fit inside each training fold without access to validation or test targets.
- [x] **Audit Check 15 (Target Transformation Reversibility)**: Predictions made on log return or percentage return are mapped back to actual price level and delta units using only base price $y_t$ at prediction time.

---

### Part IV: Model Selection & Hyperparameter Discipline
- [x] **Audit Check 16 (Predeclared Hyperparameter Grids)**: Hyperparameter search spaces are fixed in `experiment_config.json` before viewing validation results.
- [x] **Audit Check 17 (Frozen Model Finalists)**: No hyperparameter, feature set, or model family modification is permitted after crossing into locked test evaluation.
- [x] **Audit Check 18 (Direct Horizon Specialization)**: Separate, independent models are evaluated for 7d, 14d, and 30d; no single model is forced across all horizons.
- [x] **Audit Check 19 (No Forced Global Winner)**: Different assets and horizons may select different winning models or conclude `NO RELIABLE EDGE`.
- [x] **Audit Check 20 (Plateau Stability Enforcement)**: Hyperparameters showing an isolated spike surrounded by sharp drops (>20 pp swing) are flagged as `UNSTABLE` and disqualified from promotion.

---

### Part V: Locked-Test Evaluation & Statistical Honesty
- [x] **Audit Check 21 (Hard Test Lock Boundary)**: Cell 15 asserts the irreversible lock state `===== TEST SET LOCKED =====`. The test set is evaluated exactly once per locked finalist.
- [x] **Audit Check 22 (Uncertainty Calibration Isolation)**: Empirical residual quantiles ($P_{10}, P_{90}$) and threshold $\tau$ are estimated exclusively from validation residuals, never from test outcomes.
- [x] **Audit Check 23 (Serial Dependence & Overlap Caveat)**: $h$-day forward targets create overlapping serial correlation. Both raw test count $N$ and effective sample size $N_{\text{eff}} \approx N / h$ are reported.
- [x] **Audit Check 24 (Dual Confidence Intervals)**: Non-parametric test bootstrap intervals are reported alongside exact binomial intervals (Clopper-Pearson / Wilson) to prevent sample-size distortion on small $N$.
- [x] **Audit Check 25 (Zero Accuracy Manufacturing)**: If candidate models do not meaningfully beat naive or existing Ridge baselines on locked test, the model is designated `NO RELIABLE EDGE`.
