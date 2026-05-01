# Credit Risk Data Preparation Pipeline - Complete Guide

## 📋 Overview

This guide explains the **60/20/20 data split strategy**, how to implement it, and what comes after data preparation.

---

## 1️⃣ The Train/Validation/Test Split Explained

### Why Split the Data?

When building ML models, you need **three independent datasets** to avoid overfitting:

```
Total Dataset (100%)
│
├─ Training Set (60%)
│  └─ Used to train the model and learn patterns
│
├─ Validation Set (20%)
│  └─ Used to tune hyperparameters and select best model
│
└─ Test Set (20%)
   └─ Used for final evaluation on unseen data
```

### The Problem Without Splitting

If you use the same data to train AND evaluate:
- ❌ Model memorizes the data (overfitting)
- ❌ You get unrealistic performance metrics
- ❌ Low performance on real-world data

### The 60/20/20 Strategy

**Training Set (60%)**
- Largest set for model to learn from
- Model sees patterns, learns weights, minimizes loss
- Example: 7,738 samples

**Validation Set (20%)**
- Independent data to check model performance
- Used to:
  - Monitor performance during training
  - Tune hyperparameters (learning rate, regularization, etc.)
  - Select the best model version
  - Detect overfitting early
- Example: 2,580 samples

**Test Set (20%)**
- Completely unseen data held until end
- Used for final performance report
- Gives realistic estimate of model performance
- Example: 2,580 samples

---

## 2️⃣ How the Split is Implemented

### Step-by-Step Process

```python
# Step 1: First split - 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Result: 80% (9,518) for train+val, 20% (2,580) for test

# Step 2: Second split - Split 80% into 75% train and 25% val
# 75% of 80% = 60% (7,738 train)
# 25% of 80% = 20% (2,580 val)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

# Step 3: Scale features (IMPORTANT: fit ONLY on training data to prevent data leakage)
scaler = StandardScaler().set_output(transform="pandas")
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
```

### Important: Stratification

```python
stratify=y  # Maintains target distribution in all three sets
```

**What does stratification do?**
- Ensures each split has same class balance (15% defaults)
- Prevents random splits where validation might have 5% defaults and test has 25%
- Results in reliable performance estimates

**Example with our data:**
```
Before split:
- Original: 15% defaults (1,935 / 12,898)

After split with stratification:
- Training:   15.00% defaults ✓
- Validation: 15.00% defaults ✓
- Test:       15.00% defaults ✓
```

### Key Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `test_size=0.2` | 20% | Hold out 20% for final testing |
| `test_size=0.25` (second split) | 25% | Split remaining 80% into 60/20 |
| `random_state=42` | Fixed seed | Reproducibility |
| `stratify=y` | Balance maintained | Representative splits |

---

## 3️⃣ Your Current Data Split

### Raw Data
```
Raw Dataset: 30,000 records
│
├─ Issue: Contains outliers & duplicates
└─ Action: Data cleaning
```

### After Cleaning
```
Cleaned Dataset: 12,898 records (43% retained)
│
├─ Removed 35 duplicates
├─ Removed 17,067 outliers (IQR method)
└─ Target: 15% defaults (balanced)
```

### Final Split
```
Cleaned Data: 12,898 records
│
├─ Training Set:   7,738 (60%) - Learn patterns
├─ Validation Set: 2,580 (20%) - Tune model
└─ Test Set:       2,580 (20%) - Final evaluation
```

### Verification
All three sets have:
- ✓ Same default rate (~15%)
- ✓ Same features (23 features, standardized)
- ✓ No overlap (mutually exclusive)
- ✓ No data leakage

---

## 4️⃣ What Happens After

### Phase 1: Model Training & Selection (Using Training + Validation Sets)

```
+──────────────┐
│ Training Set │ (7,738 samples)
│   60%        │
└──────┬───────┘
       │
       ├─ Train Model_A
       ├─ Train Model_B
       ├─ Train Model_C
       │
       ↓
+──────────────┐
│ Validation   │ (2,580 samples)
│   20%        │ Evaluate each model
└──────┬───────┘
       │
       ├─ Model_A: 82% Accuracy
       ├─ Model_B: 87% Accuracy ✓ Best
       ├─ Model_C: 79% Accuracy
       │
       ↓
+──────────────┐
│ Select       │
│ Best Model   │
└──────────────┘
```

### Phase 2: Final Evaluation (Using Test Set)

```
+──────────────┐
│ Best Model   │
│ (from Phase) │
└──────┬───────┘
       │
       ↓
+──────────────┐
│ Test Set     │ (2,580 samples)
│   20%        │ Never seen before!
└──────┬───────┘
       │
       ├─ Accuracy: 85%
       ├─ Precision: 0.88
       ├─ Recall: 0.82
       ├─ F1-Score: 0.85
       │
       ↓
+──────────────┐
│ Final Report │
└──────────────┘
```

### Remember: Test Set is Sacred!
- ⚠️ **Never** touch test set until ready for final evaluation
- ⚠️ **Never** use test set for hyperparameter tuning
- ⚠️ **Never** use test results to retrain the model
- ✓ This ensures unbiased performance estimate

---

## 5️⃣ Workflow After Data Preparation

### Next Steps (In Order)

```
1. Exploratory Data Analysis (EDA)
   └─ Use training set only
   └─ Understand patterns, distributions, correlations
   └─ Create: EDA.ipynb ✓ (Already created)

2. Feature Engineering
   └─ Create/transform features based on EDA insights
   └─ Encode categorical variables (if needed)
   └─ Handle special cases

3. Model Selection
   └─ Try multiple algorithms:
      - Logistic Regression
      - Random Forest
      - XGBoost
      - Neural Networks
   └─ Train on 60% (training set)
   └─ Evaluate on 20% (validation set)

4. Hyperparameter Tuning
   └─ Use validation set to fine-tune:
      - Learning rate
      - Tree depth
      - Regularization strength
   └─ Try different combinations
   └─ Select best configuration

5. Final Evaluation
   └─ Use best model + best hyperparameters
   └─ Evaluate on test set (20%)
   └─ Generate final metrics & report

6. Deployment
   └─ Save trained model
   └─ Create prediction pipeline
   └─ Monitor performance in production
```

---

## 6️⃣ Files Generated

### Data Files
```
data/processed/
├── X_train.csv         (7,738 × 23) - Training features
├── X_val.csv           (2,580 × 23) - Validation features
├── X_test.csv          (2,580 × 23) - Test features
├── y_train.csv         (7,738 × 1)  - Training targets
├── y_val.csv           (2,580 × 1)  - Validation targets
├── y_test.csv          (2,580 × 1)  - Test targets
└── scaler.pkl          StandardScaler for feature scaling
```

### Code Files
```
src/
├── data_preparation.ipynb    - Interactive notebook (documented)
├── data_preparation.py       - Reusable Python script
└── EDA.ipynb                 - Exploratory Data Analysis
```

### Logs & Statistics
```
logs/
└── cleaning_stats_*.json     - Data cleaning statistics
```

---

## 7️⃣ Using the Data Preparation Script

### Run Standalone Python Script

```bash
cd /home/abood/ML_Model_Monitoring_System_for_Credit_Risk
python src/data_preparation.py
```

Output:
```
============================================================
🔄 CREDIT RISK DATA PREPARATION PIPELINE
============================================================
✓ Loaded data: (30000, 24)
📋 DATA INSPECTION: ...
🧹 DATA CLEANING: ...
✂️  TRAIN/VAL/TEST SPLIT: ...
💾 DATA SAVED to ...
✅ PIPELINE COMPLETE!
```

### Use in Your Code

```python
from src.data_preparation import DataPreparation

# Create pipeline instance
pipeline = DataPreparation('data/raw/credit_risk.csv')

# Run complete pipeline
X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run_pipeline()

# Or run step-by-step
pipeline.load_data().inspect_data().clean_data()
pipeline.extract_features_target()
X_train, X_val, X_test, y_train, y_val, y_test = pipeline.scale_and_split()
pipeline.save_data('data/processed', X_train, X_val, X_test, y_train, y_val, y_test)
```

---

## 8️⃣ Key Statistics from Your Pipeline

### Data Cleaning Results
```
Raw Records:         30,000
Duplicates Removed:      35
Outliers Removed:    17,067
                     ------
Final Records:       12,898 (43% retained)
```

### Target Distribution
```
Non-Default (0):  10,963 (85%)
Default (1):       1,935 (15%)
```

### Dataset Sizes
```
Training:    7,738 (60%)  → Model learning
Validation:  2,580 (20%)  → Hyperparameter tuning
Test:        2,580 (20%)  → Final evaluation
Total:      12,898 (100%)
```

### Features
- **Count**: 23 numerical features
- **Scaling**: StandardScaler (mean=0, std=1)
- **Missing**: None
- **Outliers**: Handled

---

## 9️⃣ Common Questions

### Q: Why remove outliers with IQR method?
**A:** IQR (Interquartile Range) removes extreme values that:
- Don't represent typical customers
- Can bias model training
- May be data entry errors
- Reduce model generalization

### Q: Why stratify the split?
**A:** Ensures:
- Training, validation, test have same class balance
- Fair comparison between models
- Reliable performance estimates
- Prevent lucky/unlucky splits

### Q: Can I use test set for validation?
**A:** ❌ Never!
- Test set must be completely unseen
- Any data peeking causes:
  - Overfitting
  - Unrealistic performance metrics
  - Poor real-world results

### Q: Should I retrain after seeing test results?
**A:** ❌ No, that's cheating!
- Test set is for final evaluation only
- If you retrain using test insights, create a NEW test set
- Otherwise, you have data leakage

### Q: What's the hyperparameter tuning process?
**A:** Use validation set:
1. Train models with different hyperparameters
2. Evaluate on validation set
3. Pick best performing hyperparameters
4. Train final model
5. Evaluate once on test set

---

## 🔟 Next: Training Models

Ready to train? Follow this approach:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').iloc[:, 0]
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]

# 1. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Evaluate on validation set
val_pred = model.predict(X_val)
print("VALIDATION RESULTS:")
print(classification_report(y_val, val_pred))

# 3. Adjust hyperparameters if needed, retrain, re-evaluate
# ... (repeat as needed)

# 4. Final evaluation (only once!)
test_pred = model.predict(X_test)
print("\nTEST RESULTS (Final Evaluation):")
print(classification_report(y_test, test_pred))
```

---

## 1️⃣1️⃣ Modular Pipeline Architecture Refactoring

We have recently refactored the original `data_preparation.py` script into a robust, scalable MLOps pipeline. Here is the step-by-step breakdown of what was implemented:

### Step 1: Configuration Setup
We centralized all hardcoded variables and paths into YAML files to make the pipeline configurable without changing code.
- `config/config.yaml`: Defines where to read data from and where to save output artifacts for each stage (Ingestion, Validation, Transformation).
- `schema.yaml`: Contains the exact data types and expected columns of our dataset to strictly catch data drift or schema changes.
- `params.yaml`: Prepared for storing our future model hyperparameters.

### Step 2: Entity Definitions (`src/creditrisk/entity/config_entity.py`)
We created Python `dataclasses` (e.g., `DataIngestionConfig`) to enforce strict type hinting and type safety when passing configurations between components.

### Step 3: Configuration Manager (`src/creditrisk/config/configuration.py`)
We built a centralized `ConfigurationManager` class that uses custom utilities (like `read_yaml` using `ConfigBox` and `ensure_annotations`) to read the YAML files and seamlessly return the defined dataclasses to the pipeline stages.

### Step 4: Independent Pipeline Components
We separated the core logic into modular stages in `src/creditrisk/components/`:
- **Data Ingestion**: Safely transfers the raw `credit_risk.csv` into a managed `artifacts/data_ingestion` directory.
- **Data Validation**: Cross-checks data against `schema.yaml` and verifies that all columns match the expected schema perfectly. It writes a boolean flag to `status.txt`.
- **Data Transformation**: Reads the valid data, applies the strict 60/20/20 train/validation/test split, handles outliers, and scales features securely (fitting only on `X_train`).

### Step 5: Pipeline Construction & Execution
- **Stage Pipelines**: Wrapped each component inside `src/creditrisk/pipeline/` (e.g., `stage_01_data_ingestion.py`) to isolate execution.
- **`main.py` Entry Point**: Created a master script to chain all stages together. Running `python main.py` now executes the entire workflow seamlessly and reliably from start to finish.

---

## Summary

| Phase | Set | Size | Purpose | Access |
|-------|-----|------|---------|--------|
| **Training** | Train | 60% | Learn model | ✓ Open |
| **Tuning** | Val | 20% | Tune params | ✓ Open |
| **Evaluation** | Test | 20% | Final check | ❌ Once only |

**Remember:** Train → Validate → Test → Deploy

Your data is ready! 🎉
