---

## Day 3 — Sklearn Classifiers & Model Evaluation

**Notebook:** `day3_sklearn_classifiers_titanic.ipynb`

### 🎯 Objective
Train and compare 3 machine learning classifiers on the
cleaned Titanic dataset from Day 2. Evaluate using accuracy,
F1 score, confusion matrix and cross validation.

### 🤖 Models Trained
| Model | Accuracy | F1 Score | CV Score |
|-------|----------|----------|----------|
| Logistic Regression | 84.9% | 0.79 | Best ✅ |
| Random Forest | 82.7% | 0.76 | Good |
| SVM (after scaling) | ~83% | ~0.77 | Good |

### 🏆 Best Model: Logistic Regression
- Highest cross validation score across 5 folds
- Most consistent and generalizable performance
- Simple, interpretable — ideal for production deployment

### 💡 Key Findings
- Feature scaling was critical — SVM accuracy jumped from
  67% to ~83% after applying StandardScaler
- Engineered features (FamilySize, FarePerPerson, AgeGroup)
  ranked in top 5 most important features confirming Day 2
  feature engineering added real predictive value
- Sex and Fare remained the strongest survival predictors

### 📊 Visualizations
![Confusion Matrices](plots/plot4_confusion_matrices.png)
![Feature Importance](plots/plot5_feature_importance.png)

### 🛠️ Tech Stack
Python · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · joblib

### 📂 Saved Models
| File | Description |
|------|-------------|
| models/logisticregression.pkl | Best model |
| models/randomforest.pkl | Ensemble model |
| models/svm.pkl | SVM with RBF kernel |
| models/scaler.pkl | StandardScaler for new data |