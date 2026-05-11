# Titanic Data Wrangling & EDA
## Day 2 — ML Portfolio | Dr. Muhammad Safyan

### 📌 Project Overview
End-to-end data wrangling and exploratory data analysis on the Titanic dataset
as part of a hands-on ML engineering portfolio.

### 🔧 What I Did
- Loaded and explored dataset (891 rows, 12 columns)
- Handled missing values: Age (median), Embarked (mode), dropped Cabin
- Removed irrelevant columns: Name, Ticket, PassengerId
- Encoded categorical variables: Sex (LabelEncoder), Embarked & Pclass (one-hot)
- Applied log transformation on Fare (skewness: 4.78 → 0.39)
- Engineered 4 new features: FamilySize, IsAlone, AgeGroup, FarePerPerson

### 📊 Key Findings
- Female passengers had ~74% survival rate vs ~19% for males
- 1st class passengers had significantly higher survival rates
- Passengers traveling alone had lower survival rates than families

### 📈 Visualizations
![Survival by Sex](plots/plot1_survival_by_sex.png)
![Age Distribution](plots/plot2_age_distribution.png)
![Correlation Heatmap](plots/plot3_correlation_heatmap.png)

### 🛠️ Tech Stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

### 📂 Files
| File | Description |
|------|-------------|
| day2_data_wrangling_titanic.ipynb | Main notebook |
| data/titanic_clean.csv | Cleaned dataset |
| plots/ | Saved visualizations |