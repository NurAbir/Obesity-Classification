# Obesity Classification — AI Project

A supervised and unsupervised machine learning project that classifies obesity levels from lifestyle and biometric features. Five classifiers (KNN, Decision Tree, Logistic Regression, Naive Bayes, Neural Network) are benchmarked against each other, and K-Means clustering is applied as an unsupervised baseline.

---

## Project Structure

```
obesity-classification/
├── data/
│   ├── obesity_dataset.csv          # Raw dataset (input to preprocessing)
│   └── obesity_preprocessed.csv     # Cleaned, encoded, scaled dataset
├── docs/
│   ├── obesity_report.pdf           # Project report
│   └── project_report_template.pdf  # Report template
├── preprocessing.py                 # Standalone preprocessing script
├── main.ipynb                       # Full analysis notebook (EDA → models → evaluation)
├── .gitignore
└── README.md
```

---

## Dataset

The dataset contains lifestyle, dietary, and biometric attributes used to predict obesity category. The target column `NObeyesdad` has 7 ordinal classes:

| Value | Class |
|-------|-------|
| 0 | Insufficient Weight |
| 1 | Normal Weight |
| 2 | Overweight Level I |
| 3 | Overweight Level II |
| 4 | Obesity Type I |
| 5 | Obesity Type II |
| 6 | Obesity Type III |

---

## Preprocessing (`preprocessing.py`)

Run this script to regenerate `obesity_preprocessed.csv` from the raw dataset:

```bash
python preprocessing.py
```

Steps performed:
- Missing value imputation (mean for numeric, mode for categorical)
- Standard rounding for selected columns (`Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`)
- Binary encoding (`Gender`, `family_history_with_overweight`, `FAVC`, `SMOKE`, `SCC`)
- Ordinal encoding (`CAEC`, `CALC`, target `NObeyesdad`)
- One-hot encoding (`MTRANS`)
- Min-Max scaling (`Age`, `Height`, `Weight`)

> **Note:** The script reads `Updated_Obesity_Dataset.csv` and writes `Obesity_Preprocessed.csv` in the current working directory. Run it from inside the `data/` folder or update the file paths accordingly.

---

## Notebook (`main.ipynb`)

The main notebook covers the full ML pipeline:

1. **Dataset Preprocessing** — mirrors `preprocessing.py` with inline output
2. **Exploratory Data Analysis (EDA)** — summary stats, histograms, box plots, correlation heatmaps, class balance check
3. **Data Splitting** — stratified 70 / 15 / 15 train / validation / test split
4. **Encoding & Scaling** — StandardScaler applied post-split
5. **Model Training & Testing**
   - K-Nearest Neighbors (k=7)
   - Decision Tree (max_depth=10, balanced class weights)
   - Logistic Regression (max_iter=2000, balanced class weights)
   - Naive Bayes (GaussianNB)
   - Neural Network (MLP, 3 hidden layers, softmax output)
6. **Model Comparison** — accuracy bar chart, precision/recall/F1 comparison, confusion matrices, ROC curves & AUC scores, train/val/test accuracy comparison
7. **Unsupervised Learning** — K-Means clustering (k=7, elbow method), PCA visualization, cluster profiling

> **Note:** The notebook was originally developed on Google Colab. Cells using `from google.colab import files` (file upload/download) should be replaced with local file reads when running locally.

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

### Running locally

```bash
# 1. Clone / download the repo
cd obesity-classification

# 2. (Optional) Regenerate the preprocessed CSV
cd data
python ../preprocessing.py
cd ..

# 3. Launch the notebook
jupyter notebook main.ipynb
```

---

## Models & Results

Five supervised models were trained and evaluated on the same held-out test set. Class imbalance (mild, ~1.3:1 ratio) was handled via balanced class weights. Evaluation metrics include accuracy, precision, recall, F1 score, and AUC-ROC (One-vs-Rest).

Refer to the comparison section in `main.ipynb` or `docs/obesity_report.pdf` for full results.
