import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math

df = pd.read_csv("Updated_Obesity_Dataset.csv")

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")

# Numeric columns → fill with mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    null_count = df[col].isnull().sum()          # check BEFORE filling
    if null_count > 0:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)        # ← fixed: no inplace warning
        print(f"[NULL FIX] '{col}' — filled {null_count} nulls with mean = {mean_val:.4f}")

# Categorical columns → fill with MODE for object columns (safety net)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    null_count = df[col].isnull().sum()          # check BEFORE filling
    if null_count > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)        # ← fixed: no inplace warning
        print(f"[NULL FIX] '{col}' — filled {null_count} nulls with mode = '{mode_val}'")

print(f"\nMissing values after fix: {df.isnull().sum().sum()}")  # Should be 0

def standard_round(x):
    """Standard rounding: .5 and above → up, below .5 → down"""
    return math.floor(x + 0.5)

cols_to_round = ['Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

for col in cols_to_round:
    if col in df.columns:
        df[col] = df[col].apply(standard_round)
        print(f"[ROUNDED] '{col}' → sample: {df[col].head(5).tolist()}")


# --- 4a. Binary Encoding (yes=1, no=0) ---
binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})
        print(f"[ENCODED - Binary] '{col}'")

# --- 4b. Gender ---
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    print("[ENCODED - Binary] 'Gender' (Male=1, Female=0)")

# --- 4c. Ordinal Encoding (frequency-based order) ---
caec_order = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
if 'CAEC' in df.columns:
    df['CAEC'] = df['CAEC'].map(caec_order)
    print("[ENCODED - Ordinal] 'CAEC'")

calc_order = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
if 'CALC' in df.columns:
    df['CALC'] = df['CALC'].map(calc_order)
    print("[ENCODED - Ordinal] 'CALC'")

# --- 4d. One-Hot Encoding (MTRANS — no natural order) ---
if 'MTRANS' in df.columns:
    df = pd.get_dummies(df, columns=['MTRANS'], prefix='MTRANS', dtype=int)
    print("[ENCODED - One-Hot] 'MTRANS'")

# --- 4e. Target Column — Ordinal Encoding ---
obesity_order = {
    'Insufficient_Weight': 0,
    'Normal_Weight':       1,
    'Overweight_Level_I':  2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I':      4,
    'Obesity_Type_II':     5,
    'Obesity_Type_III':    6
}
if 'NObeyesdad' in df.columns:
    df['NObeyesdad'] = df['NObeyesdad'].map(obesity_order)
    print("[ENCODED - Ordinal] 'NObeyesdad' (target)")


scale_cols = ['Age', 'Height', 'Weight']
scale_cols = [col for col in scale_cols if col in df.columns]

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print(f"\n[SCALED - MinMax] Columns: {scale_cols}")

print("\n" + "=" * 50)
print("PROCESSED DATASET")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values: {df.isnull().sum().sum()}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nStatistics:\n{df.describe()}")


df.to_csv("Obesity_Preprocessed.csv", index=False)
print("\n✅ Saved → Obesity_Preprocessed.csv")