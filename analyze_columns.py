"""
Strategic analysis: investigate all column meanings and TVT formula discovery
"""
import pandas as pd
import numpy as np
from pathlib import Path

wid = '00bbac68'
hw_train = pd.read_csv(f'train/{wid}__horizontal_well.csv')
hw_test  = pd.read_csv(f'test/{wid}__horizontal_well.csv')
tw = pd.read_csv(f'test/{wid}__typewell.csv')

mask = hw_test['TVT_input'].isna() | (hw_test['TVT_input'].astype(str).str.strip() == '')
ps = int(mask.idxmax()) if mask.any() else len(hw_test)

print(f"=== COLUMNS in horizontal well ===")
print(hw_train.columns.tolist())
print()

print(f"=== POST-PS ROWS: are ANCC/ASTNU etc. populated in TEST file? ===")
post_test = hw_test.iloc[ps:ps+5]
print(post_test[['MD', 'X', 'Y', 'Z', 'ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA', 'TVT_input']].to_string())
print()

print(f"=== SAME ROWS IN TRAINING FILE ===")
post_train = hw_train.iloc[ps:ps+5]
print(post_train[['MD', 'X', 'Y', 'Z', 'ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA', 'TVT']].to_string())
print()

# Check NaN status for each column in post-PS test
print("=== NaN counts in test POST-PS rows ===")
post_all = hw_test.iloc[ps:]
for col in hw_test.columns:
    n_nan = post_all[col].isna().sum()
    pct = 100*n_nan/len(post_all)
    print(f"  {col}: {n_nan}/{len(post_all)} NaN ({pct:.1f}%)")
print()

# Try to derive TVT formula from training data
print("=== TVT FORMULA INVESTIGATION ===")
cols = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
pre = hw_train.iloc[:ps].copy()

# Compute mean of all boundary depth columns
for col in cols:
    print(f"  {col}: min={pre[col].min():.2f}, max={pre[col].max():.2f}, NaN={pre[col].isna().sum()}")

# TVT formula hypothesis: TVT = f(ANCC, ASTNU, etc.)
# All these Z-like columns are negative, TVT is positive
# Hypothesis: TVT = some reference - |mean of boundary cols|
mean_z_cols = pre[cols].mean(axis=1)
print(f"\n  Mean of Z-boundary cols: {mean_z_cols.head(3).values}")
print(f"  TVT values:              {pre['TVT'].head(3).values}")
print(f"  -mean_z: {(-mean_z_cols).head(3).values}")

# Check correlation with TVT
from scipy.stats import pearsonr
for col in cols:
    valid = pre[[col, 'TVT']].dropna()
    r, _ = pearsonr(valid[col], valid['TVT'])
    print(f"  corr({col}, TVT) = {r:.6f}")

# Specific formula test
print("\n  TVT vs -ANCC:")
sample = pre[['TVT', 'ANCC']].dropna().head(5)
print(sample.assign(neg_ANCC=-sample['ANCC'], diff=lambda d: d['TVT'] + d['ANCC']).to_string())
