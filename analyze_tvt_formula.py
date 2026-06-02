"""
TVT Formula Investigation: How is TVT computed from boundary sensor columns?
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import warnings; warnings.filterwarnings('ignore')

BCOLS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']

def load_well(wid, split='train'):
    return pd.read_csv(f'{split}/{wid}__horizontal_well.csv')

def get_ps(hw):
    mask = hw['TVT_input'].isna() | (hw['TVT_input'].astype(str).str.strip() == '')
    return int(mask.idxmax()) if mask.any() else len(hw)

# -----------------------------------------------------------
# 1. For test wells (which appear in train): check TVT formula
# -----------------------------------------------------------
print("=== TVT vs boundary columns relationship ===\n")
for wid in ['000d7d20', '00bbac68', '00e12e8b']:
    hw = load_well(wid, 'train')
    ps = get_ps(hw)
    
    # All rows with valid boundary data
    valid = hw[BCOLS + ['TVT', 'Z']].dropna()
    print(f"Well {wid}: {len(valid)} rows with all boundary cols")
    
    # Try linear combinations of boundary cols -> TVT
    X = valid[BCOLS].values
    y = valid['TVT'].values
    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)
    resid = np.abs(pred - y)
    print(f"  Linear regression RMSE: {resid.mean():.4f} ft, max={resid.max():.4f}")
    print(f"  Coefficients: {dict(zip(BCOLS, reg.coef_.round(6)))}")
    print(f"  Intercept: {reg.intercept_:.4f}")
    
    # Try simple mean of boundary cols
    mean_bc = valid[BCOLS].mean(axis=1)
    simple_diff = (valid['TVT'] - (-mean_bc)).abs()
    print(f"  TVT vs -mean(boundary): mean_diff={simple_diff.mean():.4f}, max={simple_diff.max():.4f}")
    
    # Try TVT vs -ANCC (just the first boundary col)
    diff_ancc = (valid['TVT'] - (-valid['ANCC'])).abs()
    print(f"  TVT vs -ANCC: mean_diff={diff_ancc.mean():.4f}, max={diff_ancc.max():.4f}")
    print()

# -----------------------------------------------------------
# 2. Does the formula generalize across ALL training wells?
# -----------------------------------------------------------
print("=== Formula generalization across training wells ===\n")
train_dir = Path('train')
hw_files = sorted(train_dir.glob('*__horizontal_well.csv'))[:50]  # sample 50 wells

all_bc_data = []
for f in hw_files:
    hw = pd.read_csv(f)
    if all(c in hw.columns for c in BCOLS):
        valid = hw[BCOLS + ['TVT', 'Z', 'MD']].dropna()
        all_bc_data.append(valid)

df_all = pd.concat(all_bc_data, ignore_index=True)
print(f"Total rows from {len(hw_files)} wells: {len(df_all)}")

X = df_all[BCOLS].values
y = df_all['TVT'].values
reg_all = LinearRegression().fit(X, y)
pred_all = reg_all.predict(X)
rmse_all = np.sqrt(np.mean((pred_all - y)**2))
print(f"Global linear regression RMSE: {rmse_all:.4f} ft")
print(f"R²: {reg_all.score(X, y):.6f}")
print(f"Coefficients: {dict(zip(BCOLS, reg_all.coef_.round(6)))}")

# Check if any single column explains TVT well
print("\nSingle-column predictors:")
for col in BCOLS + ['Z', 'MD']:
    x = df_all[col].values.reshape(-1, 1)
    r = LinearRegression().fit(x, y)
    rmse = np.sqrt(np.mean((r.predict(x) - y)**2))
    print(f"  {col}: R²={r.score(x, y):.6f}, RMSE={rmse:.4f}")
