"""Check TVT vs TVT_input relationship in training data for test wells."""
import pandas as pd, numpy as np
from pathlib import Path

DATA_DIR = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")

for wid in ["000d7d20", "00bbac68", "00e12e8b"]:
    hw = pd.read_csv(DATA_DIR / "train" / f"{wid}__horizontal_well.csv")
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    ps = int(mask.idxmax()) if mask.any() else len(hw)

    pre = hw.iloc[:ps]
    tvt_inp_pre = pre["TVT_input"].astype(float)
    tvt_pre     = pre["TVT"].astype(float)
    diff_pre    = (tvt_pre - tvt_inp_pre).abs()

    post = hw.iloc[ps:ps+10]
    tvt_inp_post = post["TVT_input"]
    tvt_post     = post["TVT"].astype(float).values

    print(f"{wid}:")
    print(f"  Pre-PS  TVT vs TVT_input: max_diff={diff_pre.max():.4f} ft  mean_diff={diff_pre.mean():.4f} ft")
    print(f"  Post-PS TVT_input sample: {tvt_inp_post.tolist()}")
    print(f"  Post-PS TVT sample:       {tvt_post.round(2).tolist()}")
    
    # Check if post-PS TVT_input is NaN vs filled
    n_nan_post   = hw["TVT_input"].iloc[ps:].isna().sum()
    n_empty_post = (hw["TVT_input"].iloc[ps:].astype(str).str.strip() == "").sum()
    n_filled_post = len(hw) - ps - n_nan_post - n_empty_post
    print(f"  Post-PS TVT_input: {n_nan_post} NaN, {n_empty_post} empty, {n_filled_post} filled")
    print()
