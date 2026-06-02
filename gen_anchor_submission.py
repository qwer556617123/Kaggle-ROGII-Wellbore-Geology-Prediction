"""
Generate pure anchor submission — every post-PS row = anchor TVT (constant).
This is our baseline to compare against on Kaggle LB.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TEST  = DATA / "test"
TRAIN = DATA / "train"

sample = pd.read_csv(DATA / "sample_submission.csv")
print(f"Total submission rows: {len(sample)}")

def get_ps(hw):
    mask = hw["TVT_input"].isna() | hw["TVT_input"].astype(str).str.strip().eq("")
    return int(mask.idxmax()) if mask.any() else len(hw)

preds = {}
for wid in ["000d7d20", "00bbac68", "00e12e8b"]:
    hw = pd.read_csv(TEST / f"{wid}__horizontal_well.csv")
    ti = hw["TVT_input"].astype(str).replace("", float("nan")).astype(float)
    ps = get_ps(hw)
    anchor = float(ti.ffill().iloc[ps - 1])
    print(f"{wid}: anchor={anchor:.2f}, n_post={len(hw)-ps}")
    for i in range(len(hw)):
        preds[f"{wid}_{i}"] = anchor if i >= ps else float("nan")

rows = []
for _, row in sample.iterrows():
    sid = row["id"]
    if sid in preds and not np.isnan(preds[sid]):
        rows.append({"id": sid, "tvt": preds[sid]})

sub = pd.DataFrame(rows)
assert len(sub) == len(sample), f"Row count mismatch: {len(sub)} vs {len(sample)}"
out = DATA / "submissions" / "anchor_constant.csv"
sub.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(sub.head())
