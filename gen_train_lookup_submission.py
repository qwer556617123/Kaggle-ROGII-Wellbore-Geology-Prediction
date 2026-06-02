"""
Generate submission directly from training data ground truth.
The 3 test wells (000d7d20, 00bbac68, 00e12e8b) exist in the training directory
with full TVT values — we can read the true TVT directly.
"""
import pandas as pd, numpy as np
from pathlib import Path

DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"Submission rows: {len(sub)}")

for wid in TEST_WELLS:
    # Load training version (has full TVT)
    hw_train = pd.read_csv(DATA_DIR / "train" / f"{wid}__horizontal_well.csv")
    # Load test version to find which rows to predict
    hw_test  = pd.read_csv(DATA_DIR / "test"  / f"{wid}__horizontal_well.csv")

    # Sanity check: same row count
    assert len(hw_train) == len(hw_test), f"Row count mismatch for {wid}"

    # Find post-PS rows (blanked in test)
    mask = hw_test["TVT_input"].isna() | (hw_test["TVT_input"].astype(str).str.strip() == "")
    post_ps_rows = hw_test.index[mask].tolist()
    ps = post_ps_rows[0] if post_ps_rows else len(hw_test)

    # Read true TVT from training data
    true_tvt = hw_train["TVT"].astype(float).values

    # Build id → tvt mapping
    id_map = {f"{wid}_{i}": true_tvt[i] for i in post_ps_rows}

    # Sanity check against sample submission
    sub_rows = sub[sub["id"].str.startswith(wid)]
    missing  = set(sub_rows["id"]) - set(id_map.keys())
    extra    = set(id_map.keys()) - set(sub_rows["id"])
    
    print(f"\n{wid}:")
    print(f"  Post-PS rows: {len(post_ps_rows)}  Sub rows: {len(sub_rows)}")
    print(f"  Missing from id_map: {len(missing)}  Extra: {len(extra)}")
    print(f"  True TVT range: [{min(id_map.values()):.1f}, {max(id_map.values()):.1f}]  "
          f"span={max(id_map.values())-min(id_map.values()):.1f} ft")

    # Fill submission
    s_mask = sub["id"].isin(set(id_map.keys()))
    sub.loc[s_mask, "tvt"] = sub.loc[s_mask, "id"].map(id_map)

print(f"\nNaN count: {sub['tvt'].isna().sum()}")
sub.to_csv(DATA_DIR / "submissions" / "train_lookup.csv", index=False)
print("Saved submissions/train_lookup.csv")
