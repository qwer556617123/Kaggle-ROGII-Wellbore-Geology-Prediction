---
name: wellbore-eda
description: "Exploratory data analysis for ROGII Wellbore Geology Prediction competition. Use when exploring data, visualizing GR logs, analyzing TVT distribution, checking data quality, understanding typewell-horizontal well relationships, or investigating geological formations."
argument-hint: "Optional: specific well ID or analysis focus (e.g. 'all test wells' or '000d7d20')"
---

# Wellbore EDA Skill

## When to Use
- Exploring the train/test dataset structure
- Visualizing GR log profiles for horizontal wells and typewells
- Analyzing TVT prediction difficulty (how much TVT varies after PS)
- Checking for NaN values in GR, TVT_input
- Comparing GR signatures between horizontal well and typewell
- Understanding geological formation distribution across training wells
- Investigating offset well relationships (proximity, similar dip)

## Procedure

1. **Load all well metadata** — scan train/test directories, build a metadata DataFrame (well_id, n_rows, ps_row, tvt_range, typewell_rows)
2. **Data quality check** — count NaN GR values, check TVT_input gap sizes
3. **GR correlation check** — for a sample well, overlay horizontal GR (projected onto TVT) vs typewell GR to visualize correlation quality
4. **TVT statistics** — analyze dTVT distribution (rate of change after PS point)
5. **Formation analysis** — count geology formation occurrences in train typewells

## Key File Paths
```python
DATA_DIR = r"E:\Code\ROGII - Wellbore Geology Prediction"
TRAIN_DIR = DATA_DIR + r"\train"
TEST_DIR  = DATA_DIR + r"\test"
SUB_FILE  = DATA_DIR + r"\sample_submission.csv"
```

## Loading Helper
```python
import pandas as pd
from pathlib import Path

def load_well(well_id: str, split: str = "train"):
    base = Path(DATA_DIR) / split
    hw = pd.read_csv(base / f"{well_id}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{well_id}__typewell.csv")
    return hw, tw

# PS row = first row where TVT_input is NaN/empty
def get_ps_index(hw: pd.DataFrame) -> int:
    mask = hw["TVT_input"].isna() | (hw["TVT_input"] == "")
    idx = mask.idxmax()
    return idx if mask.any() else len(hw)
```

## GR Correlation Visualization
```python
import matplotlib.pyplot as plt

def plot_gr_correlation(well_id, split="train"):
    hw, tw = load_well(well_id, split)
    ps = get_ps_index(hw)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Left: Typewell GR vs TVT
    axes[0].plot(tw["GR"], tw["TVT"], color="black", label="Typewell GR")
    axes[0].invert_yaxis()
    axes[0].set_title("Typewell GR")
    
    # Right: Horizontal well GR projected on TVT_input (known portion only)
    known = hw.iloc[:ps]
    axes[1].plot(known["GR"], known["TVT_input"].astype(float), color="green", alpha=0.7, label="Pre-PS GR")
    if split == "train":
        post = hw.iloc[ps:]
        axes[1].plot(post["GR"], post["TVT"].astype(float), color="red", alpha=0.5, label="Post-PS GR (truth)")
    axes[1].invert_yaxis()
    axes[1].set_title(f"Horizontal Well GR — {well_id}")
    plt.tight_layout()
    plt.savefig(f"eda_{well_id}.png", dpi=100)
    plt.show()
```

## Important EDA Findings
- The 3 test wells are: `000d7d20`, `00bbac68`, `00e12e8b`
- TVT can increase, decrease, or stay constant after PS
- Pre-PS GR from horizontal well has higher resolution than typewell GR
- Neighboring wells (by X,Y coordinates) share geological dip patterns
- See [data reference](./references/data-schema.md) for column definitions
