"""
Spatial neighbor analysis for test wells.
Find training wells whose HORIZONTAL SECTION passes through the same 
geographic area as each test well's horizontal section.
Then extract dip coefficients from those training wells as priors.
"""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

DATA_DIR  = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)

# ----- Gather training well post-PS centroids and dip info -----
print("Loading training well spatial data...")
train_ids = [p.stem.replace("__horizontal_well", "")
             for p in TRAIN_DIR.glob("*__horizontal_well.csv")]

well_data = {}
for wid in train_ids:
    try:
        hw = pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv")
        ps = get_ps(hw)
        if len(hw) - ps < 100:
            continue
        x = hw["X"].values; y = hw["Y"].values; z = hw["Z"].values
        x0, y0, z0 = x[ps], y[ps], z[ps]
        dx = x[ps:] - x0; dy = y[ps:] - y0; dz = z[ps:] - z0
        xy_disp = np.sqrt(dx**2 + dy**2)
        if xy_disp.max() < 200:
            continue  # not really horizontal

        # Pre-PS physics
        tvt_inp = hw["TVT_input"].astype(str).replace("", "nan").astype(float)
        known = tvt_inp[:ps].values; vm = ~np.isnan(known)
        pre_z = z[:ps][vm]; pre_tvt = known[vm]
        if len(pre_z) < 5:
            continue
        slope = float(LinearRegression().fit(pre_z.reshape(-1,1), pre_tvt).coef_[0])
        anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps-1)])
        phys_post = anchor_tvt + slope * dz

        # True correction
        true_tvt = hw["TVT"].astype(float).values[ps:]
        correction = true_tvt - phys_post

        # Dip coefficients: correction ~ alpha_x * dx + alpha_y * dy
        Xmat = np.column_stack([dx, dy])
        reg = LinearRegression(fit_intercept=True).fit(Xmat, correction)
        pred_corr = reg.predict(Xmat)
        r2 = float(1 - np.var(correction - pred_corr) / (np.var(correction) + 1e-9))
        alpha_x, alpha_y = float(reg.coef_[0]), float(reg.coef_[1])

        # Post-PS section centroid
        cx = np.median(x[ps:]); cy = np.median(y[ps:])

        well_data[wid] = {
            "cx": cx, "cy": cy,
            "x0": x0, "y0": y0,
            "alpha_x": alpha_x, "alpha_y": alpha_y, "r2_xy": r2,
            "dx_total": x[-1] - x0, "dy_total": y[-1] - y0,
            "anchor_tvt": anchor_tvt,
            "max_disp": xy_disp.max(),
            "mean_corr": float(correction.mean()),
            "std_corr": float(correction.std()),
        }
    except Exception:
        pass

train_df = pd.DataFrame(well_data).T
print(f"Loaded {len(train_df)} training wells with valid post-PS sections")

# ----- Analyze each test well -----
for wid in TEST_WELLS:
    print(f"\n{'='*60}")
    print(f"TEST WELL: {wid}")
    
    hw = pd.read_csv(TEST_DIR / f"{wid}__horizontal_well.csv")
    ps = get_ps(hw)
    x = hw["X"].values; y = hw["Y"].values; z = hw["Z"].values
    x0, y0, z0 = x[ps], y[ps], z[ps]
    
    # Post-PS centroid
    cx = np.median(x[ps:]); cy = np.median(y[ps:])
    dx_total = x[-1] - x0; dy_total = y[-1] - y0
    
    print(f"  PS anchor: X={x0:.0f}  Y={y0:.0f}")
    print(f"  Post-PS centroid: X={cx:.0f}  Y={cy:.0f}")
    print(f"  Drill direction: dX={dx_total:.0f}  dY={dy_total:.0f}")
    
    # Distance from each training well centroid to test well centroid
    dist_c = np.sqrt((train_df["cx"] - cx)**2 + (train_df["cy"] - cy)**2)
    dist_origin = np.sqrt((train_df["x0"] - x0)**2 + (train_df["y0"] - y0)**2)
    
    print(f"\n  *** Nearest by POST-PS CENTROID ***")
    nearest_c = dist_c.nsmallest(15)
    sub_c = train_df.loc[nearest_c.index].copy()
    sub_c["dist_c"] = nearest_c
    sub_c["dist_origin"] = dist_origin.loc[nearest_c.index]
    cols = ["dist_c", "dist_origin", "alpha_x", "alpha_y", "r2_xy", "mean_corr", "std_corr", "anchor_tvt", "max_disp"]
    print(sub_c[cols].round(3).to_string())
    
    # Weighted average dip from nearby wells
    w = 1.0 / (nearest_c.values[:5] + 1)
    alpha_x_est = np.average(sub_c["alpha_x"].values[:5], weights=w)
    alpha_y_est = np.average(sub_c["alpha_y"].values[:5], weights=w)
    print(f"\n  Weighted avg (top 5): alpha_x={alpha_x_est:.4f}  alpha_y={alpha_y_est:.4f}")
    
    # Estimate correction at end of well
    est_corr = alpha_x_est * dx_total + alpha_y_est * dy_total
    print(f"  Estimated total correction at end: {est_corr:.1f} ft")
    
    print(f"\n  *** Nearest by PS ORIGIN ***")
    nearest_o = dist_origin.nsmallest(10)
    sub_o = train_df.loc[nearest_o.index].copy()
    sub_o["dist_origin"] = nearest_o
    print(sub_o[["dist_origin", "alpha_x", "alpha_y", "r2_xy", "mean_corr", "anchor_tvt"]].round(3).to_string())
