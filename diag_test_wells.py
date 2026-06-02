"""Deep diagnostic on 3 test wells to understand their trajectories and expected corrections."""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

v8_sub   = pd.read_csv(DATA_DIR / "submissions" / "lgbm_v8_final.csv")
lstm_sub = pd.read_csv(DATA_DIR / "submissions" / "lstm_v2.csv")

for wid in TEST_WELLS:
    hw = pd.read_csv(DATA_DIR / "test" / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_DIR / "test" / f"{wid}__typewell.csv")

    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    ps = int(mask.idxmax()) if mask.any() else len(hw)

    x = hw["X"].values; y = hw["Y"].values
    z = hw["Z"].values; md = hw["MD"].values; gr = hw["GR"].values
    x0, y0, z0 = x[ps], y[ps], z[ps]
    dx = x[ps:] - x0; dy = y[ps:] - y0; dz = z[ps:] - z0
    xy_disp = np.sqrt(dx**2 + dy**2)

    tvt_inp = hw["TVT_input"].astype(str).replace("", "nan").astype(float)
    known = tvt_inp[:ps].values; vm = ~np.isnan(known)
    pre_z = z[:ps][vm]; pre_tvt = known[vm]
    slope = float(LinearRegression().fit(pre_z.reshape(-1,1), pre_tvt).coef_[0]) if len(pre_z) > 5 else -1.0
    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps-1)])
    phys_post = anchor_tvt + slope * dz

    # Typewell interpolation at physics TVT
    tw_tvt = tw["TVT"].astype(float).values
    tw_gr  = pd.Series(tw["GR"]).ffill().bfill().astype(float).values
    gr_post = gr[ps:]
    tw_gr_at_phys = np.interp(phys_post, tw_tvt, tw_gr, left=tw_gr[0], right=tw_gr[-1])
    gr_dev = gr_post - tw_gr_at_phys

    # XY dip fit if trajectory allows
    if xy_disp.max() > 100 and len(dx) > 10:
        X_mat = np.column_stack([dx, dy])
        try:
            reg_dip = LinearRegression().fit(X_mat, phys_post)
            r2_xy = 1 - np.var(phys_post - reg_dip.predict(X_mat)) / (np.var(phys_post) + 1e-9)
        except Exception:
            r2_xy = 0.
    else:
        r2_xy = 0.

    rows_v8   = v8_sub[v8_sub["id"].str.startswith(wid)]["tvt"]
    rows_lstm = lstm_sub[lstm_sub["id"].str.startswith(wid)]["tvt"]
    incl = np.degrees(np.arctan2(xy_disp[-1], abs(dz[-1]) + 1e-9))

    print(f"=== {wid} ===")
    print(f"  Rows: total={len(hw)}  PS={ps}  post={len(hw)-ps}")
    print(f"  Anchor TVT: {anchor_tvt:.1f} ft  Pre-PS slope: {slope:.4f} (TVT/ft-Z)")
    print(f"  Post-PS: dZ={dz[-1]:.1f} ft  dXY={xy_disp[-1]:.1f} ft  dMD={md[ps:].max()-md[ps:].min():.1f} ft")
    print(f"  Inclination from vertical: {incl:.1f} deg  (90=fully horizontal)")
    print(f"  XY direction: dX={x[-1]-x0:.1f} ft  dY={y[-1]-y0:.1f} ft")
    print(f"  Physics correction over post-PS: {phys_post[-1]-anchor_tvt:.1f} ft")
    print(f"  R²_XY (physics vs XY): {r2_xy:.4f}  (>0.9 = dipping)")
    print(f"  GR dev mean (post-PS): {gr_dev.mean():.2f}  std: {gr_dev.std():.2f}")
    print(f"  LGBM v8 TVT: [{rows_v8.min():.1f}, {rows_v8.max():.1f}]  range={rows_v8.max()-rows_v8.min():.1f}")
    print(f"  LSTM v2 TVT: [{rows_lstm.min():.1f}, {rows_lstm.max():.1f}]  range={rows_lstm.max()-rows_lstm.min():.1f}")
    print()
