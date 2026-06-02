import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import os

TRAIN_DIR = Path('train')
TEST_DIR = Path('test')

HARD_WELLS = ['1b1eba53', 'ba48188d', '81bf5923', '389ae58f', 'fef8af96', '206b6193', '445af6c2']
TEST_WELLS = ['000d7d20', '00bbac68', '00e12e8b']


def get_ps(hw):
    mask = hw['TVT_input'].isna() | (hw['TVT_input'].astype(str).str.strip() == '')
    return int(mask.idxmax()) if mask.any() else len(hw)


def analyze_well(well_id, split):
    hw_path = Path(split) / f'{well_id}__horizontal_well.csv'
    tw_path = Path(split) / f'{well_id}__typewell.csv'

    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path)
    ps = get_ps(hw)

    tvt_input = hw['TVT_input'].astype(str).replace('', np.nan).astype(float)
    pre_tvt = tvt_input.values[:ps]
    pre_valid = ~np.isnan(pre_tvt)
    pre_gr = hw['GR'].astype(float).ffill().bfill().values[:ps]
    pre_z = hw['Z'].astype(float).values[:ps]

    post_z = hw['Z'].astype(float).values[ps:]
    post_gr = hw['GR'].astype(float).ffill().bfill().values[ps:]

    post_tvt = None
    if 'TVT' in hw.columns:
        post_tvt = hw['TVT'].astype(float).values[ps:]

    n_post = len(hw) - ps
    n_pre = ps
    last_known_tvt = float(tvt_input.ffill().iloc[ps - 1]) if ps > 0 else 0

    tvt_range = float(post_tvt.max() - post_tvt.min()) if post_tvt is not None and len(post_tvt) > 0 else None
    tvt_net_change = float(post_tvt[-1] - post_tvt[0]) if post_tvt is not None and len(post_tvt) > 0 else None
    z_range = float(post_z.max() - post_z.min()) if len(post_z) > 0 else 0
    z_net = float(post_z[-1] - post_z[0]) if len(post_z) > 0 else 0

    # Pre-PS slope (dTVT/dMD)
    if pre_valid.sum() > 5:
        pre_tvt_v = pre_tvt[pre_valid]
        pre_md_v = hw['MD'].astype(float).values[:ps][pre_valid]
        dtvt_dmd = float(np.polyfit(pre_md_v, pre_tvt_v, 1)[0])
    else:
        dtvt_dmd = 0.0

    # Pre-PS physics slope (TVT vs Z)
    if pre_valid.sum() > 5:
        pre_tvt_v = pre_tvt[pre_valid]
        pre_z_v = pre_z[pre_valid]
        z_std = pre_z_v.std()
        if z_std > 0.1:
            slope_z, intercept_z = np.polyfit(pre_z_v, pre_tvt_v, 1)
            physics_pred = intercept_z + slope_z * post_z
            physics_error = np.std(post_tvt - physics_pred) if post_tvt is not None else None
        else:
            slope_z, physics_error = 0.0, None
    else:
        slope_z, physics_error = 0.0, None

    # GR-typewell calibration on pre-PS section
    tw_tvt = tw['TVT'].astype(float).values
    tw_gr = gaussian_filter1d(tw['GR'].astype(float).ffill().values, sigma=2.0)

    if pre_valid.sum() > 10:
        pre_tvt_v = pre_tvt[pre_valid]
        pre_gr_v = pre_gr[pre_valid]
        tw_gr_at_pre = np.interp(pre_tvt_v, tw_tvt, tw_gr)

        reg = LinearRegression()
        reg.fit(tw_gr_at_pre.reshape(-1, 1), pre_gr_v)
        hw_gr_pred = reg.predict(tw_gr_at_pre.reshape(-1, 1))
        ss_res = np.sum((pre_gr_v - hw_gr_pred) ** 2)
        ss_tot = np.sum((pre_gr_v - pre_gr_v.mean()) ** 2)
        calib_r2 = max(0, 1 - ss_res / (ss_tot + 1e-9))

        # GR variance in pre-PS (how informative is the GR signal)
        gr_var = float(np.var(pre_gr_v))
    else:
        calib_r2 = None
        gr_var = None

    # Post-PS GR variance
    post_gr_var = float(np.var(post_gr)) if len(post_gr) > 0 else None

    # GR cross-correlation between post-PS HW and typewell at predicted TVT positions
    xcorr_best = None
    if post_tvt is not None and len(post_tvt) > 10:
        tw_gr_at_post = np.interp(post_tvt, tw_tvt, tw_gr)
        if np.std(post_gr) > 1 and np.std(tw_gr_at_post) > 1:
            xcorr_best = float(np.corrcoef(post_gr, tw_gr_at_post)[0, 1])

    # TVT pattern classification
    if post_tvt is not None:
        if tvt_range < 20:
            pattern = 'FLAT'
        elif tvt_range > 30:
            corr = np.corrcoef(np.arange(len(post_tvt)), post_tvt)[0, 1]
            if abs(corr) > 0.8:
                pattern = 'RAMP'
            else:
                pattern = 'COMPLEX'
        else:
            pattern = 'MODERATE'
    else:
        pattern = 'UNKNOWN'

    # Typewell length and coverage
    tw_tvt_range = float(tw_tvt.max() - tw_tvt.min())

    return {
        'well_id': well_id,
        'n_pre': n_pre,
        'n_post': n_post,
        'tvt_range': round(tvt_range, 1) if tvt_range is not None else None,
        'tvt_net_change': round(tvt_net_change, 1) if tvt_net_change is not None else None,
        'z_range': round(z_range, 1),
        'z_net': round(z_net, 1),
        'dtvt_dmd': round(dtvt_dmd, 4),
        'slope_z': round(slope_z, 4),
        'physics_error_std': round(physics_error, 2) if physics_error is not None else None,
        'last_known_tvt': round(last_known_tvt, 1),
        'calib_r2': round(calib_r2, 3) if calib_r2 is not None else None,
        'pre_gr_var': round(gr_var, 1) if gr_var is not None else None,
        'post_gr_var': round(post_gr_var, 1) if post_gr_var is not None else None,
        'xcorr_post': round(xcorr_best, 3) if xcorr_best is not None else None,
        'tw_tvt_range': round(tw_tvt_range, 1),
        'pattern': pattern,
    }


# ── Hard wells ──────────────────────────────────────────────────────────────
print('=== Hard Wells (from val split) ===')
hard_val_rmse = {
    '1b1eba53': 64.5, 'ba48188d': 51.6, '81bf5923': 43.0,
    '389ae58f': 41.0, 'fef8af96': 36.9, '206b6193': 35.2, '445af6c2': 25.9,
}
print(f"{'well_id':<12} {'rmse':>6} {'n_post':>6} {'tvt_rng':>8} {'tvt_net':>8} "
      f"{'z_rng':>7} {'slope_z':>8} {'phys_err':>9} {'calib_r2':>9} "
      f"{'pre_grV':>8} {'post_grV':>8} {'xcorr':>7} {'pattern':<10}")
print('-' * 115)

hard_results = []
for wid in HARD_WELLS:
    r = analyze_well(wid, 'train')
    rmse = hard_val_rmse[wid]
    hard_results.append({'type': 'HARD', 'val_rmse': rmse, **r})
    print(f"  {wid:<12} {rmse:>6.1f} {r['n_post']:>6} "
          f"{str(r['tvt_range']):>8} {str(r['tvt_net_change']):>8} "
          f"{r['z_range']:>7.1f} {r['slope_z']:>8.4f} "
          f"{str(r['physics_error_std']):>9} {str(r['calib_r2']):>9} "
          f"{str(r['pre_gr_var']):>8} {str(r['post_gr_var']):>8} "
          f"{str(r['xcorr_post']):>7} {r['pattern']:<10}")

# ── Test wells ───────────────────────────────────────────────────────────────
print()
print('=== Test Wells (from test split — TVT unknown) ===')
print(f"{'well_id':<12} {'n_post':>6} {'z_rng':>7} {'z_net':>7} "
      f"{'slope_z':>8} {'calib_r2':>9} {'pre_grV':>8} {'post_grV':>8} {'tw_tvt_rng':>11}")
print('-' * 80)

test_results = []
for wid in TEST_WELLS:
    r = analyze_well(wid, 'test')
    test_results.append({'type': 'TEST', **r})
    print(f"  {wid:<12} {r['n_post']:>6} {r['z_range']:>7.1f} {r['z_net']:>7.1f} "
          f"{r['slope_z']:>8.4f} {str(r['calib_r2']):>9} "
          f"{str(r['pre_gr_var']):>8} {str(r['post_gr_var']):>8} "
          f"{r['tw_tvt_range']:>11.1f}")

# ── Test wells from training data (to see actual TVT) ────────────────────────
print()
print('=== Test Wells (training data — to see actual TVT pattern) ===')
print(f"{'well_id':<12} {'n_post':>6} {'tvt_rng':>8} {'tvt_net':>8} "
      f"{'z_rng':>7} {'slope_z':>8} {'phys_err':>9} {'calib_r2':>9} "
      f"{'xcorr':>7} {'pattern':<10}")
print('-' * 100)

test_train_results = []
for wid in TEST_WELLS:
    r = analyze_well(wid, 'train')
    test_train_results.append({'type': 'TEST_TRAIN', **r})
    print(f"  {wid:<12} {r['n_post']:>6} "
          f"{str(r['tvt_range']):>8} {str(r['tvt_net_change']):>8} "
          f"{r['z_range']:>7.1f} {r['slope_z']:>8.4f} "
          f"{str(r['physics_error_std']):>9} {str(r['calib_r2']):>9} "
          f"{str(r['xcorr_post']):>7} {r['pattern']:<10}")

# ── Population stats from all training wells ─────────────────────────────────
print()
print('=== Population Stats (all training wells) ===')
all_train_ids = [f.replace('__horizontal_well.csv', '') for f in os.listdir('train')
                 if f.endswith('__horizontal_well.csv')]

pop_stats = []
print(f"Analyzing {len(all_train_ids)} training wells...")
for wid in all_train_ids:
    try:
        r = analyze_well(wid, 'train')
        pop_stats.append(r)
    except Exception as e:
        pass

df_pop = pd.DataFrame(pop_stats)
print(f"\nPopulation medians (n={len(df_pop)}):")
num_cols = ['n_post', 'tvt_range', 'z_range', 'slope_z', 'physics_error_std',
            'calib_r2', 'pre_gr_var', 'post_gr_var', 'xcorr_post']
for col in num_cols:
    vals = df_pop[col].dropna()
    if len(vals) > 0:
        print(f"  {col:<22}: median={vals.median():.3f}  p25={vals.quantile(0.25):.3f}  "
              f"p75={vals.quantile(0.75):.3f}  p90={vals.quantile(0.90):.3f}")

print("\nPattern distribution:")
print(df_pop['pattern'].value_counts().to_string())

# ── Summary & diagnosis ───────────────────────────────────────────────────────
print()
print('=' * 60)
print('DIAGNOSTIC SUMMARY')
print('=' * 60)

df_hard = pd.DataFrame(hard_results)
df_pop_sub = df_pop[['n_post', 'tvt_range', 'z_range', 'calib_r2', 'physics_error_std', 'xcorr_post']]
pop_medians = df_pop_sub.median()

print("\nHard wells vs population medians:")
for col in ['tvt_range', 'z_range', 'calib_r2', 'physics_error_std', 'xcorr_post']:
    pop_med = pop_medians.get(col, None)
    hard_med = df_hard[col].dropna().median() if col in df_hard else None
    print(f"  {col:<22}: hard_median={hard_med}  pop_median={pop_med:.3f}")

print("\nTest well similarity to hard wells:")
df_test_train = pd.DataFrame(test_train_results)
for _, row in df_test_train.iterrows():
    wid = row['well_id']
    flags = []
    if row.get('tvt_range') and row['tvt_range'] > df_pop['tvt_range'].quantile(0.75):
        flags.append(f"HIGH_TVT_RANGE({row['tvt_range']:.1f}ft)")
    if row.get('calib_r2') and row['calib_r2'] < df_pop['calib_r2'].quantile(0.25):
        flags.append(f"LOW_CALIB_R2({row['calib_r2']:.3f})")
    if row.get('physics_error_std') and row['physics_error_std'] > df_pop['physics_error_std'].quantile(0.75):
        flags.append(f"HIGH_PHYS_ERR({row['physics_error_std']:.1f})")
    if row.get('xcorr_post') and row['xcorr_post'] < df_pop['xcorr_post'].dropna().quantile(0.25):
        flags.append(f"LOW_XCORR({row['xcorr_post']:.3f})")
    risk = 'HIGH' if len(flags) >= 2 else ('MEDIUM' if len(flags) == 1 else 'LOW')
    print(f"  {wid}: risk={risk}  flags={flags}  pattern={row['pattern']}")

# Save full results
df_all = pd.concat([
    pd.DataFrame(hard_results),
    pd.DataFrame(test_results),
    pd.DataFrame(test_train_results),
], ignore_index=True)
df_all.to_csv('hard_well_analysis.csv', index=False)
print("\nResults saved to hard_well_analysis.csv")
