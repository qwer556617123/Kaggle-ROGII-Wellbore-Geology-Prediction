import numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR=Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR=DATA_DIR/"train"; TEST_DIR=DATA_DIR/"test"
MODELS_DIR=DATA_DIR/"models"; SUBS_DIR=DATA_DIR/"submissions"; FEAT_DIR=DATA_DIR/"features"
TEST_WELLS=["000d7d20","00bbac68","00e12e8b"]

def fill_arr(a): return pd.Series(a).ffill().bfill().astype(float).values
def load_well(wid,split="train"):
    d=TRAIN_DIR if split=="train" else TEST_DIR
    return pd.read_csv(d/f"{wid}__horizontal_well.csv"),pd.read_csv(d/f"{wid}__typewell.csv")
def get_ps(hw):
    mask=hw["TVT_input"].isna()|(hw["TVT_input"].astype(str).str.strip()=="")
    return int(mask.idxmax()) if mask.any() else len(hw)

def build(hw,tw,well_id,split="train"):
    n=len(hw); ps=get_ps(hw)
    gr=fill_arr(hw["GR"]); md=hw["MD"].astype(float).values
    z=hw["Z"].astype(float).values; x=hw["X"].astype(float).values; y=hw["Y"].astype(float).values
    tvt_inp=hw["TVT_input"].astype(str).replace("",np.nan).astype(float)
    lkt=tvt_inp.ffill().bfill().values; tw_tvt=tw["TVT"].astype(float).values; tw_gr=fill_arr(tw["GR"]); gr_s=pd.Series(gr)
    if split=="train" and "TVT" in hw.columns:
        pre_z=z[:ps]; pre_tvt=hw["TVT"].astype(float).values[:ps]
    else:
        known=tvt_inp[:ps].values; valid=~np.isnan(known); pre_z=z[:ps][valid]; pre_tvt=known[valid]
    if len(pre_z)>5:
        reg=LinearRegression().fit(pre_z.reshape(-1,1),pre_tvt)
        slope=float(reg.coef_[0]); pre_tvt_pred=reg.predict(pre_z.reshape(-1,1))
        pre_r2=float(1-np.var(pre_tvt-pre_tvt_pred)/np.var(pre_tvt)) if np.var(pre_tvt)>0 else 1.
    else: slope=-1.0; pre_r2=0.
    anchor_tvt=float(tvt_inp.ffill().iloc[max(0,ps-1)]); Z_anchor=z[max(0,ps-1)]
    anchored_physics=anchor_tvt+slope*(z-Z_anchor)
    post_z=z[ps:] if ps<n else z[:]; post_md=md[ps:] if ps<n else md[:]
    n_post=len(post_z); z_end=float(post_z[-1]) if n_post>0 else float(z[-1])
    total_z=z_end-Z_anchor; total_md=(float(post_md[-1])-float(post_md[0])) if n_post>1 else 1.
    post_dz_rate=total_z/total_md; phys_tvt_end=slope*total_z
    pre_dz_md=((z[ps-1]-z[0])/(md[ps-1]-md[0])) if ps>1 and abs(md[ps-1]-md[0])>0.01 else -0.01
    dz_rate_change=post_dz_rate-pre_dz_md
    gr_at_physics=np.interp(anchored_physics,tw_tvt,tw_gr,left=tw_gr[0],right=tw_gr[-1]); gr_dev=gr-gr_at_physics
    feat={"gr":gr,"gr_diff":np.gradient(gr),"gr_diff2":np.gradient(np.gradient(gr))}
    for w in [5,11,21,51,101]:
        roll=gr_s.rolling(w,center=True,min_periods=1)
        feat[f"gr_mean_{w}"]=roll.mean().values; feat[f"gr_std_{w}"]=roll.std().fillna(0).values; feat[f"gr_range_{w}"]=(roll.max()-roll.min()).values
    feat.update({"md":md,"z":z,"dz_dmd":np.gradient(z,md),"dx_dmd":np.gradient(x,md),"dy_dmd":np.gradient(y,md),
        "inclination":np.arctan2(np.sqrt(np.gradient(x)**2+np.gradient(y)**2),np.abs(np.gradient(z))+1e-9)*180/np.pi,
        "raw_dz":np.diff(z,prepend=z[0]),"raw_dmd":np.diff(md,prepend=md[0]),
        "last_known_tvt":lkt,"rows_since_ps":np.maximum(0,np.arange(n)-ps).astype(float),
        "rows_before_ps":np.maximum(0,ps-np.arange(n)).astype(float),"ps_idx":float(ps),
        "tvt_z_slope":np.full(n,slope),"physics_tvt":anchored_physics,"physics_vs_lkt":anchored_physics-lkt,
        "physics_dtvt":slope*np.diff(z,prepend=z[0]),"gr_at_physics":gr_at_physics,"gr_dev_physics":gr_dev,
        "total_z_change_post":np.full(n,total_z),"physics_tvt_at_end":np.full(n,phys_tvt_end),
        "post_dz_rate":np.full(n,post_dz_rate),"dz_rate_change":np.full(n,dz_rate_change),
        "n_post_ps":np.full(n,float(n_post)),
        "post_ps_frac":np.where(n_post>0,np.maximum(0,np.arange(n)-ps)/n_post,0.).astype(float),
        "pre_r2":np.full(n,pre_r2)})
    if ps>2:
        pre_dz=np.gradient(z[:ps],md[:ps]) if ps>1 else np.array([0.])
        vtvt=tvt_inp[:ps].values; vtvt=vtvt[~np.isnan(vtvt)]; pdtvt=np.diff(vtvt) if len(vtvt)>1 else np.array([0.])
        feat["pre_ps_dz_slope"]=np.full(n,pre_dz[-min(50,len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"]=np.full(n,pdtvt[-min(50,len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]=np.full(n,(pdtvt[-min(50,len(pdtvt)):].std() if len(pdtvt)>1 else 0.))
    else:
        for k in ["pre_ps_dz_slope","pre_ps_dtvt_mean","pre_ps_dtvt_std"]: feat[k]=np.zeros(n)
    df=pd.DataFrame(feat); df["well_id"]=well_id; df["is_post_ps"]=(np.arange(n)>=ps).astype(int)
    if split=="train" and "TVT" in hw.columns:
        tvt_true=hw["TVT"].astype(float).values
        df["target_correction"]=tvt_true-anchored_physics; df["anchored_physics_col"]=anchored_physics
    return df

train_ids=pd.read_csv(FEAT_DIR/"train_ids.csv",header=None)[0].tolist()
val_ids=pd.read_csv(FEAT_DIR/"val_ids.csv",header=None)[0].tolist()

tr_dfs=[]; vl_dfs=[]
for i,wid in enumerate(train_ids):
    try: tr_dfs.append(build(*load_well(wid),wid,"train"))
    except: pass
    if(i+1)%150==0: print(f"  train {i+1}/{len(train_ids)}")
for wid in val_ids:
    try: vl_dfs.append(build(*load_well(wid),wid,"train"))
    except: pass
tr_all=pd.concat(tr_dfs,ignore_index=True); vl_all=pd.concat(vl_dfs,ignore_index=True)

EXCL={"well_id","target_correction","target_tvt","is_post_ps","anchored_physics_col"}
FCOLS=[c for c in tr_all.columns if c not in EXCL]
tr_p=tr_all[tr_all["is_post_ps"]==1]; vl_p=vl_all[vl_all["is_post_ps"]==1]
X_tr=tr_p[FCOLS].values.astype(np.float32); y_tr=tr_p["target_correction"].values.astype(np.float32)
X_vl=vl_p[FCOLS].values.astype(np.float32); y_vl=vl_p["target_correction"].values.astype(np.float32)

results={}
for tag,params in [
    ("v13_huber", {"objective":"huber","alpha":10.,"metric":"rmse","num_leaves":255,"learning_rate":0.02,
                   "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,
                   "min_child_samples":20,"lambda_l1":0.05,"lambda_l2":0.05,"verbose":-1,"seed":42,"n_jobs":-1}),
    ("v13_reg",   {"objective":"regression","metric":"rmse","num_leaves":255,"learning_rate":0.02,
                   "feature_fraction":0.7,"bagging_fraction":0.7,"bagging_freq":5,
                   "min_child_samples":50,"lambda_l1":0.3,"lambda_l2":0.3,"verbose":-1,"seed":42,"n_jobs":-1}),
]:
    print(f"\nTraining {tag}...")
    lgb_tr=lgb.Dataset(X_tr,label=y_tr,feature_name=FCOLS)
    lgb_vl=lgb.Dataset(X_vl,label=y_vl,feature_name=FCOLS,reference=lgb_tr)
    m=lgb.train(params,lgb_tr,num_boost_round=15000,valid_sets=[lgb_vl],
        callbacks=[lgb.early_stopping(500,verbose=False),lgb.log_evaluation(3000)])
    print(f"  best_iter={m.best_iteration}, val_rmse={m.best_score['valid_0']['rmse']:.4f}")
    rmses=[]
    for wid in val_ids:
        try:
            hw,tw=load_well(wid); ps=get_ps(hw)
            df=build(hw,tw,wid,"train"); df_p=df[df["is_post_ps"]==1]
            corr=m.predict(df_p[FCOLS].values.astype(np.float32))
            tvt_pred=gaussian_filter1d(df_p["anchored_physics_col"].values+corr,sigma=1.0)
            tvt_true=hw["TVT"].astype(float).values[ps:]
            rmses.append(np.sqrt(np.mean((tvt_pred[:len(tvt_true)]-tvt_true[:len(tvt_pred)])**2)))
        except: pass
    rmses=np.array(rmses)
    results[tag]={"mean":rmses.mean(),"iter":m.best_iteration,"row_rmse":m.best_score["valid_0"]["rmse"]}
    print(f"  per-well mean={rmses.mean():.3f}  median={np.median(rmses):.3f}")
    if rmses.mean()<11.983:
        joblib.dump(m,MODELS_DIR/f"lgbm_{tag}.pkl")
        print(f"  ** BETTER than v8! Saved.")
    
print("\nSummary:")
for k,v in results.items():
    print(f"  {k}: per-well={v['mean']:.3f}  row_rmse={v['row_rmse']:.4f}  iter={v['iter']}")
print(f"  v8:      per-well=11.983  row_rmse=15.17xx  iter=7983")
