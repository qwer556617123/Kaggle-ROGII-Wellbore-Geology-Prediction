import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
import lightgbm as lgb, joblib, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR=Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR=DATA_DIR/"train"; MODELS_DIR=DATA_DIR/"models"; FEAT_DIR=DATA_DIR/"features"; SUBS_DIR=DATA_DIR/"submissions"
CTX_LEN=200; MAX_POST=5000; N_FEAT=10; HIDDEN=128; N_LAYERS=2

def fill_arr(a): return pd.Series(a).ffill().bfill().astype(float).values
def load_well(wid,split="train"):
    d=TRAIN_DIR if split=="train" else DATA_DIR/"test"
    return pd.read_csv(d/f"{wid}__horizontal_well.csv"),pd.read_csv(d/f"{wid}__typewell.csv")
def get_ps(hw):
    mask=hw["TVT_input"].isna()|(hw["TVT_input"].astype(str).str.strip()=="")
    return int(mask.idxmax()) if mask.any() else len(hw)

def extract_features(hw,tw,split="train"):
    n=len(hw); ps=get_ps(hw)
    gr=fill_arr(hw["GR"]); md=hw["MD"].astype(float).values
    z=hw["Z"].astype(float).values
    tvt_inp=hw["TVT_input"].astype(str).replace("",np.nan).astype(float)
    lkt=tvt_inp.ffill().bfill().values
    if split=="train" and "TVT" in hw.columns:
        pre_z_fit=z[:ps]; pre_tvt=hw["TVT"].astype(float).values[:ps]
    else:
        known=tvt_inp[:ps].values; valid=~np.isnan(known)
        pre_z_fit=z[:ps][valid]; pre_tvt=known[valid]
    if len(pre_z_fit)>5:
        from sklearn.linear_model import LinearRegression
        reg=LinearRegression().fit(pre_z_fit.reshape(-1,1),pre_tvt)
        slope=float(reg.coef_[0])
    else: slope=-1.0
    anchor_tvt=float(tvt_inp.ffill().iloc[max(0,ps-1)])
    Z_anchor=z[max(0,ps-1)]
    anchored_physics=anchor_tvt+slope*(z-Z_anchor)
    post_z=z[ps:] if ps<n else z[:]
    post_md=md[ps:] if ps<n else md[:]
    n_post=len(post_z)
    z_end=float(post_z[-1]) if n_post>0 else float(z[-1])
    total_z=z_end-Z_anchor
    total_md=(float(post_md[-1])-float(post_md[0])) if n_post>1 else 1.
    post_dz_rate=total_z/total_md; phys_at_end=slope*total_z
    gr_mean=np.mean(gr[:ps]) if ps>0 else np.mean(gr); gr_std=np.std(gr[:ps])+1e-6
    gr_norm=(gr-gr_mean)/gr_std; dz_dmd=np.gradient(z,md)
    phys_vs_lkt=anchored_physics-lkt; phys_dtvt=slope*np.diff(z,prepend=z[0])
    progress=np.maximum(0,np.arange(n)-ps).astype(float)/max(n_post,1)
    phys_end_col=np.full(n,phys_at_end/100.); slope_col=np.full(n,slope)
    post_dz_col=np.full(n,post_dz_rate); is_post=(np.arange(n)>=ps).astype(float)
    feats=np.stack([gr_norm,dz_dmd,phys_vs_lkt,phys_dtvt,progress,phys_end_col,slope_col,post_dz_col,is_post,lkt/1e4],axis=1).astype(np.float32)
    ctx_start=max(0,ps-CTX_LEN); ctx_feat=feats[ctx_start:ps]; post_feat=feats[ps:]
    if split=="train" and "TVT" in hw.columns:
        tvt_true=hw["TVT"].astype(float).values
        correction=(tvt_true-anchored_physics)[ps:]
    else: correction=None
    return ctx_feat,post_feat,correction,anchored_physics[ps:]

class WellLSTM(nn.Module):
    def __init__(self,input_dim=N_FEAT,hidden_dim=HIDDEN,n_layers=N_LAYERS,dropout=0.15):
        super().__init__()
        self.hidden_dim=hidden_dim; self.n_layers=n_layers
        self.encoder=nn.LSTM(input_dim,hidden_dim//2,n_layers,batch_first=True,bidirectional=True,dropout=dropout if n_layers>1 else 0.)
        self.decoder=nn.LSTM(input_dim,hidden_dim,n_layers,batch_first=True,bidirectional=False,dropout=dropout if n_layers>1 else 0.)
        self.h_proj=nn.Linear(hidden_dim*n_layers,hidden_dim*n_layers)
        self.c_proj=nn.Linear(hidden_dim*n_layers,hidden_dim*n_layers)
        self.output=nn.Sequential(nn.Linear(hidden_dim,64),nn.ReLU(),nn.Linear(64,1))
    def encode_ctx(self,ctx_seq,ctx_lens):
        B=ctx_seq.size(0)
        packed=pack_padded_sequence(ctx_seq,ctx_lens.cpu(),batch_first=True,enforce_sorted=False)
        _,(h_n,c_n)=self.encoder(packed)
        h_n=h_n.view(self.n_layers,2,B,self.hidden_dim//2)
        c_n=c_n.view(self.n_layers,2,B,self.hidden_dim//2)
        h_n=torch.cat([h_n[:,0],h_n[:,1]],dim=-1); c_n=torch.cat([c_n[:,0],c_n[:,1]],dim=-1)
        h_flat=h_n.permute(1,0,2).reshape(B,-1); c_flat=c_n.permute(1,0,2).reshape(B,-1)
        h_out=self.h_proj(h_flat).reshape(B,self.n_layers,self.hidden_dim).permute(1,0,2).contiguous()
        c_out=self.c_proj(c_flat).reshape(B,self.n_layers,self.hidden_dim).permute(1,0,2).contiguous()
        return h_out,c_out
    def decode_post(self,post_seq,post_lens,h0,c0):
        packed=pack_padded_sequence(post_seq,post_lens.cpu(),batch_first=True,enforce_sorted=False)
        out,_=self.decoder(packed,(h0,c0))
        out_pad,_=pad_packed_sequence(out,batch_first=True)
        return self.output(out_pad).squeeze(-1)
    def forward(self,ctx_seq,post_seq,ctx_lens,post_lens):
        h0,c0=self.encode_ctx(ctx_seq,ctx_lens); return self.decode_post(post_seq,post_lens,h0,c0)

# Load LSTM
model=WellLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODELS_DIR/"lstm_best.pt",map_location=DEVICE))
model.eval()

# Load LGBM
lgbm_model=joblib.load(MODELS_DIR/"lgbm_final.pkl")

# Load v8 LGBM for val comparison (trained on 658 wells only)
lgbm_v8=joblib.load(MODELS_DIR/"lgbm_v8.pkl")
feat_imp=pd.read_csv(MODELS_DIR/"lgbm_v8_importance.csv")
FCOLS=feat_imp["feature"].tolist() if "feature" in feat_imp.columns else lgbm_v8.feature_name()

val_ids=pd.read_csv(FEAT_DIR/"val_ids.csv",header=None)[0].tolist()

def build_lgbm_features(hw,tw,split="train"):
    n=len(hw); ps=get_ps(hw)
    gr=fill_arr(hw["GR"]); md=hw["MD"].astype(float).values
    z=hw["Z"].astype(float).values; x=hw["X"].astype(float).values; y=hw["Y"].astype(float).values
    tvt_inp=hw["TVT_input"].astype(str).replace("",np.nan).astype(float)
    lkt=tvt_inp.ffill().bfill().values
    tw_tvt=tw["TVT"].astype(float).values; tw_gr=fill_arr(tw["GR"])
    if split=="train" and "TVT" in hw.columns:
        pre_z=z[:ps]; pre_tvt=hw["TVT"].astype(float).values[:ps]
    else:
        known=tvt_inp[:ps].values; valid=~np.isnan(known); pre_z=z[:ps][valid]; pre_tvt=known[valid]
    if len(pre_z)>5:
        reg=LinearRegression().fit(pre_z.reshape(-1,1),pre_tvt)
        slope=float(reg.coef_[0]); intercept=float(reg.intercept_)
        pre_tvt_pred=reg.predict(pre_z.reshape(-1,1))
        pre_r2=float(1-np.var(pre_tvt-pre_tvt_pred)/np.var(pre_tvt)) if np.var(pre_tvt)>0 else 1.
    else:
        slope=-1.0; intercept=float(lkt[0] if not np.isnan(lkt[0]) else 11000); pre_r2=0.
    anchor_tvt=float(tvt_inp.ffill().iloc[max(0,ps-1)])
    Z_anchor=z[max(0,ps-1)]
    anchored_physics=anchor_tvt+slope*(z-Z_anchor)
    post_z=z[ps:] if ps<n else z[:]; post_md=md[ps:] if ps<n else md[:]
    n_post=len(post_z); z_end=float(post_z[-1]) if n_post>0 else float(z[-1])
    total_z=z_end-Z_anchor; total_md=(float(post_md[-1])-float(post_md[0])) if n_post>1 else 1.
    post_dz_rate=total_z/total_md; physics_tvt_at_end=slope*total_z
    pre_dz_md=((z[ps-1]-z[0])/(md[ps-1]-md[0])) if ps>1 and abs(md[ps-1]-md[0])>0.01 else -0.01
    dz_rate_change=post_dz_rate-pre_dz_md
    gr_at_physics=np.interp(anchored_physics,tw_tvt,tw_gr,left=tw_gr[0],right=tw_gr[-1])
    gr_dev=gr-gr_at_physics
    gr_s=pd.Series(gr)
    feat={"gr":gr,"gr_diff":np.gradient(gr),"gr_diff2":np.gradient(np.gradient(gr))}
    for w in [5,11,21,51,101]:
        roll=gr_s.rolling(w,center=True,min_periods=1)
        feat[f"gr_mean_{w}"]=roll.mean().values; feat[f"gr_std_{w}"]=roll.std().fillna(0).values
        feat[f"gr_range_{w}"]=(roll.max()-roll.min()).values
    feat.update({"md":md,"z":z,"dz_dmd":np.gradient(z,md),"dx_dmd":np.gradient(x,md),"dy_dmd":np.gradient(y,md),
        "inclination":np.arctan2(np.sqrt(np.gradient(x)**2+np.gradient(y)**2),np.abs(np.gradient(z))+1e-9)*180/np.pi,
        "raw_dz":np.diff(z,prepend=z[0]),"raw_dmd":np.diff(md,prepend=md[0]),
        "last_known_tvt":lkt,"rows_since_ps":np.maximum(0,np.arange(n)-ps).astype(float),
        "rows_before_ps":np.maximum(0,ps-np.arange(n)).astype(float),"ps_idx":float(ps),
        "tvt_z_slope":np.full(n,slope),"physics_tvt":anchored_physics,
        "physics_vs_lkt":anchored_physics-lkt,"physics_dtvt":slope*np.diff(z,prepend=z[0]),
        "gr_at_physics":gr_at_physics,"gr_dev_physics":gr_dev,
        "total_z_change_post":np.full(n,total_z),"physics_tvt_at_end":np.full(n,physics_tvt_at_end),
        "post_dz_rate":np.full(n,post_dz_rate),"dz_rate_change":np.full(n,dz_rate_change),
        "n_post_ps":np.full(n,float(n_post)),
        "post_ps_frac":np.where(n_post>0,np.maximum(0,np.arange(n)-ps)/n_post,0.).astype(float),
        "pre_r2":np.full(n,pre_r2)})
    if ps>2:
        pre_dz=np.gradient(z[:ps],md[:ps]) if ps>1 else np.array([0.])
        vtvt=tvt_inp[:ps].values; vtvt=vtvt[~np.isnan(vtvt)]
        pdtvt=np.diff(vtvt) if len(vtvt)>1 else np.array([0.])
        feat["pre_ps_dz_slope"]=np.full(n,pre_dz[-min(50,len(pre_dz)):].mean())
        feat["pre_ps_dtvt_mean"]=np.full(n,pdtvt[-min(50,len(pdtvt)):].mean())
        feat["pre_ps_dtvt_std"]=np.full(n,(pdtvt[-min(50,len(pdtvt)):].std() if len(pdtvt)>1 else 0.))
    else:
        for k in ["pre_ps_dz_slope","pre_ps_dtvt_mean","pre_ps_dtvt_std"]: feat[k]=np.zeros(n)
    df=pd.DataFrame(feat); df["is_post_ps"]=(np.arange(n)>=ps).astype(int)
    if split=="train" and "TVT" in hw.columns:
        tvt_true=hw["TVT"].astype(float).values
        df["target_correction"]=tvt_true-anchored_physics; df["anchored_physics_col"]=anchored_physics
    return df

print("Computing val RMSE per well for both models...")
lgbm_rmses=[]; lstm_rmses=[]; well_ids_ok=[]
for wid in val_ids:
    try:
        hw,tw=load_well(wid,"train"); ps=get_ps(hw)
        tvt_true=hw["TVT"].astype(float).values[ps:]
        
        # LGBM prediction
        df=build_lgbm_features(hw,tw,"train"); df_p=df[df["is_post_ps"]==1]
        X=df_p[lgbm_v8.feature_name()].values.astype(np.float32)
        corr=lgbm_v8.predict(X)
        lgbm_tvt=gaussian_filter1d(df_p["anchored_physics_col"].values+corr,sigma=1.0)
        lgbm_rmse=np.sqrt(np.mean((lgbm_tvt[:len(tvt_true)]-tvt_true[:len(lgbm_tvt)])**2))
        
        # LSTM prediction
        ctx_f,post_f,correction,anch_phys=extract_features(hw,tw,"train")
        if len(post_f)>MAX_POST: post_f=post_f[:MAX_POST]; anch_phys=anch_phys[:MAX_POST]
        ctx_t=torch.from_numpy(ctx_f).unsqueeze(0).to(DEVICE)
        post_t=torch.from_numpy(post_f).unsqueeze(0).to(DEVICE)
        cl=torch.tensor([len(ctx_f)],dtype=torch.long).to(DEVICE)
        pl=torch.tensor([len(post_f)],dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            pred_corr=model(ctx_t,post_t,cl,pl).squeeze(0).cpu().numpy()
        lstm_tvt=gaussian_filter1d(anch_phys+pred_corr[:len(anch_phys)],sigma=1.0)
        lstm_rmse=np.sqrt(np.mean((lstm_tvt[:len(tvt_true)]-tvt_true[:len(lstm_tvt)])**2))
        
        lgbm_rmses.append(lgbm_rmse); lstm_rmses.append(lstm_rmse); well_ids_ok.append(wid)
    except Exception as e:
        pass

lgbm_rmses=np.array(lgbm_rmses); lstm_rmses=np.array(lstm_rmses)
print(f"LGBM v8 val:  mean={lgbm_rmses.mean():.3f}  median={np.median(lgbm_rmses):.3f}")
print(f"LSTM val:     mean={lstm_rmses.mean():.3f}  median={np.median(lstm_rmses):.3f}")

# Find optimal blend weight
best_alpha=0.5; best_rmse=999
for alpha in np.arange(0.,1.01,0.05):
    ens_rmses=alpha*lgbm_rmses+(1-alpha)*lstm_rmses
    m=ens_rmses.mean()
    if m<best_rmse: best_rmse=m; best_alpha=alpha
print(f"Optimal alpha (LGBM weight): {best_alpha:.2f}, ensemble mean RMSE: {best_rmse:.3f}")
for alpha in [0.0,0.2,0.4,0.5,0.6,0.7,0.8,1.0]:
    ens_rmses=alpha*lgbm_rmses+(1-alpha)*lstm_rmses
    print(f"  alpha={alpha:.1f}  ens_mean={ens_rmses.mean():.3f}")
