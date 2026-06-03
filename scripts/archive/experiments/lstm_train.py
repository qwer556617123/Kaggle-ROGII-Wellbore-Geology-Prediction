"""
LSTM sequence model for TVT correction prediction.
Target: correction = true_tvt - anchored_physics (same as LightGBM v8)
Architecture:
  - Context encoder: BiLSTM over last K pre-PS steps
  - Post-PS decoder: LSTM initialized from encoder hidden state
  - Per-step output: scalar correction
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import joblib, warnings, time
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"E:\Code\ROGII - Wellbore Geology Prediction")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
MODELS_DIR = DATA_DIR / "models"
SUBS_DIR   = DATA_DIR / "submissions"
FEAT_DIR   = DATA_DIR / "features"
TEST_WELLS = ["000d7d20", "00bbac68", "00e12e8b"]

CTX_LEN  = 200   # pre-PS context steps
MAX_POST = 5000  # truncate post-PS to this length
N_FEAT   = 10    # input features per step
HIDDEN   = 128
N_LAYERS = 2
DROPOUT  = 0.15
BATCH    = 32
EPOCHS   = 60
LR       = 1e-3


# ── helpers ────────────────────────────────────────────────────────────────
def fill_arr(a):
    return pd.Series(a).ffill().bfill().astype(float).values

def load_well(wid, split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    return (pd.read_csv(d / f"{wid}__horizontal_well.csv"),
            pd.read_csv(d / f"{wid}__typewell.csv"))

def get_ps(hw):
    mask = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    return int(mask.idxmax()) if mask.any() else len(hw)


def extract_features(hw, tw, split="train"):
    """Return (ctx_feat, post_feat, post_correction, anchored_physics_post) numpy arrays."""
    n   = len(hw)
    ps  = get_ps(hw)
    gr  = fill_arr(hw["GR"])
    md  = hw["MD"].astype(float).values
    z   = hw["Z"].astype(float).values
    tvt_inp = hw["TVT_input"].astype(str).replace("", np.nan).astype(float)
    lkt     = tvt_inp.ffill().bfill().values

    # Physics model (pre-PS linear fit)
    if split == "train" and "TVT" in hw.columns:
        pre_tvt = hw["TVT"].astype(float).values[:ps]
    else:
        known = tvt_inp[:ps].values
        pre_tvt = known[~np.isnan(known)]
    pre_z_all = z[:ps]
    if split == "train":
        pre_z_fit = pre_z_all
    else:
        valid = ~np.isnan(tvt_inp[:ps].values)
        pre_z_fit = pre_z_all[valid]

    if len(pre_z_fit) > 5:
        reg = LinearRegression().fit(pre_z_fit.reshape(-1, 1), pre_tvt)
        slope = float(reg.coef_[0])
    else:
        slope = -1.0

    anchor_tvt = float(tvt_inp.ffill().iloc[max(0, ps - 1)])
    Z_anchor   = z[max(0, ps - 1)]
    anchored_physics = anchor_tvt + slope * (z - Z_anchor)

    # Global trajectory features
    post_z   = z[ps:] if ps < n else z[:]
    post_md  = md[ps:] if ps < n else md[:]
    n_post   = len(post_z)
    z_end    = float(post_z[-1]) if n_post > 0 else float(z[-1])
    total_z  = z_end - Z_anchor
    total_md = (float(post_md[-1]) - float(post_md[0])) if n_post > 1 else 1.
    post_dz_rate = total_z / total_md
    phys_at_end  = slope * total_z  # physics-predicted TVT change at end

    # Per-step features (10-dim)
    gr_mean  = np.mean(gr[:ps]) if ps > 0 else np.mean(gr)
    gr_std   = np.std(gr[:ps]) + 1e-6
    gr_norm  = (gr - gr_mean) / gr_std
    dz_dmd   = np.gradient(z, md)
    phys_vs_lkt  = anchored_physics - lkt
    phys_dtvt    = slope * np.diff(z, prepend=z[0])
    progress     = np.maximum(0, np.arange(n) - ps).astype(float) / max(n_post, 1)
    phys_end_col = np.full(n, phys_at_end / 100.)
    slope_col    = np.full(n, slope)
    post_dz_col  = np.full(n, post_dz_rate)
    is_post      = (np.arange(n) >= ps).astype(float)

    feats = np.stack([
        gr_norm, dz_dmd, phys_vs_lkt, phys_dtvt,
        progress, phys_end_col, slope_col, post_dz_col, is_post,
        lkt / 1e4  # scaled absolute depth anchor
    ], axis=1).astype(np.float32)  # [n, 10]

    # Context: last CTX_LEN pre-PS steps
    ctx_start = max(0, ps - CTX_LEN)
    ctx_feat  = feats[ctx_start:ps]  # up to [CTX_LEN, 10]

    # Post-PS sequence
    post_feat  = feats[ps:]  # [n_post, 10]

    if split == "train" and "TVT" in hw.columns:
        tvt_true = hw["TVT"].astype(float).values
        correction = (tvt_true - anchored_physics)[ps:]  # [n_post]
    else:
        correction = None

    return ctx_feat, post_feat, correction, anchored_physics[ps:]


# ── Dataset ────────────────────────────────────────────────────────────────
class WellDataset(Dataset):
    def __init__(self, samples):
        # samples: list of (ctx_feat, post_feat, correction, n_post)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad variable-length sequences within a batch."""
    ctx_feats, post_feats, corrections = zip(*batch)

    ctx_lens  = [len(c) for c in ctx_feats]
    post_lens = [len(p) for p in post_feats]
    max_ctx   = max(ctx_lens)
    max_post  = max(post_lens)

    B = len(batch)
    ctx_pad  = np.zeros((B, max_ctx, N_FEAT), dtype=np.float32)
    post_pad = np.zeros((B, max_post, N_FEAT), dtype=np.float32)
    corr_pad = np.zeros((B, max_post), dtype=np.float32)
    mask     = np.zeros((B, max_post), dtype=bool)

    for i, (c, p, r) in enumerate(zip(ctx_feats, post_feats, corrections)):
        # right-align context (most recent steps at the end)
        ctx_pad[i, -len(c):] = c
        post_pad[i, :len(p)] = p
        corr_pad[i, :len(r)] = r
        mask[i, :len(r)] = True

    return (torch.from_numpy(ctx_pad),
            torch.from_numpy(post_pad),
            torch.from_numpy(corr_pad),
            torch.from_numpy(mask),
            torch.tensor(ctx_lens, dtype=torch.long),
            torch.tensor(post_lens, dtype=torch.long))


# ── Model ──────────────────────────────────────────────────────────────────
class WellLSTM(nn.Module):
    def __init__(self, input_dim=N_FEAT, hidden_dim=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim // 2, n_layers,
                               batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0.)
        self.decoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                               batch_first=True, bidirectional=False, dropout=dropout if n_layers > 1 else 0.)
        # Project encoder's final state → decoder init state
        self.h_proj = nn.Linear(hidden_dim * n_layers, hidden_dim * n_layers)
        self.c_proj = nn.Linear(hidden_dim * n_layers, hidden_dim * n_layers)
        self.output  = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def encode_ctx(self, ctx_seq, ctx_lens):
        """ctx_seq: [B, T_ctx, F]  → h0, c0 for decoder"""
        B = ctx_seq.size(0)
        packed = pack_padded_sequence(ctx_seq, ctx_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.encoder(packed)
        # h_n: [n_layers*2, B, H//2] → merge forward/backward
        # Reshape: [n_layers, 2, B, H//2] → [n_layers, B, H]
        h_n = h_n.view(self.n_layers, 2, B, self.hidden_dim // 2)
        c_n = c_n.view(self.n_layers, 2, B, self.hidden_dim // 2)
        h_n = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)  # [n_layers, B, H]
        c_n = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
        # Project through MLP to allow adaptation
        h_flat = h_n.permute(1, 0, 2).reshape(B, -1)  # [B, n_layers*H]
        c_flat = c_n.permute(1, 0, 2).reshape(B, -1)
        h_out  = self.h_proj(h_flat).reshape(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_out  = self.c_proj(c_flat).reshape(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h_out, c_out

    def decode_post(self, post_seq, post_lens, h0, c0):
        """post_seq: [B, T_post, F]  → corrections [B, T_post]"""
        packed = pack_padded_sequence(post_seq, post_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.decoder(packed, (h0, c0))
        out_pad, _ = pad_packed_sequence(out, batch_first=True)  # [B, T_post, H]
        corr = self.output(out_pad).squeeze(-1)  # [B, T_post]
        return corr

    def forward(self, ctx_seq, post_seq, ctx_lens, post_lens):
        h0, c0 = self.encode_ctx(ctx_seq, ctx_lens)
        return self.decode_post(post_seq, post_lens, h0, c0)


# ── Load data ──────────────────────────────────────────────────────────────
print("Loading training data...")
train_ids = pd.read_csv(FEAT_DIR / "train_ids.csv", header=None)[0].tolist()
val_ids   = pd.read_csv(FEAT_DIR / "val_ids.csv",   header=None)[0].tolist()
print(f"  Train: {len(train_ids)} wells, Val: {len(val_ids)} wells")

def load_dataset(well_ids, split="train"):
    samples = []
    for wid in well_ids:
        try:
            hw, tw = load_well(wid, split)
            ctx_f, post_f, correction, anch = extract_features(hw, tw, split)
            if correction is None or len(correction) == 0:
                continue
            # Truncate long post-PS sections
            if len(post_f) > MAX_POST:
                post_f    = post_f[:MAX_POST]
                correction = correction[:MAX_POST]
                anch       = anch[:MAX_POST]
            samples.append((ctx_f, post_f, correction))
        except Exception as e:
            print(f"  SKIP {wid}: {e}")
    return samples

train_samples = load_dataset(train_ids)
val_samples   = load_dataset(val_ids)
print(f"  Loaded: {len(train_samples)} train, {len(val_samples)} val wells")

train_ds = WellDataset(train_samples)
val_ds   = WellDataset(val_samples)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=collate_fn, num_workers=0)

# ── Train ──────────────────────────────────────────────────────────────────
model     = WellLSTM().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
criterion = nn.MSELoss()

best_val  = float("inf")
best_ep   = 0

print(f"\nTraining LSTM ({sum(p.numel() for p in model.parameters()):,} params) for {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.; tr_n = 0
    for ctx, post, corr, mask, ctx_lens, post_lens in train_dl:
        ctx   = ctx.to(DEVICE);  post = post.to(DEVICE)
        corr  = corr.to(DEVICE); mask = mask.to(DEVICE)
        ctx_lens  = ctx_lens.to(DEVICE)
        post_lens = post_lens.to(DEVICE)

        pred = model(ctx, post, ctx_lens, post_lens)  # [B, T_post]
        # Trim pred to mask shape
        T = min(pred.size(1), corr.size(1))
        loss = criterion(pred[:, :T][mask[:, :T]], corr[:, :T][mask[:, :T]])
        optimizer.zero_grad(); loss.backward(); 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item() * mask.sum().item()
        tr_n    += mask.sum().item()

    model.eval()
    val_loss = 0.; val_n = 0
    with torch.no_grad():
        for ctx, post, corr, mask, ctx_lens, post_lens in val_dl:
            ctx   = ctx.to(DEVICE);  post = post.to(DEVICE)
            corr  = corr.to(DEVICE); mask = mask.to(DEVICE)
            ctx_lens  = ctx_lens.to(DEVICE)
            post_lens = post_lens.to(DEVICE)
            pred = model(ctx, post, ctx_lens, post_lens)
            T = min(pred.size(1), corr.size(1))
            loss = criterion(pred[:, :T][mask[:, :T]], corr[:, :T][mask[:, :T]])
            val_loss += loss.item() * mask.sum().item()
            val_n    += mask.sum().item()

    tr_rmse  = np.sqrt(tr_loss  / max(tr_n,  1))
    val_rmse = np.sqrt(val_loss / max(val_n, 1))
    scheduler.step()

    if val_rmse < best_val:
        best_val = val_rmse; best_ep = epoch
        torch.save(model.state_dict(), MODELS_DIR / "lstm_best.pt")

    if epoch % 5 == 0 or epoch <= 3:
        print(f"  Ep {epoch:3d} | train RMSE={tr_rmse:.3f} | val RMSE={val_rmse:.3f}"
              f"  {'← best' if epoch==best_ep else ''}")

print(f"\nBest val RMSE: {best_val:.3f} ft at epoch {best_ep}")

# ── Per-well val RMSE ──────────────────────────────────────────────────────
print("\nPer-well val RMSE (LSTM):")
model.load_state_dict(torch.load(MODELS_DIR / "lstm_best.pt", map_location=DEVICE))
model.eval()

well_rmses = []
for wid in val_ids:
    try:
        hw, tw   = load_well(wid, "train")
        ctx_f, post_f, correction, anch_phys = extract_features(hw, tw, "train")
        if correction is None or len(correction) == 0:
            continue
        if len(post_f) > MAX_POST:
            post_f = post_f[:MAX_POST]; correction = correction[:MAX_POST]; anch_phys = anch_phys[:MAX_POST]

        ctx_t  = torch.from_numpy(ctx_f).unsqueeze(0).to(DEVICE)   # [1, T_ctx, F]
        post_t = torch.from_numpy(post_f).unsqueeze(0).to(DEVICE)  # [1, T_post, F]
        cl = torch.tensor([len(ctx_f)], dtype=torch.long).to(DEVICE)
        pl = torch.tensor([len(post_f)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            pred_corr = model(ctx_t, post_t, cl, pl).squeeze(0).cpu().numpy()
        tvt_pred = anch_phys + pred_corr[:len(anch_phys)]
        tvt_true = hw["TVT"].astype(float).values[get_ps(hw):][:len(tvt_pred)]
        rmse = np.sqrt(np.mean((tvt_pred - tvt_true) ** 2))
        well_rmses.append(rmse)
    except Exception as e:
        pass

well_rmses = np.array(well_rmses)
print(f"  Mean per-well RMSE: {well_rmses.mean():.3f} ft")
print(f"  Median:             {np.median(well_rmses):.3f} ft")
print(f"  Worst 5: {sorted(well_rmses)[-5:]}")

# ── Generate LSTM submission ───────────────────────────────────────────────
print("\nGenerating LSTM submission...")
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
for wid in TEST_WELLS:
    hw, tw = load_well(wid, "test")
    ps = get_ps(hw)
    ctx_f, post_f, _, anch_phys = extract_features(hw, tw, "test")

    ctx_t  = torch.from_numpy(ctx_f).unsqueeze(0).to(DEVICE)
    post_t = torch.from_numpy(post_f).unsqueeze(0).to(DEVICE)
    cl = torch.tensor([len(ctx_f)], dtype=torch.long).to(DEVICE)
    pl = torch.tensor([len(post_f)], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        pred_corr = model(ctx_t, post_t, cl, pl).squeeze(0).cpu().numpy()
    tvt_pred = gaussian_filter1d(anch_phys + pred_corr[:len(anch_phys)], sigma=1.0)

    empty = hw["TVT_input"].isna() | (hw["TVT_input"].astype(str).str.strip() == "")
    rows  = hw.index[empty].tolist()
    id_map = {f"{wid}_{i}": v for i, v in zip(rows, tvt_pred)}
    mask   = sub["id"].isin(set(id_map.keys()))
    sub.loc[mask, "tvt"] = sub.loc[mask, "id"].map(id_map)
    print(f"  [{wid}] TVT=[{tvt_pred.min():.1f},{tvt_pred.max():.1f}]")

sub.to_csv(SUBS_DIR / "lstm.csv", index=False)
print(f"  Saved lstm.csv  NaN={sub.tvt.isna().sum()}")

# ── Ensemble (0.5 LightGBM final + 0.5 LSTM) ──────────────────────────────
print("\nCreating ensemble submission...")
lgbm_sub = pd.read_csv(SUBS_DIR / "lgbm_final.csv")
lstm_sub  = pd.read_csv(SUBS_DIR / "lstm.csv")
ens = lgbm_sub.copy()
ens["tvt"] = 0.5 * lgbm_sub["tvt"] + 0.5 * lstm_sub["tvt"]
ens.to_csv(SUBS_DIR / "ensemble_50_50.csv", index=False)
print(f"  Saved ensemble_50_50.csv  NaN={ens.tvt.isna().sum()}")

# Also save 0.6/0.4 weights
ens2 = lgbm_sub.copy()
ens2["tvt"] = 0.6 * lgbm_sub["tvt"] + 0.4 * lstm_sub["tvt"]
ens2.to_csv(SUBS_DIR / "ensemble_60_40.csv", index=False)
print(f"  Saved ensemble_60_40.csv")

print("\nDone!")
