# LSTM Sequence Model for TVT Prediction

## Architecture

```
Input: [GR_window, Z, dZ/dMD, rows_since_PS, typewell_GR_at_estimated_TVT]
       Shape: (batch, seq_len=64, features=8)

→ LSTM(hidden=256, layers=3, dropout=0.2, bidirectional=False)
  (Unidirectional — we predict causally, no future lookahead)

→ Linear(256 → 64) + ReLU
→ Linear(64 → 1)

Output: predicted dTVT for each time step
```

## Model Code

```python
import torch
import torch.nn as nn

class TVTPredictorLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden=256, n_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, n_layers,
            batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, h=None):
        out, h = self.lstm(x, h)
        pred = self.head(out)  # (B, T, 1)
        return pred.squeeze(-1), h
```

## Training Setup (RTX 3060 Ti)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TVTPredictorLSTM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()

# Recommended batch: 32 wells, seq_len 64 → fits in 8GB VRAM
BATCH_SIZE = 32
SEQ_LEN = 64
```

## Inference (Autoregressive)

During inference for post-PS rows, feed the model step by step,
using the predicted dTVT to update the estimated TVT for the next step:

```python
def predict_autoregressive(model, hw, tw, ps_idx, device):
    model.eval()
    known_tvt = hw.iloc[:ps_idx]["TVT_input"].astype(float).values
    current_tvt = known_tvt[-1]
    
    predictions = []
    h = None
    
    with torch.no_grad():
        for i in range(ps_idx, len(hw)):
            feat = make_lstm_features(hw, tw, i, current_tvt)
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            dtvt_pred, h = model(x, h)
            current_tvt += dtvt_pred.item()
            predictions.append(current_tvt)
    
    return np.array(predictions)
```

## Notes
- Use teacher forcing during training (feed true TVT at each step)
- Curriculum learning: start with shorter prediction horizons
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
