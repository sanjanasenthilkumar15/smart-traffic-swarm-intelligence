"""
gru_predictor.py - Tier 2: GRU Short-Horizon Queue Forecaster
KPIT Sparkle 2027 - Team InnovateX - RMK Engineering College

Trains a small GRU network on traffic_training_data.csv to predict
queue length ~30 seconds ahead. The forecast is injected into the
ONNX decision model so signals can adjust BEFORE congestion forms.

Architecture:
    Input  : 30-step rolling window of mean queue length
    Model  : GRU(input=1, hidden=32, num_layers=1, batch_first=True)
             -> Linear(32, 1)
    Output : Predicted queue count in ~30 seconds (scalar)
    Speed  : <5 ms on CPU - Raspberry Pi 5 compatible
"""

import os
import csv
import math
from collections import deque

import torch
import torch.nn as nn

WINDOW_SIZE   = 30
FORECAST_STEP = 6
HIDDEN_SIZE   = 32
NUM_LAYERS    = 1
EPOCHS        = 60
LR            = 1e-3
BATCH_SIZE    = 64
MODEL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gru_predictor.pth')
DATA_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'traffic_training_data.csv')


class _GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru    = nn.GRU(1, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.linear = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])


class GRUPredictor:
    """Thin wrapper around _GRUModel for use inside the simulation loop."""

    def __init__(self):
        self._model   = _GRUModel()
        self._history = deque(maxlen=WINDOW_SIZE)
        self._mean    = 0.0
        self._std     = 1.0
        self._loaded  = False

    def load(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            self._model.load_state_dict(ckpt['model_state'])
            self._mean = ckpt.get('mean', 0.0)
            self._std  = ckpt.get('std',  1.0)
            self._model.eval()
            self._loaded = True
            return True
        except Exception:
            return False

    def update(self, total_queue: float):
        self._history.append(total_queue)

    def predict(self, history_list) -> float:
        if not self._loaded or len(history_list) < WINDOW_SIZE:
            return float(sum(history_list[-WINDOW_SIZE:]) / max(len(history_list), 1))
        return self._infer(list(history_list[-WINDOW_SIZE:]))

    def forecast(self) -> float:
        if not self._loaded or len(self._history) < WINDOW_SIZE:
            return float(sum(self._history) / max(len(self._history), 1))
        return self._infer(list(self._history))

    def _infer(self, window) -> float:
        std = self._std if self._std > 0 else 1.0
        normed = [((v - self._mean) / std) for v in window]
        x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_norm = self._model(x).item()
        return max(0.0, pred_norm * std + self._mean)


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    """Train the GRU on traffic_training_data.csv and save weights."""
    rows = []
    with open(data_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                q = (float(row['queue_lane_0']) + float(row['queue_lane_1']) +
                     float(row['queue_lane_2']) + float(row['queue_lane_3'])) / 4.0
                rows.append(q)
            except (KeyError, ValueError):
                continue

    if len(rows) < WINDOW_SIZE + FORECAST_STEP + 1:
        print("[GRU] Not enough data to train.")
        return

    mean_q = sum(rows) / len(rows)
    std_q  = math.sqrt(sum((r - mean_q) ** 2 for r in rows) / len(rows)) or 1.0
    normed = [(r - mean_q) / std_q for r in rows]

    X, Y = [], []
    for i in range(len(normed) - WINDOW_SIZE - FORECAST_STEP):
        X.append(normed[i: i + WINDOW_SIZE])
        Y.append(normed[i + WINDOW_SIZE + FORECAST_STEP - 1])

    n = len(X)
    print(f"[GRU] Training on {n} samples for {EPOCHS} epochs ...")

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    model     = _GRUModel()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, batches = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx  = perm[i: i + BATCH_SIZE]
            xb, yb = X_t[idx], Y_t[idx]
            loss = criterion(model(xb), yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            batches += 1
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  loss={epoch_loss / batches:.4f}")

    torch.save({'model_state': model.state_dict(), 'mean': mean_q, 'std': std_q}, model_path)
    print(f"[GRU] Model saved: {model_path}")

    predictor = GRUPredictor()
    predictor.load(model_path)
    pred = predictor.predict(rows[:WINDOW_SIZE])
    print(f"[GRU] Test prediction: {pred:.1f} vehicles")


if __name__ == '__main__':
    train()
