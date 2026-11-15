"""Quality classifiers: rule-based SQI thresholds, gradient boosting and a small 1D CNN."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from ecgq.quality.features import ECG_FEATURES

CLASSES = np.array([1, 2, 3])


def balanced_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Weight each window by 1 / (windows of its class) * 1 / (windows of its record).

    Three BUT QDB records are annotated for 24 h or more and the other fifteen for about
    40 minutes, and class 1 dominates. Without this, the model would mostly learn the three
    long records and the majority class.
    """
    w = np.ones(len(y), dtype=float)
    for c in np.unique(y):
        w[y == c] /= np.sum(y == c)
    for g in np.unique(groups):
        w[groups == g] /= np.sqrt(np.sum(groups == g))
    return w * len(y) / w.sum()


# ------------------------------------------------------------------ rule-based baseline


@dataclass
class RuleClassifier:
    """Interpretable thresholds on three SQIs, fitted by grid search on training data.

    class 3 if bSQI < b3 or the heart rate is implausible (QRS not reliably detectable)
    class 1 if bSQI >= b1 and tSQI >= t1 and basSQI >= s1 (clean morphology)
    class 2 otherwise
    """

    b3: float = 0.6
    b1: float = 0.9
    t1: float = 0.8
    s1: float = 0.9
    hr_range: tuple[float, float] = (30.0, 220.0)
    grid: dict = field(default_factory=lambda: {
        "b3": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "b1": [0.8, 0.85, 0.9, 0.95, 0.98],
        "t1": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        "s1": [0.8, 0.9, 0.95, 0.98, 0.99],
    })

    def _cols(self, X: np.ndarray):
        idx = {k: ECG_FEATURES.index(k) for k in ("bsqi", "tsqi", "bassqi", "hr_pt")}
        return X[:, idx["bsqi"]], X[:, idx["tsqi"]], X[:, idx["bassqi"]], X[:, idx["hr_pt"]]

    def _predict(self, cols, b3, b1, t1, s1) -> np.ndarray:
        b, t, s, hr = cols
        bad_hr = (hr < self.hr_range[0]) | (hr > self.hr_range[1])
        pred = np.full(len(b), 2)
        pred[(b >= b1) & (t >= t1) & (s >= s1)] = 1
        pred[(b < b3) | bad_hr] = 3
        return pred

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> RuleClassifier:
        cols = self._cols(X)
        best, best_f1 = None, -1.0
        g = self.grid
        for b3, b1, t1, s1 in itertools.product(g["b3"], g["b1"], g["t1"], g["s1"]):
            if b3 >= b1:
                continue
            p = self._predict(cols, b3, b1, t1, s1)
            f = f1_score(y, p, labels=CLASSES, average="macro", sample_weight=sample_weight,
                         zero_division=0)
            if f > best_f1:
                best, best_f1 = (b3, b1, t1, s1), f
        self.b3, self.b1, self.t1, self.s1 = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict(self._cols(X), self.b3, self.b1, self.t1, self.s1)

    def params(self) -> dict:
        return {"b3": self.b3, "b1": self.b1, "t1": self.t1, "s1": self.s1}


# ----------------------------------------------------------------- gradient boosting


def make_gbm(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


# -------------------------------------------------------------------------- 1D CNN


def build_cnn(in_channels: int, n_classes: int = 3):
    """Small 1D CNN (about 35k parameters) for 10 s windows at 100 Hz.

    Four conv blocks (conv, batch norm, ReLU, max-pool) widen the receptive field to about
    1.5 s, enough to see one or two beats and the noise around them. Global average and max
    pooling make the head independent of where in the window the artefact sits.
    """
    import torch
    from torch import nn

    class QualityCNN(nn.Module):
        def __init__(self):
            super().__init__()

            def block(cin, cout, k):
                return nn.Sequential(
                    nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                )

            self.features = nn.Sequential(
                block(in_channels, 16, 7), block(16, 32, 7), block(32, 64, 5), block(64, 64, 5),
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, x):
            h = self.features(x)
            h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
            return self.head(h)

    return QualityCNN()


def train_cnn(X: np.ndarray, y: np.ndarray, weights: np.ndarray, epochs: int = 8,
              samples_per_epoch: int = 12000, batch: int = 128, lr: float = 1e-3,
              seed: int = 0, log=None):
    """Train with a weighted sampler (class and record balance) and light augmentation."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_cnn(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    p = weights / weights.sum()
    yt = torch.as_tensor(y - 1, dtype=torch.long)
    for ep in range(epochs):
        model.train()
        idx = rng.choice(len(X), size=samples_per_epoch, replace=True, p=p)
        tot = 0.0
        for k in range(0, len(idx), batch):
            b = idx[k : k + batch]
            xb = torch.as_tensor(X[b], dtype=torch.float32)
            # Augmentation: ECG polarity flip and amplitude scaling (electrode placement
            # differs between people), small additive noise.
            flip = torch.where(torch.rand(len(b), 1) < 0.5, -1.0, 1.0)
            xb[:, 0] = xb[:, 0] * flip * (0.8 + 0.4 * torch.rand(len(b), 1))
            xb = xb + 0.02 * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        if log:
            log.info("    epoch %d loss %.4f", ep + 1, tot / len(idx))
    return model


def predict_cnn(model, X: np.ndarray, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            xb = torch.as_tensor(X[k : k + batch], dtype=torch.float32)
            probs.append(torch.softmax(model(xb), dim=1).numpy())
    pr = np.concatenate(probs) if probs else np.zeros((0, 3))
    return pr.argmax(axis=1) + 1, pr
