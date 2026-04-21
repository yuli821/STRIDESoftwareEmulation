"""TCN predictor stub (active only if PyTorch is installed)."""
from __future__ import annotations

import numpy as np

from .base import BasePredictor

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if _HAS_TORCH:
    class _CausalConv1d(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, k: int) -> None:
            super().__init__()
            self.pad = k - 1
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=self.pad)

        def forward(self, x):
            y = self.conv(x)
            if self.pad > 0:
                y = y[:, :, :-self.pad]
            return y

    class _TCN(nn.Module):
        def __init__(self, in_ch: int, ch: int, k: int, layers: int) -> None:
            super().__init__()
            blocks = []
            c_in = in_ch
            for _ in range(layers):
                blocks.append(_CausalConv1d(c_in, ch, k))
                blocks.append(nn.ReLU())
                c_in = ch
            self.seq = nn.Sequential(*blocks)
            self.head = nn.Linear(ch, 1)

        def forward(self, x):
            y = self.seq(x)
            y = y.mean(dim=-1)
            return torch.sigmoid(self.head(y)).squeeze(-1)


class TCNPredictor(BasePredictor):
    def __init__(self, num_queues: int, window: int,
                 channels: int = 16, kernel: int = 3, layers: int = 2) -> None:
        super().__init__(num_queues, window)
        if not _HAS_TORCH:
            self.model = None
            return
        self.model = _TCN(4, channels, kernel, layers)
        self.model.eval()

    def predict(self) -> np.ndarray:
        if not _HAS_TORCH or self.model is None or len(self._history) == 0:
            return np.zeros(self.num_queues, dtype=np.float64)
        hist = np.stack(list(self._history), axis=0)
        x = torch.from_numpy(hist.transpose(1, 2, 0).astype(np.float32))
        with torch.no_grad():
            y = self.model(x).cpu().numpy()
        return np.clip(y, 0.0, 1.0)
