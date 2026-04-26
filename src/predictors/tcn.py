"""TCN per-queue hotspot predictor.

Architecture
------------
Small causal temporal conv net. Input: a length-W window of per-queue
features ``[B_q_gen, R_q_peak, K_q, P_q]``. The queue index is used as
the batch axis (weights are shared across queues). Output: a single
scalar per queue that approximates P_q at the next stateless epoch.

The training-time head is a raw logit (no sigmoid) so we can train with
``BCEWithLogitsLoss`` / ``MSELoss`` on bounded targets in ``[0, 1]``
without the near-saturation gradient collapse of sigmoid+MSE. The
sigmoid is applied only at inference.

Checkpoint format
-----------------
``models/tcn_pred.pt`` (created by ``scripts/train_tcn.py``) is a
dict with:

    state_dict : trained parameters
    norm_mean  : (input_channels,) per-channel mean used at training
    norm_std   : (input_channels,) per-channel stddev used at training
    W          : training window length (asserted at load)

Inference contract
------------------
Constructing ``TCNPredictor(..., checkpoint=...)`` with an empty string
or a missing file raises, rather than silently running with random
weights (which was the failure mode when the predictor factory was
first wired up).
"""
from __future__ import annotations

import os
from typing import Optional

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
        """1-D conv with strict causality (no future leakage).

        We pad ``kernel - 1`` zeros on the LEFT only (trim right after
        the conv) so output[t] depends on inputs <= t.
        """

        def __init__(self, in_ch: int, out_ch: int, k: int) -> None:
            super().__init__()
            self.pad = k - 1
            self.conv = nn.Conv1d(in_ch, out_ch,
                                  kernel_size=k, padding=self.pad)

        def forward(self, x):
            y = self.conv(x)
            if self.pad > 0:
                y = y[:, :, :-self.pad]
            return y

    class _TCN(nn.Module):
        """Stacked causal conv + ReLU blocks, global mean-pool, linear
        head. ``sigmoid_output`` is false at training time (BCE handles
        the sigmoid internally) and true only if callers want a baked-in
        probability output.
        """

        def __init__(self, in_ch: int, ch: int, k: int, layers: int,
                     sigmoid_output: bool = False) -> None:
            super().__init__()
            blocks = []
            c_in = in_ch
            for _ in range(layers):
                blocks.append(_CausalConv1d(c_in, ch, k))
                blocks.append(nn.ReLU())
                c_in = ch
            self.seq = nn.Sequential(*blocks)
            self.head = nn.Linear(ch, 1)
            self.sigmoid_output = bool(sigmoid_output)

        def forward(self, x):
            """Input: (batch, in_ch, T). Output: (batch,) logits or
            probabilities depending on ``sigmoid_output``."""
            y = self.seq(x)
            y = y.mean(dim=-1)              # (batch, ch)
            y = self.head(y).squeeze(-1)    # (batch,)
            if self.sigmoid_output:
                y = torch.sigmoid(y)
            return y


class TCNPredictor(BasePredictor):
    """Load-and-run inference wrapper around ``_TCN``.

    Parameters mirror ``PredictorConfig``. ``checkpoint`` (path to a
    trained ``.pt`` file) is required; constructing without one raises
    so callers don't accidentally schedule on random-initialised
    weights.
    """

    def __init__(self, num_queues: int, window: int,
                 channels: int = 16, kernel: int = 3, layers: int = 2,
                 checkpoint: str = "") -> None:
        super().__init__(num_queues, window)
        if not _HAS_TORCH:
            raise RuntimeError(
                "TCN predictor requires PyTorch; install torch or set "
                "predictor_type to ewma/linear/oracle/none.")
        if not checkpoint:
            raise RuntimeError(
                "TCN predictor requires predictor.tcn_checkpoint to "
                "point at a trained .pt file. Run "
                "scripts/train_tcn.py first.")
        ckpt_path = os.path.abspath(checkpoint)
        if not os.path.exists(ckpt_path):
            raise RuntimeError(
                f"TCN checkpoint not found: {ckpt_path}")

        self.model = _TCN(4, channels, kernel, layers,
                          sigmoid_output=False)
        ckpt = torch.load(ckpt_path, map_location="cpu",
                          weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        ckpt_W = int(ckpt.get("W", window))
        if ckpt_W != window:
            raise RuntimeError(
                f"TCN window mismatch: checkpoint W={ckpt_W}, sim "
                f"predictor.W_window_epochs={window}. Regenerate the "
                f"dataset/train with the same window as the sim, or "
                f"change the sim window to match.")
        self.W = ckpt_W

        mean = np.asarray(ckpt["norm_mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(ckpt["norm_std"], dtype=np.float32).reshape(-1)
        if mean.size != 4 or std.size != 4:
            raise RuntimeError(
                f"TCN checkpoint normalization shape wrong: "
                f"mean={mean.shape}, std={std.shape}")
        # Broadcast-ready tensors: shape (1, 4, 1).
        self._mean = torch.from_numpy(mean).view(1, 4, 1)
        self._std = torch.from_numpy(np.where(std > 1e-12, std, 1.0)
                                     .astype(np.float32)).view(1, 4, 1)

    def predict(self) -> np.ndarray:
        """Return per-queue risk in ``[0, 1]``.

        Pads on the left with zeros when the history deque has fewer
        than ``W`` entries so the network always sees a ``(N, 4, W)``
        tensor (same shape it was trained on). This keeps the first
        few epochs' outputs statistically consistent with training.
        """
        if len(self._history) == 0:
            return np.zeros(self.num_queues, dtype=np.float64)
        hist = np.stack(list(self._history), axis=0)    # (T, N, 4)
        T, N, C = hist.shape
        W = self.W
        if T < W:
            pad = np.zeros((W - T, N, C), dtype=hist.dtype)
            hist = np.concatenate([pad, hist], axis=0)
        elif T > W:
            hist = hist[-W:]
        # Reshape to (N, C, W) and z-score per channel.
        x = torch.from_numpy(
            hist.transpose(1, 2, 0).astype(np.float32))  # (N, 4, W)
        x = (x - self._mean) / self._std
        with torch.no_grad():
            logits = self.model(x)                       # (N,)
            y = torch.sigmoid(logits).cpu().numpy()
        return np.clip(y.astype(np.float64), 0.0, 1.0)
