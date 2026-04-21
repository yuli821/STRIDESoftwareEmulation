"""Toeplitz 5-tuple hash (matches the FPGA `toeplitz_hash.sv` implementation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


DEFAULT_RSS_KEY_40 = bytes([
    0x6d, 0x5a, 0x56, 0xda, 0x25, 0x5b, 0x0e, 0xc2,
    0x41, 0x67, 0x25, 0x3d, 0x43, 0xa3, 0x8f, 0xb0,
    0xd0, 0xca, 0x2b, 0xcb, 0xae, 0x7b, 0x30, 0xb4,
    0x77, 0xcb, 0x2d, 0xa3, 0x80, 0x30, 0xf2, 0x0c,
    0x6a, 0x42, 0xb7, 0x3b, 0xbe, 0xac, 0x01, 0xfa,
])


def _build_window_table(key: bytes) -> np.ndarray:
    n_bits = len(key) * 8
    key_bits = np.zeros(n_bits, dtype=np.uint8)
    for i, byte in enumerate(key):
        for b in range(8):
            key_bits[i * 8 + b] = 1 if (byte & (0x80 >> b)) else 0
    tiled = np.concatenate([key_bits, key_bits[:32]])
    windows = np.zeros(n_bits, dtype=np.uint64)
    for p in range(n_bits):
        w = 0
        for j in range(32):
            if tiled[p + j]:
                w |= (1 << (31 - j))
        windows[p] = w
    return windows


@dataclass
class Toeplitz:
    key: bytes = DEFAULT_RSS_KEY_40

    def __post_init__(self) -> None:
        self._windows = _build_window_table(self.key)

    def hash_5tuple(
        self,
        src_ip: np.ndarray, dst_ip: np.ndarray,
        src_port: np.ndarray, dst_port: np.ndarray,
        proto: np.ndarray,
    ) -> np.ndarray:
        src_ip = np.asarray(src_ip, dtype=np.uint32)
        dst_ip = np.asarray(dst_ip, dtype=np.uint32)
        src_port = np.asarray(src_port, dtype=np.uint16)
        dst_port = np.asarray(dst_port, dtype=np.uint16)
        proto = np.asarray(proto, dtype=np.uint8)
        n = max(src_ip.size, dst_ip.size, src_port.size, dst_port.size, proto.size)
        shape = (n,)
        src_ip = np.broadcast_to(src_ip, shape)
        dst_ip = np.broadcast_to(dst_ip, shape)
        src_port = np.broadcast_to(src_port, shape)
        dst_port = np.broadcast_to(dst_port, shape)
        proto = np.broadcast_to(proto, shape)

        packed = np.zeros((n, 13), dtype=np.uint8)
        packed[:, 0] = (src_ip >> 24) & 0xFF
        packed[:, 1] = (src_ip >> 16) & 0xFF
        packed[:, 2] = (src_ip >> 8) & 0xFF
        packed[:, 3] = src_ip & 0xFF
        packed[:, 4] = (dst_ip >> 24) & 0xFF
        packed[:, 5] = (dst_ip >> 16) & 0xFF
        packed[:, 6] = (dst_ip >> 8) & 0xFF
        packed[:, 7] = dst_ip & 0xFF
        packed[:, 8] = (src_port >> 8) & 0xFF
        packed[:, 9] = src_port & 0xFF
        packed[:, 10] = (dst_port >> 8) & 0xFF
        packed[:, 11] = dst_port & 0xFF
        packed[:, 12] = proto

        n_bits = 13 * 8
        bits = np.unpackbits(packed, axis=1)[:, :n_bits]
        windows = self._windows[:n_bits].astype(np.uint32)
        mask = bits.astype(np.uint32)
        out = np.bitwise_xor.reduce(windows[None, :] * mask, axis=1)
        return out.astype(np.uint32)

    def bucket_of(self, *args, num_buckets: int) -> np.ndarray:
        h = self.hash_5tuple(*args)
        return (h & (num_buckets - 1)).astype(np.int64)
