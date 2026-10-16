"""Step 6: protocol-aware compression of the weight delta (229 values)."""
import numpy as np
from src.network_replay import MAX_FRAME

N_PARAMS = 229
OVERHEAD = 68               # header bytes: 297 byte shared payload - 229 int8 values
MAX_FRAGMENTS = 3           # LoRa budget: never send an update that needs more fragments
DENSE_INT8_BYTES = OVERHEAD + N_PARAMS      # 297

def int8_dense(vec):
    scale = np.abs(vec).max() / 127.0 or 1.0
    return np.clip(np.round(vec / scale), -127, 127) * scale

def topk_int8(vec, k):
    """Keep the k largest magnitudes, send them as (uint8 index, int8 value) pairs."""
    idx = np.argpartition(np.abs(vec), -k)[-k:]
    kept = vec[idx]
    scale = np.abs(kept).max() / 127.0 or 1.0
    out = np.zeros_like(vec)
    out[idx] = np.clip(np.round(kept / scale), -127, 127) * scale
    return out

def lora_k(sf):
    """Largest k so that header + 2k bytes fits in MAX_FRAGMENTS frames at this SF.
    Returns None when the dense int8 update already fits (no sparsification needed)."""
    budget = MAX_FRAGMENTS * MAX_FRAME[int(sf)]
    if budget >= DENSE_INT8_BYTES:
        return None
    return max((budget - OVERHEAD) // 2, 1)

def compress(vec, protocol, sf=None):
    """Returns (reconstructed vector as the server sees it, payload bytes, method name)."""
    if protocol == "WIFI":
        return vec.astype(np.float32).astype(float), OVERHEAD + 4 * N_PARAMS, "float32"
    if protocol == "LTE":
        return int8_dense(vec), DENSE_INT8_BYTES, "int8"
    k = lora_k(sf)
    if k is None:
        return int8_dense(vec), DENSE_INT8_BYTES, "int8"
    return topk_int8(vec, k), OVERHEAD + 2 * k, f"topk{k}_int8"