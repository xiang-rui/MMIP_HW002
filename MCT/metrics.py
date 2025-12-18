import numpy as np

def rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return float(np.sqrt(np.mean((x - y) ** 2)))

def psnr(x: np.ndarray, y: np.ndarray, bit_depth: int) -> float:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mse = float(np.mean((x - y) ** 2))
    if mse == 0.0:
        return float("inf")
    maxv = (2 ** bit_depth) - 1
    return float(20.0 * np.log10(maxv) - 10.0 * np.log10(mse))
