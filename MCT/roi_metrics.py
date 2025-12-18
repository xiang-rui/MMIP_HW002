import numpy as np

def roi_psnr(x: np.ndarray, y: np.ndarray, roi_mask: np.ndarray, bit_depth: int):
    """
    Compute PSNR only over ROI pixels.
    x, y: uint16 images
    roi_mask: bool or 0/1 mask, same shape
    """
    assert x.shape == y.shape == roi_mask.shape
    roi = roi_mask.astype(bool)

    if roi.sum() == 0:
        raise ValueError("ROI mask is empty")

    diff = x.astype(np.float32)[roi] - y.astype(np.float32)[roi]
    mse = float((diff ** 2).mean())
    if mse == 0:
        return float("inf")

    maxv = (2 ** bit_depth) - 1
    return float(20 * np.log10(maxv) - 10 * np.log10(mse))
