import numpy as np

def dct_matrix(N: int) -> np.ndarray:
    C = np.zeros((N, N), dtype=np.float32)
    alpha0 = np.sqrt(1.0 / N)
    alpha = np.sqrt(2.0 / N)
    for k in range(N):
        for n in range(N):
            C[k, n] = (alpha0 if k == 0 else alpha) * np.cos(np.pi * (2*n + 1) * k / (2*N))
    return C

def block_dct2(block: np.ndarray, C: np.ndarray) -> np.ndarray:
    # DCT:  C * x * C^T
    return C @ block @ C.T

def block_idct2(coeff: np.ndarray, C: np.ndarray) -> np.ndarray:
    # IDCT: C^T * X * C
    return C.T @ coeff @ C
