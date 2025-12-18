import numpy as np

def zigzag_indices(N: int):
    idx = []
    for s in range(2*N - 1):
        if s % 2 == 0:
            # even: go down-left
            r = min(s, N-1)
            c = s - r
            while r >= 0 and c < N:
                idx.append((r, c))
                r -= 1
                c += 1
        else:
            # odd: go up-right
            c = min(s, N-1)
            r = s - c
            while c >= 0 and r < N:
                idx.append((r, c))
                r += 1
                c -= 1
    return idx

def zigzag_scan(block: np.ndarray) -> np.ndarray:
    N = block.shape[0]
    idx = zigzag_indices(N)
    return np.array([block[r, c] for r, c in idx], dtype=block.dtype)

def zigzag_unscan(vec: np.ndarray, N: int) -> np.ndarray:
    idx = zigzag_indices(N)
    out = np.zeros((N, N), dtype=vec.dtype)
    for k, (r, c) in enumerate(idx):
        out[r, c] = vec[k]
    return out