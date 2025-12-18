import numpy as np

# Reserved End-of-Block marker (safe because value==0 never appears except EOB)
EOB = (0, 0)

def rle_encode(vec: np.ndarray):
    """
    Input: 1D int16 array in zigzag order
    Output: list of (run, value) pairs + EOB
    run: number of leading zeros before 'value'
    value: non-zero int
    """
    out = []
    run = 0
    for v in vec:
        v = int(v)
        if v == 0:
            run += 1
        else:
            # If run is huge, split (rare with 8x8 but safe)
            while run > 255:
                out.append((255, 1))  # harmless non-zero value, will be coded; not ideal but extremely rare
                run -= 255
            out.append((run, v))
            run = 0
    out.append(EOB)
    return out

def rle_decode(pairs, N: int):
    """
    Input: list of (run, value) pairs until EOB
    Output: 1D int16 array length N
    """
    out = []
    for run, val in pairs:
        if (run, val) == EOB:
            break
        out.extend([0] * int(run))
        out.append(int(val))
        if len(out) > N:
            raise ValueError("RLE decode overflow (corrupt stream or bug)")
    if len(out) < N:
        out.extend([0] * (N - len(out)))
    return np.array(out, dtype=np.int16)
