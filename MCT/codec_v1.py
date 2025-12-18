import numpy as np
from dct import dct_matrix, block_dct2, block_idct2
from zigzag import zigzag_scan, zigzag_unscan

def pad_to_block(x: np.ndarray, N: int):
    H, W = x.shape
    padH = (N - (H % N)) % N
    padW = (N - (W % N)) % N
    if padH == 0 and padW == 0:
        return x, 0, 0
    xp = np.pad(x, ((0, padH), (0, padW)), mode="edge")
    return xp, padW, padH

def encode_v1(x_u16: np.ndarray, blockN: int, qstep: int):
    """
    Returns:
      payload: int16 array of shape (num_blocks, blockN*blockN) in zigzag order
      meta: dict
    """
    assert x_u16.dtype == np.uint16
    x, padW, padH = pad_to_block(x_u16, blockN)

    C = dct_matrix(blockN)
    H, W = x.shape
    blocks = []
    for r in range(0, H, blockN):
        for c in range(0, W, blockN):
            blk = x[r:r+blockN, c:c+blockN].astype(np.float32)
            coeff = block_dct2(blk, C)

            # uniform scalar quantization
            q = np.round(coeff / float(qstep)).astype(np.int16)

            zz = zigzag_scan(q)  # int16 length N*N
            blocks.append(zz)

    payload = np.stack(blocks, axis=0)  # (B, N*N)
    meta = {"padW": padW, "padH": padH}
    return payload, meta

def decode_v1(payload_i16: np.ndarray, *, width: int, height: int, padW: int, padH: int, blockN: int, qstep: int):
    C = dct_matrix(blockN)
    H = height + padH
    W = width + padW
    out = np.zeros((H, W), dtype=np.float32)

    idx = 0
    for r in range(0, H, blockN):
        for c in range(0, W, blockN):
            zz = payload_i16[idx]
            idx += 1
            qblk = zigzag_unscan(zz, blockN).astype(np.float32)
            coeff = qblk * float(qstep)
            blk = block_idct2(coeff, C)
            out[r:r+blockN, c:c+blockN] = blk

    out = np.clip(out, 0, 65535).astype(np.uint16)
    # crop padding
    out = out[:height, :width]
    return out
