import numpy as np

def roi_mask_from_phantom(x_u16: np.ndarray, bone_threshold: int = 9000) -> np.ndarray:
    """
    ROI pixel mask (bool) from phantom: bone region as ROI.
    """
    return x_u16 >= np.uint16(bone_threshold)

def block_roi_map(roi_mask: np.ndarray, blockN: int) -> np.ndarray:
    """
    Convert pixel ROI mask -> block ROI map (H_blocks, W_blocks) bool
    Rule: a block is ROI if ANY pixel in block is ROI.
    """
    H, W = roi_mask.shape
    Hb = (H + blockN - 1) // blockN
    Wb = (W + blockN - 1) // blockN
    out = np.zeros((Hb, Wb), dtype=np.uint8)

    for br in range(Hb):
        for bc in range(Wb):
            r0 = br * blockN
            c0 = bc * blockN
            blk = roi_mask[r0:r0+blockN, c0:c0+blockN]
            out[br, bc] = 1 if np.any(blk) else 0
    return out  # uint8 0/1

def pack_bits_u8(bits01: np.ndarray) -> bytes:
    """
    Pack a flat uint8 array of 0/1 into bytes (MSB-first).
    """
    b = bits01.astype(np.uint8).ravel()
    out = bytearray()
    cur = 0
    cnt = 0
    for v in b:
        cur = (cur << 1) | int(v)
        cnt += 1
        if cnt == 8:
            out.append(cur)
            cur = 0
            cnt = 0
    if cnt > 0:
        out.append(cur << (8 - cnt))
    return bytes(out)

def unpack_bits_u8(data: bytes, nbits: int) -> np.ndarray:
    """
    Unpack bytes -> uint8 0/1 array length nbits (MSB-first).
    """
    out = np.zeros(nbits, dtype=np.uint8)
    k = 0
    for byte in data:
        for i in range(7, -1, -1):
            if k >= nbits:
                return out
            out[k] = (byte >> i) & 1
            k += 1
    return out
