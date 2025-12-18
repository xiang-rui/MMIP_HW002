import numpy as np
from dct import dct_matrix, block_dct2, block_idct2
from zigzag import zigzag_scan, zigzag_unscan
from rle import rle_encode, rle_decode, EOB
from huff_canonical import build_code_lengths, canonical_codes_from_lengths, build_decode_trie, decode_one_symbol
from bitpack import BitWriter, BitReader

def pad_to_block(x: np.ndarray, N: int):
    H, W = x.shape
    padH = (N - (H % N)) % N
    padW = (N - (W % N)) % N
    if padH == 0 and padW == 0:
        return x, 0, 0
    xp = np.pad(x, ((0, padH), (0, padW)), mode="edge")
    return xp, padW, padH

def stage_ranges_for_8x8():
    # (k0 inclusive, k1 exclusive) in zigzag vector positions
    return [(0, 1), (1, 10), (10, 64)]

def encode_v3(x_u16: np.ndarray, *, blockN: int, qstep_bg: int, qstep_roi: int, block_roi_01: np.ndarray):
    """
    Returns:
      stages: list of dict {k0,k1, table_entries, payload_bytes}
      meta: padW,padH,Hb,Wb
    """
    assert x_u16.dtype == np.uint16 and x_u16.ndim == 2
    x, padW, padH = pad_to_block(x_u16, blockN)
    H, W = x.shape
    Hb = H // blockN
    Wb = W // blockN
    if block_roi_01.shape != (Hb, Wb):
        raise ValueError(f"block_roi shape mismatch: expected {(Hb,Wb)}, got {block_roi_01.shape}")

    C = dct_matrix(blockN)
    ranges = stage_ranges_for_8x8() if blockN == 8 else [(0, blockN*blockN)]

    # Precompute all quantized zigzag vectors per block with ROI-aware qstep
    zz_all = []
    for br in range(Hb):
        for bc in range(Wb):
            r = br * blockN
            c = bc * blockN
            blk = x[r:r+blockN, c:c+blockN].astype(np.float32)
            coeff = block_dct2(blk, C)
            qstep = qstep_roi if block_roi_01[br, bc] == 1 else qstep_bg
            q = np.round(coeff / float(qstep)).astype(np.int16)
            zz = zigzag_scan(q)
            zz_all.append(zz)
    zz_all = np.stack(zz_all, axis=0)  # (nb, 64)

    stages = []
    for (k0, k1) in ranges:
        # Build symbols for this stage only: vectors with others forced to zero
        block_streams = []
        symbols = []
        for i in range(zz_all.shape[0]):
            vec = np.zeros_like(zz_all[i])
            vec[k0:k1] = zz_all[i][k0:k1]
            pairs = rle_encode(vec)
            block_streams.append(pairs)
            symbols.extend(pairs)
        if len(symbols) == 0:
            symbols = [EOB]

        lengths = build_code_lengths(symbols)
        codes = canonical_codes_from_lengths(lengths)

        bw = BitWriter()
        for pairs in block_streams:
            for sym in pairs:
                code, L = codes[sym]
                bw.write_code(code, L)
        payload_bytes = bw.finish()
        table_entries = [(run, val, lengths[(run, val)]) for (run, val) in lengths.keys()]

        stages.append(dict(k0=k0, k1=k1, table_entries=table_entries, payload_bytes=payload_bytes))

    meta = dict(padW=padW, padH=padH, Hb=Hb, Wb=Wb)
    return stages, meta

def decode_v3(*, stages_data, width, height, padW, padH, blockN, qstep_bg, qstep_roi, block_roi_01, stages_to_decode: int):
    """
    stages_data: list of dict {k0,k1, table_entries, payload_bytes} length=nstages
    stages_to_decode: decode first N stages (1..nstages)
    """
    Hp = height + padH
    Wp = width + padW
    Hb = Hp // blockN
    Wb = Wp // blockN
    if block_roi_01.shape != (Hb, Wb):
        raise ValueError("block_roi shape mismatch in decode")

    C = dct_matrix(blockN)
    nb = Hb * Wb
    coeffs_per_block = blockN * blockN

    # Accumulate quantized coefficients in zigzag domain
    zz_acc = np.zeros((nb, coeffs_per_block), dtype=np.int16)

    for si in range(stages_to_decode):
        st = stages_data[si]
        k0, k1 = st["k0"], st["k1"]
        table_entries = st["table_entries"]
        payload_bytes = st["payload_bytes"]

        lengths = {(run, val): L for (run, val, L) in table_entries}
        codes = canonical_codes_from_lengths(lengths)
        trie = build_decode_trie(codes)
        br = BitReader(payload_bytes)

        for bi in range(nb):
            pairs = []
            while True:
                sym = decode_one_symbol(trie, br)
                pairs.append(sym)
                if sym == EOB:
                    break
                if len(pairs) > coeffs_per_block + 1:
                    raise ValueError("Corrupt stream: too many symbols in block")
            vec = rle_decode(pairs, coeffs_per_block)
            zz_acc[bi, k0:k1] = vec[k0:k1]

    # Reconstruct spatial image block-by-block with ROI-aware inverse scaling
    out = np.zeros((Hp, Wp), dtype=np.float32)
    bi = 0
    for br in range(Hb):
        for bc in range(Wb):
            qstep = qstep_roi if block_roi_01[br, bc] == 1 else qstep_bg
            qblk = zigzag_unscan(zz_acc[bi], blockN).astype(np.float32)
            bi += 1
            coeff = qblk * float(qstep)
            blk = block_idct2(coeff, C)
            r = br * blockN
            c = bc * blockN
            out[r:r+blockN, c:c+blockN] = blk

    out = np.clip(out, 0, 65535).astype(np.uint16)
    return out[:height, :width]
