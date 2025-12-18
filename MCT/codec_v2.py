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

def encode_v2(x_u16: np.ndarray, blockN: int, qstep: int):
    """
    Returns:
      table_entries: list of (run, val, codelen)
      payload_bytes: bytes
      meta: dict (padW, padH, nb)
    """
    assert x_u16.dtype == np.uint16 and x_u16.ndim == 2
    x, padW, padH = pad_to_block(x_u16, blockN)
    H, W = x.shape
    nb = (H // blockN) * (W // blockN)

    C = dct_matrix(blockN)

    # 1) Build symbol stream (RLE pairs per block)
    symbols = []  # list of (run,val)
    block_streams = []  # store per-block list of symbols to re-encode after building codebook
    for r in range(0, H, blockN):
        for c in range(0, W, blockN):
            blk = x[r:r+blockN, c:c+blockN].astype(np.float32)
            coeff = block_dct2(blk, C)
            q = np.round(coeff / float(qstep)).astype(np.int16)
            zz = zigzag_scan(q)
            pairs = rle_encode(zz)
            block_streams.append(pairs)
            symbols.extend(pairs)

    # Ensure EOB exists at least
    if len(symbols) == 0:
        symbols = [EOB]

    # 2) Huffman lengths + canonical codes
    lengths = build_code_lengths(symbols)
    codes = canonical_codes_from_lengths(lengths)  # sym -> (code_int, L)

    # 3) Pack payload bits block-by-block
    bw = BitWriter()
    for pairs in block_streams:
        for sym in pairs:
            code, L = codes[sym]
            bw.write_code(code, L)
    payload_bytes = bw.finish()

    # 4) Export table entries (run,val,length)
    # canonical requires only code lengths; decoder can rebuild codes
    table_entries = [(run, val, lengths[(run, val)]) for (run, val) in lengths.keys()]

    meta = {"padW": padW, "padH": padH, "nb": nb}
    return table_entries, payload_bytes, meta

def decode_v2(payload_bytes: bytes, *, table_entries, width, height, padW, padH, blockN, qstep):
    """
    table_entries: list of (run, val, codelen)
    """
    # 1) Rebuild canonical codes from lengths
    lengths = {(run, val): L for (run, val, L) in table_entries}
    codes = canonical_codes_from_lengths(lengths)
    trie = build_decode_trie(codes)

    Hp = height + padH
    Wp = width + padW
    nb = (Hp // blockN) * (Wp // blockN)
    coeffs_per_block = blockN * blockN

    C = dct_matrix(blockN)
    br = BitReader(payload_bytes)

    out = np.zeros((Hp, Wp), dtype=np.float32)

    # 2) Decode each block until EOB
    idx_block = 0
    for r in range(0, Hp, blockN):
        for c in range(0, Wp, blockN):
            pairs = []
            # decode symbols until EOB
            while True:
                sym = decode_one_symbol(trie, br)
                pairs.append(sym)
                if sym == EOB:
                    break
                # basic safety (should never exceed coeff count)
                if len(pairs) > coeffs_per_block + 1:
                    raise ValueError("Corrupt stream: too many symbols in a block")

            zz = rle_decode(pairs, coeffs_per_block)
            qblk = zigzag_unscan(zz, blockN).astype(np.float32)
            coeff = qblk * float(qstep)
            blk = block_idct2(coeff, C)
            out[r:r+blockN, c:c+blockN] = blk

            idx_block += 1
            if idx_block > nb:
                raise ValueError("Corrupt stream: too many blocks decoded")

    out = np.clip(out, 0, 65535).astype(np.uint16)
    return out[:height, :width]
