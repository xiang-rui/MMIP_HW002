import numpy as np
from dct import dct_matrix, block_dct2, block_idct2
from zigzag import zigzag_scan, zigzag_unscan
from rle import rle_encode, rle_decode, EOB
from huff_canonical import build_code_lengths, canonical_codes_from_lengths, build_decode_trie, decode_one_symbol
from bitpack import BitWriter, BitReader
from phys_quant import block_stats, attenuation_scale, noise_scale, stage_freq_matrix, quantize_block_scale

def qmin_for_stage(stage_id: int) -> float:
    # 防溢位的最小量化步階（對 16-bit + 8x8 很安全）
    # DC 係數最容易爆，所以給更大的下界
    if stage_id == 0:
        return 16.0
    else:
        return 8.0

def pad_to_block(x: np.ndarray, N: int):
    H, W = x.shape
    padH = (N - (H % N)) % N
    padW = (N - (W % N)) % N
    if padH == 0 and padW == 0:
        return x, 0, 0
    xp = np.pad(x, ((0, padH), (0, padW)), mode="edge")
    return xp, padW, padH

def stage_ranges_for_8x8():
    # spectral selection over zigzag positions: [k0,k1)
    return [(0, 1), (1, 10), (10, 64)]

def stage_id_from_range(k0, k1):
    if (k0, k1) == (0, 1): return 0
    if (k0, k1) == (1, 10): return 1
    return 2

def encode_v4(x_u16: np.ndarray, *, blockN: int, qstep_bg: int, qstep_roi: int, block_roi_01: np.ndarray, sb_qscale: int = 16):
    """
    Returns:
      roi_map: uint8 0/1 (Hb,Wb)
      sb_q: uint8 (Hb,Wb) quantized block-scale
      stages: list {k0,k1, table_entries, payload_bytes}
      meta: padW,padH,Hb,Wb
    """
    assert x_u16.dtype == np.uint16 and x_u16.ndim == 2
    x_pad, padW, padH = pad_to_block(x_u16, blockN)
    H, W = x_pad.shape
    Hb, Wb = H // blockN, W // blockN
    if block_roi_01.shape != (Hb, Wb):
        raise ValueError(f"block_roi_01 mismatch: expected {(Hb,Wb)}, got {block_roi_01.shape}")

    # ---- Physics block scale (encoder side) ----
    mu_blk, sd_blk = block_stats(x_pad, blockN)
    s_att = attenuation_scale(mu_blk, tau=9000.0, kappa=1200.0, alpha=1.5)
    s_noise = noise_scale(mu_blk, sd_blk, lam=0.8, c=300.0)
    s_block = s_att * s_noise  # float (Hb,Wb)
    sb_q = quantize_block_scale(s_block, qscale=sb_qscale)  # uint8 map stored in bitstream
    # decoder uses sb = sb_q / sb_qscale
    sb = (sb_q.astype(np.float32) / float(sb_qscale))
    sb = np.clip(sb, 1.0, 1.6)

    C = dct_matrix(blockN)
    ranges = stage_ranges_for_8x8() if blockN == 8 else [(0, blockN*blockN)]

    stages = []
    # For each stage: build symbol stream + huffman + payload
    for (k0, k1) in ranges:
        sid = stage_id_from_range(k0, k1)
        M = stage_freq_matrix(blockN, sid)            # (N,N)
        Mzz = zigzag_scan(M).astype(np.float32)       # (N*N,)

        block_streams = []
        symbols = []

        for br in range(Hb):
            for bc in range(Wb):
                r = br * blockN
                c = bc * blockN
                blk = x_pad[r:r+blockN, c:c+blockN].astype(np.float32)
                coeff = block_dct2(blk, C)
                coeff_zz = zigzag_scan(coeff).astype(np.float32)

                # ROI base step (hard clinical priority) + physics soft scale + stage MTF weight
                qbase = float(qstep_roi if block_roi_01[br, bc] == 1 else qstep_bg)
                Qzz = (qbase * float(sb[br, bc])) * Mzz  # per-coefficient step
                Qzz = np.maximum(Qzz, qmin_for_stage(sid))

                zzq = np.zeros(blockN * blockN, dtype=np.int16)
                zzq[k0:k1] = np.round(coeff_zz[k0:k1] / Qzz[k0:k1]).astype(np.int16)

                pairs = rle_encode(zzq)
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
    return sb_q, stages, meta

def decode_v4(*, width, height, padW, padH, blockN, qstep_bg, qstep_roi,
              block_roi_01: np.ndarray, sb_q: np.ndarray, sb_qscale: int,
              stages_data, stages_to_decode: int):
    """
    stages_data: list of dict {k0,k1, table_entries, payload_bytes}
    sb_q: uint8 (Hb,Wb) stored map
    """
    Hp = height + padH
    Wp = width + padW
    Hb = Hp // blockN
    Wb = Wp // blockN
    if block_roi_01.shape != (Hb, Wb):
        raise ValueError("ROI map shape mismatch in decode")
    if sb_q.shape != (Hb, Wb):
        raise ValueError("block-scale map shape mismatch in decode")

    sb = (sb_q.astype(np.float32) / float(sb_qscale))

    C = dct_matrix(blockN)
    nb = Hb * Wb
    K = blockN * blockN
    zz_acc = np.zeros((nb, K), dtype=np.int16)

    nstages = len(stages_data)
    n = max(1, min(stages_to_decode, nstages))

    # Decode each stage, fill its coefficient subset
    for si in range(n):
        st = stages_data[si]
        k0, k1 = st["k0"], st["k1"]
        sid = stage_id_from_range(k0, k1)
        # NOTE: quantization is applied in final reconstruction, not here.
        # Here we only recover integer q-coeffs in this stage range.
        lengths = {(run, val): L for (run, val, L) in st["table_entries"]}
        codes = canonical_codes_from_lengths(lengths)
        trie = build_decode_trie(codes)
        br = BitReader(st["payload_bytes"])

        for bi in range(nb):
            pairs = []
            while True:
                sym = decode_one_symbol(trie, br)
                pairs.append(sym)
                if sym == EOB:
                    break
                if len(pairs) > K + 1:
                    raise ValueError("Corrupt stream: too many symbols in block")
            vec = rle_decode(pairs, K)
            zz_acc[bi, k0:k1] = vec[k0:k1]

    # Reconstruct spatial image: apply stage-specific Qzz to the pieces we decoded
    out = np.zeros((Hp, Wp), dtype=np.float32)
    bi = 0
    for br in range(Hb):
        for bc in range(Wb):
            # Start with empty coefficient zigzag vector (float)
            coeff_zz = np.zeros(K, dtype=np.float32)
            qbase = float(qstep_roi if block_roi_01[br, bc] == 1 else qstep_bg)
            sbv = float(sb[br, bc])

            # For each decoded stage, apply its own MTF matrix
            for si in range(n):
                st = stages_data[si]
                k0, k1 = st["k0"], st["k1"]

                # 注意：decoder 不再使用 MTF / stage 權重
                # 只使用 encoder 已決定好的 base quantization scale
                Qbase = qbase * sbv

                coeff_zz[k0:k1] = (
                        zz_acc[bi, k0:k1].astype(np.float32) * Qbase
                )

            coeff = zigzag_unscan(coeff_zz, blockN).astype(np.float32)
            blk = block_idct2(coeff, C)

            r = br * blockN
            c = bc * blockN
            out[r:r+blockN, c:c+blockN] = blk
            bi += 1

    out = np.clip(out, 0, 65535).astype(np.uint16)
    return out[:height, :width]