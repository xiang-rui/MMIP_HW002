import argparse, os
import numpy as np
from roi import roi_mask_from_phantom, block_roi_map, pack_bits_u8
from codec_v4 import encode_v4
from bitstream_v4 import write_header, write_stage_header, write_table

def quality_to_qsteps(q: int):
    base = max(1, int(round(220 / max(1, q))))
    q_roi = max(1, base // 2)
    q_bg  = max(1, base * 2)
    return q_bg, q_roi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .npy (uint16 2D)")
    ap.add_argument("--output", required=True, help="path to .mmip (v4)")
    ap.add_argument("--quality", required=True, type=int)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--bone_threshold", type=int, default=9000)
    ap.add_argument("--sb_qscale", type=int, default=16, help="block-scale quant factor (default 16)")
    args = ap.parse_args()

    x = np.load(args.input)
    if x.dtype != np.uint16 or x.ndim != 2:
        raise ValueError("Input must be a 2D uint16 .npy array")

    H, W = x.shape
    blockN = args.block

    # padding sizes (must match codec padding mode=edge)
    padH = (blockN - (H % blockN)) % blockN
    padW = (blockN - (W % blockN)) % blockN

    # ROI map from phantom threshold on padded mask
    roi_pix = roi_mask_from_phantom(x, bone_threshold=args.bone_threshold)
    if padH or padW:
        roi_pix = np.pad(roi_pix, ((0, padH), (0, padW)), mode="edge")
    roi_blk = block_roi_map(roi_pix, blockN).astype(np.uint8)  # (Hb,Wb)

    roi_bits = roi_blk.size
    roi_bytes = pack_bits_u8(roi_blk)

    q_bg, q_roi = quality_to_qsteps(args.quality)

    sb_q, stages, meta = encode_v4(
        x, blockN=blockN,
        qstep_bg=q_bg, qstep_roi=q_roi,
        block_roi_01=roi_blk,
        sb_qscale=args.sb_qscale
    )
    sb_bytes = sb_q.tobytes(order="C")  # 1 byte per block

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        write_header(
            f,
            flags=0, bitdepth=16, blockN=blockN,
            width=W, height=H, padW=meta["padW"], padH=meta["padH"],
            qstep_bg=q_bg, qstep_roi=q_roi,
            roi_bits=roi_bits, roi_bytes=len(roi_bytes),
            sb_qscale=args.sb_qscale, sb_bytes=len(sb_bytes),
            nstages=len(stages)
        )
        f.write(roi_bytes)
        f.write(sb_bytes)

        for st in stages:
            write_stage_header(f, st["k0"], st["k1"], len(st["table_entries"]), len(st["payload_bytes"]))
            write_table(f, st["table_entries"])
            f.write(st["payload_bytes"])

    print(f"[encode_v4] wrote {args.output}")
    print(f"[encode_v4] q_bg={q_bg}, q_roi={q_roi}, sb_qscale={args.sb_qscale}")
    print(f"[encode_v4] ROI blocks={roi_bits}, sb_bytes={len(sb_bytes)}")
    for i, st in enumerate(stages):
        print(f"[encode_v4] stage{i}: k[{st['k0']}:{st['k1']}) table={len(st['table_entries'])} payload={len(st['payload_bytes'])}B")

if __name__ == "__main__":
    main()
