import argparse, os
import numpy as np
from roi import roi_mask_from_phantom, block_roi_map, pack_bits_u8
from codec_v3 import encode_v3
from bitstream_v3 import write_header, write_stage_header, write_table

def quality_to_qsteps(q: int):
    # q 越大 => 越高品質 => qstep 越小
    base = max(1, int(round(220 / max(1, q))))
    # ROI 給更多 bits：更小 qstep；背景更大 qstep
    q_roi = max(1, base // 2)
    q_bg  = max(1, base * 2)
    return q_bg, q_roi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .npy (uint16 2D)")
    ap.add_argument("--output", required=True, help="path to .mmip")
    ap.add_argument("--quality", required=True, type=int)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--bone_threshold", type=int, default=9000, help="phantom bone threshold for ROI")
    args = ap.parse_args()

    x = np.load(args.input)
    if x.dtype != np.uint16 or x.ndim != 2:
        raise ValueError("Input must be 2D uint16 .npy")

    q_bg, q_roi = quality_to_qsteps(args.quality)
    # q_bg, q_roi = quality_to_qsteps(args.quality)
    q_roi = q_bg

    # ROI map from phantom
    roi_pix = roi_mask_from_phantom(x, bone_threshold=args.bone_threshold)
    # Note: encode_v3 internally pads; we need ROI on padded shape too
    # simplest: build ROI on padded by edge padding consistent with codec
    H, W = x.shape
    padH = (args.block - (H % args.block)) % args.block
    padW = (args.block - (W % args.block)) % args.block
    if padH or padW:
        roi_pix = np.pad(roi_pix, ((0, padH), (0, padW)), mode="edge")

    roi_blk = block_roi_map(roi_pix, args.block)  # (Hb,Wb) uint8
    roi_bits = roi_blk.size
    roi_bytes = pack_bits_u8(roi_blk)

    stages, meta = encode_v3(
        x, blockN=args.block,
        qstep_bg=q_bg, qstep_roi=q_roi,
        block_roi_01=roi_blk
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        write_header(
            f,
            flags=0, bitdepth=16, blockN=args.block,
            width=W, height=H, padW=meta["padW"], padH=meta["padH"],
            qstep_bg=q_bg, qstep_roi=q_roi,
            roi_map_bits=roi_bits, roi_map_bytes=len(roi_bytes),
            nstages=len(stages)
        )
        f.write(roi_bytes)

        for st in stages:
            table_entries = st["table_entries"]
            payload_bytes = st["payload_bytes"]
            write_stage_header(f, st["k0"], st["k1"], len(table_entries), len(payload_bytes))
            write_table(f, table_entries)
            f.write(payload_bytes)

    print(f"[encode_v3] wrote {args.output}")
    print(f"[encode_v3] q_bg={q_bg}, q_roi={q_roi}, roi_blocks={roi_bits}")
    for i, st in enumerate(stages):
        print(f"[encode_v3] stage{i}: k[{st['k0']}:{st['k1']}) table={len(st['table_entries'])} payload={len(st['payload_bytes'])}B")

if __name__ == "__main__":
    main()
