import argparse
import os
import numpy as np
from codec_v2 import encode_v2
from bitstream_v2 import write_header, write_table

def quality_to_qstep(q: int) -> int:
    # 你可以之後在 report 中調參；先用穩定單調映射
    # q 越大 => qstep 越小 => 更高品質
    return max(1, int(round(220 / max(1, q))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .npy (uint16 2D)")
    ap.add_argument("--output", required=True, help="path to .mmip")
    ap.add_argument("--quality", required=True, type=int, help="quality knob (bigger=better)")
    ap.add_argument("--block", type=int, default=8, help="block size (default 8)")
    args = ap.parse_args()

    x = np.load(args.input)
    if x.dtype != np.uint16 or x.ndim != 2:
        raise ValueError("Input must be a 2D uint16 .npy array")

    qstep = quality_to_qstep(args.quality)

    table_entries, payload_bytes, meta = encode_v2(x, blockN=args.block, qstep=qstep)

    flags = 0
    bitdepth = 16
    height, width = x.shape
    padW, padH = meta["padW"], meta["padH"]
    table_len = len(table_entries)
    payload_len = len(payload_bytes)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        write_header(
            f,
            flags=flags, bitdepth=bitdepth, blockN=args.block,
            width=width, height=height, padW=padW, padH=padH,
            qstep=qstep, table_len=table_len, payload_len=payload_len
        )
        write_table(f, table_entries)
        f.write(payload_bytes)

    print(f"[encode_v2] wrote {args.output}")
    print(f"[encode_v2] shape={x.shape}, block={args.block}, qstep={qstep}")
    print(f"[encode_v2] table_len={table_len}, payload_len={payload_len} bytes")

if __name__ == "__main__":
    main()