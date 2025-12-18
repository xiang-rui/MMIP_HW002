import argparse
import numpy as np
from codec_v1 import encode_v1
from bitstream import write_header

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

    # simple mapping: quality -> qstep (you can tune later)
    # higher quality => smaller qstep
    qstep = max(1, int(round(200 / args.quality)))

    payload, meta = encode_v1(x, blockN=args.block, qstep=qstep)

    flags = 0
    bitdepth = 16
    height, width = x.shape
    padW, padH = meta["padW"], meta["padH"]

    with open(args.output, "wb") as f:
        write_header(
            f,
            flags=flags, bitdepth=bitdepth, blockN=args.block,
            width=width, height=height, padW=padW, padH=padH,
            qstep=qstep
        )
        # payload stored as little-endian int16
        payload.astype("<i2").tofile(f)

    print(f"[encode] wrote {args.output}")
    print(f"[encode] shape={x.shape}, block={args.block}, qstep={qstep}, blocks={payload.shape[0]}")

if __name__ == "__main__":
    main()
