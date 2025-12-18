import argparse
import numpy as np
from bitstream import read_header
from codec_v1 import decode_v1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .mmip")
    ap.add_argument("--output", required=True, help="path to output .npy (uint16)")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        h = read_header(f)
        blockN = h["blockN"]
        width, height = h["width"], h["height"]
        padW, padH = h["padW"], h["padH"]
        qstep = h["qstep"]

        # compute number of blocks
        Hp = height + padH
        Wp = width + padW
        nb = (Hp // blockN) * (Wp // blockN)
        coeffs_per_block = blockN * blockN

        payload = np.fromfile(f, dtype="<i2", count=nb * coeffs_per_block)
        if payload.size != nb * coeffs_per_block:
            raise ValueError("Malformed stream: payload too short")
        payload = payload.reshape(nb, coeffs_per_block)

    y = decode_v1(payload, width=width, height=height, padW=padW, padH=padH, blockN=blockN, qstep=qstep)
    np.save(args.output, y)
    print(f"[decode] wrote {args.output} shape={y.shape} dtype={y.dtype}")

if __name__ == "__main__":
    main()
