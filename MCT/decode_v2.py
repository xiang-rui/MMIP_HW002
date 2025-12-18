import argparse
import os
import numpy as np
from bitstream_v2 import read_header, read_table
from codec_v2 import decode_v2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .mmip")
    ap.add_argument("--output", required=True, help="path to output .npy (uint16)")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        h = read_header(f)
        table_entries = read_table(f, h["table_len"])
        payload = f.read(h["payload_len"])
        if len(payload) != h["payload_len"]:
            raise ValueError("Malformed stream: payload truncated")

    y = decode_v2(
        payload,
        table_entries=table_entries,
        width=h["width"], height=h["height"],
        padW=h["padW"], padH=h["padH"],
        blockN=h["blockN"], qstep=h["qstep"]
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, y)
    print(f"[decode_v2] wrote {args.output} shape={y.shape} dtype={y.dtype}")

if __name__ == "__main__":
    main()
