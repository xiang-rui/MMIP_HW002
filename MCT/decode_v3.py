import argparse, os
import numpy as np
from bitstream_v3 import read_header, read_stage_header, read_table
from roi import unpack_bits_u8
from codec_v3 import decode_v3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .mmip (v3)")
    ap.add_argument("--output", required=True, help="path to output .npy")
    ap.add_argument("--stages", type=int, default=3, help="decode first N stages (1..3)")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        h = read_header(f)

        roi_bytes = f.read(h["roi_map_bytes"])
        if len(roi_bytes) != h["roi_map_bytes"]:
            raise ValueError("Malformed stream: ROI map truncated")
        roi_flat = unpack_bits_u8(roi_bytes, h["roi_map_bits"])

        Hp = h["height"] + h["padH"]
        Wp = h["width"] + h["padW"]
        Hb = Hp // h["blockN"]
        Wb = Wp // h["blockN"]
        roi_blk = roi_flat.reshape(Hb, Wb).astype(np.uint8)

        stages_data = []
        for _ in range(h["nstages"]):
            sh = read_stage_header(f)
            tbl = read_table(f, sh["table_len"])
            payload = f.read(sh["payload_len"])
            if len(payload) != sh["payload_len"]:
                raise ValueError("Malformed stream: stage payload truncated")
            stages_data.append(dict(k0=sh["k0"], k1=sh["k1"], table_entries=tbl, payload_bytes=payload))

    n = max(1, min(args.stages, h["nstages"]))
    y = decode_v3(
        stages_data=stages_data,
        width=h["width"], height=h["height"],
        padW=h["padW"], padH=h["padH"],
        blockN=h["blockN"],
        qstep_bg=h["qstep_bg"], qstep_roi=h["qstep_roi"],
        block_roi_01=roi_blk,
        stages_to_decode=n
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, y)
    print(f"[decode_v3] wrote {args.output} shape={y.shape} dtype={y.dtype} stages={n}/{h['nstages']}")

if __name__ == "__main__":
    main()
