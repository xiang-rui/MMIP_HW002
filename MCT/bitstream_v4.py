import struct
from typing import List, Tuple

MAGIC = b"MMIP"
VERSION = 4

# Main header (little-endian):
# magic(4) ver(1) flags(1) bitdepth(1) blockN(1)
# width(u16) height(u16) padW(u16) padH(u16)
# qstep_bg(u16) qstep_roi(u16)
# roi_bits(u32) roi_bytes(u32)
# sb_qscale(u16) sb_bytes(u32)   # block-scale quantization factor + byte length
# nstages(u8) reserved(3)
HDR_FMT = "<4sBBBBHHHHHHIIH I B3s"
HDR_SIZE = struct.calcsize(HDR_FMT)

# Stage header:
# k0(u8) k1(u8) table_len(u16) payload_len(u32)
STG_FMT = "<BBHI"
STG_SIZE = struct.calcsize(STG_FMT)

# Huffman table entry:
# run(u8) value(i16) codelen(u8)
TBL_FMT = "<Bhb"
TBL_SIZE = struct.calcsize(TBL_FMT)

def write_header(
    f, *, flags, bitdepth, blockN, width, height, padW, padH,
    qstep_bg, qstep_roi,
    roi_bits, roi_bytes,
    sb_qscale, sb_bytes,
    nstages
):
    f.write(struct.pack(
        HDR_FMT, MAGIC, VERSION, flags, bitdepth, blockN,
        width, height, padW, padH,
        qstep_bg, qstep_roi,
        roi_bits, roi_bytes,
        sb_qscale, sb_bytes,
        nstages, b"\x00\x00\x00"
    ))

def read_header(f):
    data = f.read(HDR_SIZE)
    if len(data) != HDR_SIZE:
        raise ValueError("Malformed stream: header too short")
    (magic, ver, flags, bitdepth, blockN,
     width, height, padW, padH,
     qbg, qroi,
     roi_bits, roi_bytes,
     sb_qscale, sb_bytes,
     nstages, _) = struct.unpack(HDR_FMT, data)
    if magic != MAGIC:
        raise ValueError("Bad magic")
    if ver != VERSION:
        raise ValueError(f"Unsupported version: {ver}")
    return dict(
        flags=flags, bitdepth=bitdepth, blockN=blockN,
        width=width, height=height, padW=padW, padH=padH,
        qstep_bg=qbg, qstep_roi=qroi,
        roi_bits=roi_bits, roi_bytes=roi_bytes,
        sb_qscale=sb_qscale, sb_bytes=sb_bytes,
        nstages=nstages
    )

def write_stage_header(f, k0: int, k1: int, table_len: int, payload_len: int):
    f.write(struct.pack(STG_FMT, k0, k1, table_len, payload_len))

def read_stage_header(f):
    data = f.read(STG_SIZE)
    if len(data) != STG_SIZE:
        raise ValueError("Malformed stream: stage header truncated")
    k0, k1, table_len, payload_len = struct.unpack(STG_FMT, data)
    return dict(k0=k0, k1=k1, table_len=table_len, payload_len=payload_len)

def write_table(f, entries: List[Tuple[int, int, int]]):
    for run, val, L in entries:
        if not (0 <= run <= 255): raise ValueError("run out of range")
        if not (-32768 <= val <= 32767): raise ValueError("value out of range")
        if not (1 <= L <= 31): raise ValueError("codelen out of range")
        f.write(struct.pack(TBL_FMT, run, int(val), int(L)))

def read_table(f, table_len: int):
    out = []
    for _ in range(table_len):
        data = f.read(TBL_SIZE)
        if len(data) != TBL_SIZE:
            raise ValueError("Malformed stream: table truncated")
        run, val, L = struct.unpack(TBL_FMT, data)
        out.append((int(run), int(val), int(L)))
    return out
