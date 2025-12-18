import struct
from typing import List, Tuple

MAGIC = b"MMIP"
VERSION = 2

# Header (little-endian):
# magic(4) version(1) flags(1) bitdepth(1) blockN(1)
# width(u16) height(u16) padW(u16) padH(u16) qstep(u16)
# table_len(u16) payload_len(u32)
HDR_FMT = "<4sBBBBHHHHHHI"
HDR_SIZE = struct.calcsize(HDR_FMT)

def write_header(f, *, flags: int, bitdepth: int, blockN: int,
                 width: int, height: int, padW: int, padH: int, qstep: int,
                 table_len: int, payload_len: int):
    f.write(struct.pack(
        HDR_FMT, MAGIC, VERSION, flags, bitdepth, blockN,
        width, height, padW, padH, qstep, table_len, payload_len
    ))

def read_header(f):
    data = f.read(HDR_SIZE)
    if len(data) != HDR_SIZE:
        raise ValueError("Malformed stream: header too short")
    magic, ver, flags, bitdepth, blockN, width, height, padW, padH, qstep, table_len, payload_len = struct.unpack(HDR_FMT, data)
    if magic != MAGIC:
        raise ValueError("Bad magic number (not MMIP)")
    if ver != VERSION:
        raise ValueError(f"Unsupported version: {ver}")
    return dict(
        flags=flags, bitdepth=bitdepth, blockN=blockN,
        width=width, height=height, padW=padW, padH=padH,
        qstep=qstep, table_len=table_len, payload_len=payload_len
    )

# Huffman table entry:
# run(u8) value(i16) codelen(u8)
TBL_FMT = "<Bhb"
TBL_SIZE = struct.calcsize(TBL_FMT)

def write_table(f, entries: List[Tuple[int, int, int]]):
    for run, val, L in entries:
        if not (0 <= run <= 255):
            raise ValueError("run out of range")
        if not (-32768 <= val <= 32767):
            raise ValueError("value out of int16 range")
        if not (1 <= L <= 31):
            raise ValueError("code length out of range (1..31)")
        f.write(struct.pack(TBL_FMT, run, int(val), int(L)))

def read_table(f, table_len: int):
    entries = []
    for _ in range(table_len):
        data = f.read(TBL_SIZE)
        if len(data) != TBL_SIZE:
            raise ValueError("Malformed stream: table truncated")
        run, val, L = struct.unpack(TBL_FMT, data)
        entries.append((int(run), int(val), int(L)))
    return entries
