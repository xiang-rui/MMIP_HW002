import struct

MAGIC = b"MMIP"   # 4 bytes
VERSION = 1       # 1 byte

# Header (little-endian):
# magic(4) version(1) flags(1) bitdepth(1) blockN(1)
# width(u16) height(u16)
# padW(u16) padH(u16)
# qstep(u16)
HEADER_FMT = "<4sBBBBHHHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

def write_header(f, *, flags, bitdepth, blockN, width, height, padW, padH, qstep):
    data = struct.pack(
        HEADER_FMT,
        MAGIC, VERSION, flags, bitdepth, blockN,
        width, height, padW, padH, qstep
    )
    f.write(data)

def read_header(f):
    data = f.read(HEADER_SIZE)
    if len(data) != HEADER_SIZE:
        raise ValueError("Malformed stream: header too short")
    magic, ver, flags, bitdepth, blockN, width, height, padW, padH, qstep = struct.unpack(HEADER_FMT, data)
    if magic != MAGIC:
        raise ValueError("Bad magic number (not MMIP)")
    if ver != VERSION:
        raise ValueError(f"Unsupported version: {ver}")
    return {
        "flags": flags,
        "bitdepth": bitdepth,
        "blockN": blockN,
        "width": width,
        "height": height,
        "padW": padW,
        "padH": padH,
        "qstep": qstep,
    }