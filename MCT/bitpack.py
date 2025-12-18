class BitWriter:
    def __init__(self):
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0  # bits currently in _cur (0..7)

    def write_code(self, code: int, length: int):
        """Write 'length' bits of code (MSB-first)."""
        for i in range(length - 1, -1, -1):
            bit = (code >> i) & 1
            self._cur = (self._cur << 1) | bit
            self._nbits += 1
            if self._nbits == 8:
                self._buf.append(self._cur)
                self._cur = 0
                self._nbits = 0

    def finish(self) -> bytes:
        """Pad remaining bits with zeros."""
        if self._nbits > 0:
            self._buf.append(self._cur << (8 - self._nbits))
            self._cur = 0
            self._nbits = 0
        return bytes(self._buf)

class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.i = 0
        self.bit = 0  # bit index in current byte (0..7), MSB-first

    def read_bit(self) -> int:
        if self.i >= len(self.data):
            raise EOFError("Unexpected end of bitstream")
        b = (self.data[self.i] >> (7 - self.bit)) & 1
        self.bit += 1
        if self.bit == 8:
            self.bit = 0
            self.i += 1
        return b
