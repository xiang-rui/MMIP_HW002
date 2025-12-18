from __future__ import annotations
import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

Symbol = Tuple[int, int]  # (run, value)

@dataclass
class _Node:
    freq: int
    sym: Optional[Symbol] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    def __lt__(self, other):  # for heapq
        return self.freq < other.freq

def _build_tree(freqs: Dict[Symbol, int]) -> _Node:
    pq = [_Node(freq=f, sym=s) for s, f in freqs.items()]
    heapq.heapify(pq)
    if len(pq) == 1:
        # Edge case: only one symbol -> give it length 1
        only = pq[0]
        return _Node(freq=only.freq, left=only, right=_Node(freq=0, sym=None))
    while len(pq) > 1:
        a = heapq.heappop(pq)
        b = heapq.heappop(pq)
        heapq.heappush(pq, _Node(freq=a.freq + b.freq, left=a, right=b))
    return pq[0]

def _collect_lengths(node: _Node, depth: int, out: Dict[Symbol, int]):
    if node.sym is not None:
        out[node.sym] = max(1, depth)  # avoid 0-length
        return
    _collect_lengths(node.left, depth + 1, out)
    _collect_lengths(node.right, depth + 1, out)

def build_code_lengths(symbols: List[Symbol]) -> Dict[Symbol, int]:
    freqs = Counter(symbols)
    tree = _build_tree(freqs)
    lengths: Dict[Symbol, int] = {}
    _collect_lengths(tree, 0, lengths)
    return lengths

def canonical_codes_from_lengths(lengths: Dict[Symbol, int]) -> Dict[Symbol, Tuple[int, int]]:
    """
    Return mapping: sym -> (code_int, code_len), canonical Huffman.
    Canonical ordering: sort by (code_len, sym_bytes)
    """
    items = sorted(lengths.items(), key=lambda kv: (kv[1], _sym_key(kv[0])))
    code = 0
    prev_len = 0
    out: Dict[Symbol, Tuple[int, int]] = {}
    for sym, L in items:
        code <<= (L - prev_len)
        out[sym] = (code, L)
        code += 1
        prev_len = L
    return out

def build_decode_trie(codes: Dict[Symbol, Tuple[int, int]]):
    """
    Build a binary trie for decoding bits -> symbol.
    """
    root = {}
    for sym, (code, L) in codes.items():
        cur = root
        for i in range(L - 1, -1, -1):
            bit = (code >> i) & 1
            cur = cur.setdefault(bit, {})
        cur["sym"] = sym
    return root

def decode_one_symbol(trie, bitreader) -> Symbol:
    cur = trie
    while "sym" not in cur:
        b = bitreader.read_bit()
        if b not in cur:
            raise ValueError("Invalid Huffman code (corrupt stream)")
        cur = cur[b]
    return cur["sym"]

def _sym_key(sym: Symbol):
    # stable ordering by serialized bytes: run (0..255), value (-32768..32767)
    run, val = sym
    # shift val to unsigned for ordering
    return (run, val + 32768)
