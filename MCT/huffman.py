import heapq
from collections import Counter

class Node:
    def __init__(self, sym=None, freq=0, left=None, right=None):
        self.sym = sym
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_tree(freqs):
    pq = [Node(sym=s, freq=f) for s, f in freqs.items()]
    heapq.heapify(pq)
    while len(pq) > 1:
        a = heapq.heappop(pq)
        b = heapq.heappop(pq)
        heapq.heappush(pq, Node(freq=a.freq+b.freq, left=a, right=b))
    return pq[0]

def build_codebook(node, prefix="", code=None):
    if code is None:
        code = {}
    if node.sym is not None:
        code[node.sym] = prefix
    else:
        build_codebook(node.left, prefix + "0", code)
        build_codebook(node.right, prefix + "1", code)
    return code
