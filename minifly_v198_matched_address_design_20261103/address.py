"""Deterministic row substitution with the inherited 16-byte window."""
from hashlib import sha256
import numpy as np
from source import parent

ARMS = ("SHORT", "MULTISCALE", "REHASH")
ROWS = 65536

def short_grams(h):
    return [257*a+b for a,b in zip(h,h[1:])] + [
        65599*a+257*b+7919*c+1 for a,b,c in zip(h,h[1:],h[2:])
    ]

def rank(domain, value):
    return sha256(f"MINIFLY|V198|{domain}|{value}".encode()).digest()

def ordered(rows, domain):
    return sorted(rows, key=lambda row: (rank(domain, row), row))

def codes(prefix):
    h = prefix[-16:]
    short, feature = parent.bag(h, ROWS)
    original = set(short.tolist())
    four = {((16777259*a+65599*b+257*c+31*d+3)*2654435761+j*40503)%ROWS
            for a,b,c,d in zip(h,h[1:],h[2:],h[3:]) for j in range(4)}
    rehash = {(int.from_bytes(rank("short-rehash", g)[:2], "big")+j*40503)%ROWS
              for g in short_grams(h) for j in range(4)}
    four -= original
    rehash -= original
    q = min(len(short)//4, len(four), len(rehash))
    kept = original - set(ordered(original,"drop")[:q])
    result = {"SHORT": (short, feature)}
    for arm, pool in (("MULTISCALE",four),("REHASH",rehash)):
        rows = np.array(sorted(kept | set(ordered(pool,arm)[:q])), dtype=int)
        result[arm] = rows, feature
    return result, q
