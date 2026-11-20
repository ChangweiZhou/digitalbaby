#!/usr/bin/env python3
"""The compiled judging of candidate lines (kernm.c) for replay6.py. The numpy function replay6.mismatch is the reference (check_replay6.py)."""
import ctypes, os, subprocess
import numpy as np
from fastr import FLAGS, I64, PD, P32
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, "kernm.so"); SRC = os.path.join(HERE, "kernm.c")
_lib = None
def lib():
    global _lib
    if _lib is None:
        if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
            tmp = f"{SO}.{os.getpid()}.tmp"; subprocess.check_call(["gcc", *FLAGS, "-o", tmp, SRC, "-lm"]); os.replace(tmp, SO)
        _lib = ctypes.CDLL(SO); _lib.judge_lines.restype = None; _lib.judge_lines.argtypes = [I64] * 5 + [PD] * 4 + [I64, P32, PD]
    return _lib
def judge(s, L, END):
    L = np.ascontiguousarray(L, np.int32); out = np.zeros(len(L)); NL = s.W1.shape[1]; S = s.W2.shape[0]; p = lambda a: a.ctypes.data_as(PD)
    lib().judge_lines(s.H, NL, S, NL // S, END, p(s.W1), p(s.b1), p(s.W2), p(s.b2), len(L), L.ctypes.data_as(P32), p(out)); return out
if __name__ == "__main__": lib(); print("built", SO)
