#!/usr/bin/env python3
"""The relay system's opinions for the whole text, by the author's relay.opinions(0) (scratch copy of nano/table and nano/loop), saved for the full-size run;
then the same with the WHOLE last tenth replaced by other bytes: the opinions for the stream and for the first byte of the last tenth must be bit-identical
(the relay system's strengths at the end of the stream cannot have seen the last tenth)."""
import os, sys, time, resource
W = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(W, "table")); sys.dont_write_bytecode = True
import numpy as np
import score, relay
import loop as LP
assert os.path.dirname(os.path.abspath(LP.__file__)) == os.path.join(W, "loop")
x, cut, S = score.load(); xl_, cutl, Sl, pad = LP.load(); assert (xl_ == x).all() and cutl == cut and Sl == S and pad == 0
print("loop.load gives the same text, cut and values as score.load:", True, "| pad (line feed) index", pad, flush=True)
t0 = time.process_time(); X, xlev, alone = relay.opinions(0); print("relay alone:", alone, "| cpu", round(time.process_time() - t0, 1), flush=True)
np.save(os.path.join(W, "relay_X.npy"), X); np.save(os.path.join(W, "relay_lev.npy"), xlev)
print("X", X.shape, X.dtype, "levels", np.bincount(xlev, minlength=10).tolist(), "| rows are logs of a distribution: max |sum exp - 1|", float(np.abs(np.exp(X.astype(np.float64)).sum(1) - 1).max()), flush=True)
rep = {"nats": 1.916333556175232, "right": 44.281871974179666, "stream_nats_as_read": 1.9669312238693237}      # res/relay_arm1.json (the author's run, seed 1)
print("the author's relay_arm1.json relay_alone:", rep, "| equal:", {k: alone[k] == rep[k] for k in rep}, flush=True)
if "second" in sys.argv:
    xm = x.copy(); xm[cut:] = np.random.default_rng(77).integers(0, S, len(x) - cut); orig = LP.load; LP.load = lambda: (xm.astype(np.int32), cut, S, pad)
    t1 = time.process_time(); X2, lev2, alone2 = relay.opinions(0); LP.load = orig
    print("last tenth replaced by random bytes (", int((xm != x).sum()), "bytes differ ): opinions for positions 0..cut bit-identical:", bool((X2[:cut + 1] == X[:cut + 1]).all()), "| levels:", bool((lev2[:cut + 1] == xlev[:cut + 1]).all()),
          "| rows of the last tenth after its first that differ:", int((X2[cut + 1:] != X[cut + 1:]).any(1).sum()), "of", len(x) - cut - 1, "| relay alone on the random tail:", alone2, "| cpu", round(time.process_time() - t1, 1), flush=True)
print("total cpu", round(time.process_time() - t0, 1), "peak RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
