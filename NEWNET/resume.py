"""Resumable long tests (24 November 2026). The cloud containers restart every 30-45 minutes and kill every process; the life tests
of wide nets (cells per column M=128) run 1-4 hours, so they never finished. A test saves its whole state (the net, the random
generators, its counters and where it is) to the file in $CKPT every $CKPT_EVERY seconds (default 300) and at its stage ends; when it is
started again with the same command it loads the file and goes on from there. The state is saved only between steps, and nothing is
recomputed, so a resumed run gives the same numbers as one that was never stopped (checked with SMOKE runs: see SEARCH2_STATE.md).
No $CKPT (the default, and every SMOKE run): nothing is saved. The file is removed when the test finishes. A file written by another
command (other options) is ignored."""
import os, sys, time, pickle, gzip
P = os.environ.get("CKPT")
EVERY = float(os.environ.get("CKPT_EVERY", 300))
KEY = " ".join(sys.argv)
_last = [time.time()]
def load():
    """the saved state dict, or None"""
    if not P or not os.path.exists(P): return None
    try:
        with gzip.open(P, "rb") as f: s = pickle.load(f)
    except Exception as e:
        print(f"resume: ignoring unreadable {P}: {e}", flush=True); return None
    if s.get("_key") != KEY:
        print(f"resume: ignoring {P} (written by another command)", flush=True); return None
    print(f"resume: going on from {P} ({s.get('_where')})", flush=True); return s
def save(state, where, force=False):
    """state: a dict of everything the test needs to go on; saved if forced or EVERY seconds have passed"""
    if not P or (not force and time.time() - _last[0] < EVERY): return
    state = dict(state, _key=KEY, _where=where); tmp = P + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(P)), exist_ok=True)
    with gzip.open(tmp, "wb", compresslevel=1) as f: pickle.dump(state, f, protocol=4)
    os.replace(tmp, P); _last[0] = time.time()
def done():
    if P and os.path.exists(P): os.remove(P)
