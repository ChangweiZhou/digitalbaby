"""V148 corpus: public-domain texts from the NLTK mirror of Project Gutenberg, pinned by SHA-256.

A record is one non-empty line (title line in brackets dropped; lines > 128 bytes cut into 128-byte pieces).
The symbol stream of a record is its bytes followed by EOS (256), the model's own end-of-line decision.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EOS = 256
TEXTS = {   # name: (file, sha256)
    "alice": ("carroll-alice.txt", "5ce9042c0d8536b072b61d4c7c4e6d70c72d9dbb5f3d816261326d9280749266"),
    "buster": ("burgess-busterbrown.txt", "7d8b1648ffeaa7268819c62852e5653f2f295528411c8c895ce3941581146331"),
    "bryant": ("bryant-stories.txt", "c87f2c50a00055010f02edb6bb643aabe2473465da7f62db43cb4cda1a1c768d"),
}
MAX_LINE = 128


def load(name):
    fn, digest = TEXTS[name]
    raw = (HERE / "data" / fn).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SystemExit(f"corpus {name}: SHA-256 mismatch")
    return raw


def records(raw):
    out = []
    lines = raw.split(b"\n")
    if lines and lines[0].startswith(b"["):        # NLTK title line, e.g. "[Alice's Adventures ... 1865]"
        lines = lines[1:]
    for line in lines:
        line = line.rstrip(b"\r")
        if not line.strip():
            continue
        for i in range(0, len(line), MAX_LINE):
            out.append(line[i:i + MAX_LINE])
    return out


def symbols(rec):
    return len(rec) + 1          # bytes + EOS


def split(recs, train_kb, calib_kb, held_kb):
    """Contiguous blocks: train pool from the start, calibration right after it, held-out from the end.
    Sizes count record symbols (bytes + EOS). Returns lists of records."""
    def take(seq, kb):
        acc, out = 0, []
        for r in seq:
            if acc >= kb * 1024:
                break
            out.append(r)
            acc += symbols(r)
        return out
    train = take(recs, train_kb)
    calib = take(recs[len(train):], calib_kb)
    held = take(recs[::-1], held_kb)[::-1]
    first_held = len(recs) - len(held)
    if len(train) + len(calib) > first_held:
        raise SystemExit("blocks overlap: text too short for these sizes")
    return {"train": train, "calib": calib, "held": held,
            "offsets": {"train": [0, len(train)], "calib": [len(train), len(train) + len(calib)],
                        "held": [first_held, len(recs)], "n_records": len(recs)}}


def prefixes(train, kbs):
    """Record counts at which the cumulative taught symbols first reach each size (nested prefixes)."""
    out, acc, i = {}, 0, 0
    for kb in kbs:
        while i < len(train) and acc < kb * 1024:
            acc += symbols(train[i])
            i += 1
        out[kb] = i
    return out
