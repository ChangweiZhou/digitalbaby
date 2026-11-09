"""Compare the byte-embedding rows (book.weight) of a birth checkpoint and an end-of-life checkpoint.

Read-only and torch-free: a torch.save(state_dict) file is a zip holding a pickle plus raw float32 storages, and this
reads it with a restricted unpickler. Run with:  python3 -I check_e17_embedding_rows.py BIRTH.pt FINAL.pt [LR WD UPDATES]

For each group of symbols it prints the mean row norm at birth and at the end, and the smallest and largest
elementwise ratio final/birth. A group whose elements were all scaled by one common factor close to (1 - LR*WD)**UPDATES
received zero gradient for the whole life and was changed by weight decay alone (the factor differs from the float64
value in the fourth digit because the optimizer works in float32). Defaults: LR 0.001, WD 0.25, UPDATES 36864.
"""
import collections
import math
import pickle
import struct
import sys
import zipfile


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *a, **k: ("T", storage, offset, tuple(size))
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a, **k: data
        if mod == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"refused {mod}.{name}")

    def persistent_load(self, pid):
        return (pid[1], pid[2])       # (storage type, storage key)


def book_rows(path):
    z = zipfile.ZipFile(path)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    root = pk[:-len("data.pkl")]
    sd = _Unpickler(z.open(pk)).load()
    _, (stype, key), off, shape = sd["book.weight"]
    assert stype == "FloatStorage", stype
    rows, cols = shape
    flat = struct.unpack_from(f"<{rows * cols}f", z.read(root + "data/" + key), off * 4)
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


GROUPS = {
    "digits 0-9": [ord(c) for c in "0123456789"],
    "'=' and ';'": [ord("="), ord(";")],
    "operators + - > <": [ord(c) for c in "+-><"],
    "distractor letters A-Z": list(range(65, 91)),
    "birth task codes p q v w": [ord(c) for c in "pqvw"],
    "life task codes r t x n s u y z c": [ord(c) for c in "rtxnsuyzc"],
}


def main():
    birth, final = book_rows(sys.argv[1]), book_rows(sys.argv[2])
    lr, wd, updates = (float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0.001, 0.25, 36864)
    expect = (1 - lr * wd) ** updates
    print(f"pure-decay factor (1 - {lr}*{wd})**{updates} = {expect:.6g}")
    norm = lambda v: math.sqrt(sum(x * x for x in v))
    for name, idx in GROUPS.items():
        ratios = [f / b for i in idx for f, b in zip(final[i], birth[i]) if abs(b) > 1e-12]
        nb = sum(norm(birth[i]) for i in idx) / len(idx)
        nf = sum(norm(final[i]) for i in idx) / len(idx)
        decay_only = min(ratios) > 0 and max(ratios) / min(ratios) - 1 < 1e-4 and abs(min(ratios) / expect - 1) < 1e-2
        print(f"{name:34s} row norm {nb:8.5f} -> {nf:10.7f}   final/birth min {min(ratios):.6g} max {max(ratios):.6g}"
              f"   {'DECAY ONLY: zero gradient all life' if decay_only else 'learned'}")


if __name__ == "__main__":
    main()
