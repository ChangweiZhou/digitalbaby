#!/usr/bin/env python3
"""Compare two torch.save files tensor by tensor, without importing torch.

Usage:   python3 -I compare_pt_tensors.py A.pt B.pt

Why not compare file hashes: torch.save writes the archive's own name into the zip, so two files that hold identical tensors under
different file names have different SHA-256 hashes (E20-B's stage 1 record, note 2). This script compares the raw bytes of every
tensor of the same name instead.

Read-only, standard library only. The pickle inside the file is read with an unpickler that refuses every class except the few
that describe a tensor, so nothing from the file is executed.
"""
import collections
import io
import pickle
import sys
import zipfile

WIDTH = {"FloatStorage": 4, "DoubleStorage": 8, "HalfStorage": 2, "LongStorage": 8, "IntStorage": 4, "ShortStorage": 2,
         "CharStorage": 1, "ByteStorage": 1, "BoolStorage": 1}


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

    def persistent_load(self, pid):          # ('storage', storage type, key, location, numel)
        return (pid[1], pid[2])


def tensors(source):
    """source: a path or the file's bytes. Returns {name: (storage type, shape, raw bytes)}."""
    z = zipfile.ZipFile(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    root = pk[:-len("data.pkl")]
    out = {}

    def walk(o, path=""):
        if isinstance(o, tuple) and len(o) == 4 and o[0] == "T":
            _, (stype, key), offset, shape = o
            n = 1
            for s in shape:
                n *= s
            w = WIDTH[stype]
            out[path] = (stype, shape, z.read(root + "data/" + key)[offset * w:(offset + n) * w])
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(_Unpickler(z.open(pk)).load())
    return out


def compare(a, b):
    """Returns (names in both, names byte-equal, names only in a, names only in b)."""
    common = sorted(set(a) & set(b))
    return common, [k for k in common if a[k] == b[k]], sorted(set(a) - set(b)), sorted(set(b) - set(a))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A, B = tensors(sys.argv[1]), tensors(sys.argv[2])
    common, equal, only_a, only_b = compare(A, B)
    print(f"tensors: {len(A)} and {len(B)}; names in both: {len(common)}; byte-equal: {len(equal)}")
    if only_a or only_b:
        print(f"only in the first: {only_a}\nonly in the second: {only_b}")
    differing = [k for k in common if k not in equal]
    if differing:
        print("differing:", ", ".join(differing))
    sys.exit(0 if (len(equal) == len(common) and not only_a and not only_b) else 1)
