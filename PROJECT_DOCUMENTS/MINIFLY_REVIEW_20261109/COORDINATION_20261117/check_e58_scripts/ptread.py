"""Read a torch.save file (zip archive) without torch: standard library and numpy only."""
import io, pickle, zipfile
from collections import OrderedDict
import numpy as np

DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}

class _Storage:
    def __init__(self, name): self.name = name

def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    arr = storage
    n = 1
    for s in size: n *= s
    if len(size) == 0:
        return arr[offset:offset + 1].reshape(())
    # contiguous tensors only
    exp = []
    acc = 1
    for s in reversed(size):
        exp.append(acc); acc *= s
    assert tuple(reversed(exp)) == tuple(stride), (size, stride)
    return arr[offset:offset + n].reshape(size).copy()

def _rebuild_parameter(data, requires_grad, hooks):
    return data

class U(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f); self.zf, self.prefix = zf, prefix
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict": return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2": return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter": return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"): return _Storage(name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")
    def persistent_load(self, pid):
        kind, st, key, loc, numel = pid
        assert kind == "storage"
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=DT[st.name])

def load(path):
    zf = zipfile.ZipFile(path)
    pk = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
    prefix = pk[:-len("/data.pkl")]
    return U(io.BytesIO(zf.read(pk)), zf, prefix).load()
