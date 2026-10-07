import hashlib
import runtime
from integrity import source_map as parent_source_map
from io_utils import digest

def source_map():
    d = parent_source_map()
    for p in sorted((runtime.ROOT / 'code').glob('*.py')):
        d[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    p = runtime.ROOT / 'PARAMETERS.json'
    d[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d

def identity(): return digest(source_map())

def require(want):
    if identity() != want: raise ValueError('source identity changed')

