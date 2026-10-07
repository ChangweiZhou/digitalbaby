"""Lock only actual new code and exact imported source dependencies."""
import hashlib,sys,json
from pathlib import Path
import bootstrap
from io_utils import digest,atomic_json
def source_map():
    paths=list(bootstrap.ROOT.glob('*.py'))+[bootstrap.ROOT/'SPEC_LOCK.md',bootstrap.ROOT/'requirements.txt']
    paths += [p for p in bootstrap.PARENT.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz') and 'scratch' not in p.parts and 'results' not in p.parts and '__pycache__' not in p.parts]
    paths += [bootstrap.V2/'v2_fixture.py',bootstrap.V2/'v2_core.py',bootstrap.COMPACT/'compact_fixture.py',bootstrap.LATIN]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
def identity():return digest(source_map())
def lock():
    import numpy,scipy,numba,pandas
    versions=(sys.version.split()[0],numpy.__version__,scipy.__version__,numba.__version__,pandas.__version__)
    if versions!=('3.11.5','2.2.6','1.14.1','0.61.2','2.2.3'):raise ValueError('pinned runtime mismatch')
    d={'identity':identity(),'files':source_map(),'runtime':versions,'science_authorized':False}
    atomic_json(bootstrap.ROOT/'SOURCE_LOCK.json',d);return d['identity']
def require(identity_expected):
    if identity()!=identity_expected:raise ValueError('stale source lock')
