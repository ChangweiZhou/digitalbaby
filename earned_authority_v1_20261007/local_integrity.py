"""Immutable science sources; reports and technical results are not code."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCIENCE_FILES=('authority_policy.py','bridge_runtime.py','run_job.py','independent_audit.py','local_integrity.py')
def source_map():
    import bridge_runtime as rt
    import integrity as old
    files=old.source_map()
    files.update({str(ROOT/n):hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SCIENCE_FILES})
    if (ROOT/'PROTOCOL_LOCK.json').exists():files[str(ROOT/'PROTOCOL_LOCK.json')]=hashlib.sha256((ROOT/'PROTOCOL_LOCK.json').read_bytes()).hexdigest()
    return files

def identity():
    from bridge_runtime import digest
    return digest(source_map())

def locked_identity():
    lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text())
    expected=lock['identity']
    require(expected)
    if lock['science_authorized'] is not False:raise ValueError('unexpected authorization in development package')
    return expected

def require(expected):
    if identity()!=expected:raise ValueError('scientific source changed during technical run')

def lock():
    import numpy,scipy,numba,pandas
    versions=[sys.version.split()[0],numpy.__version__,scipy.__version__,numba.__version__,pandas.__version__]
    assert versions==['3.11.5','2.2.6','1.14.1','0.61.2','2.2.3']
    from bridge_runtime import atomic_json
    d={'identity':identity(),'files':source_map(),'runtime':versions,'science_authorized':False,
       'local_runner':'development-worlds-only','upstream_sources':'unchanged imported ERROR core and fixtures'}
    atomic_json(ROOT/'SOURCE_LOCK.json',d);return d
if __name__=='__main__':print(json.dumps(lock(),indent=2))
