# SPDX-License-Identifier: GPL-3.0-or-later
"""New programme governance adapter for the unchanged observed-outcome learner.

Only governance predicates replace the previous study's approval checks. No
learner, store, timing, update, readout or state method is modified here.
"""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def hashes():
    files=[]
    for group in ('tests','protocol','source/runtime','provenance'):
        files.extend(p for p in (ROOT/group).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.md','.txt'))
    files.extend(p for p in (ROOT/'source/baseline/vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.npz','.txt'))
    files += [ROOT/'source/baseline/src/learner.py',ROOT/'source/baseline/src/bootstrap.py',ROOT/'source/BASELINE_PROJECTION_MANIFEST.json',ROOT/'source/baseline/vendor/COPYING']
    files += [ROOT/'README.md',ROOT/'operations/library-helper-current/library_file_transfer.py']
    files += [ROOT/'operations'/name for name in ('supervise.py','checkpoint.py','transport.py','controller.js','runner.js','host_session.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files) if p.name!='OFFICIAL_LOCK.json'}
def digest_map(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require_supervised():
    require(sys.flags.optimize==0,'optimized Python prohibited')
    require(os.environ.get('SURVIVOR_SUPERVISED')=='1','new programme supervisor required')
    active=read(ROOT/'operations/ACTIVE_JOB.json')
    require(active['child_pid']==os.getpid(),'supervisor child identity mismatch')
    require(active['source_digest']==digest_map(hashes()),'source changed since admission')
    if os.environ.get('SURVIVOR_MODE') in ('cycle1','cycle2','cycle3','science'):
        require(all(os.environ.get(k) for k in ('SURVIVOR_RESERVATION_ID','SURVIVOR_JOB_KEY','SURVIVOR_HOST_TOKEN')),'missing reservation/key/host identity')
        require(active.get('reservation_id')==os.environ.get('SURVIVOR_RESERVATION_ID') and active.get('key')==os.environ.get('SURVIVOR_JOB_KEY') and active.get('host_token')==os.environ.get('SURVIVOR_HOST_TOKEN'),'stale reservation/key/host admission')
def runtime_versions():
    import platform,numpy,scipy,numba
    return {'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'numba':numba.__version__}
def require_runtime():
    expected={'python':'3.11.15','numpy':'2.2.6','scipy':'1.14.1','numba':'0.61.2'}
    require(runtime_versions()==expected,'runtime differs from frozen source baseline')
    return expected
def require_technical():
    require_supervised()
    require_runtime()
    mode=os.environ.get('SURVIVOR_MODE')
    require(mode in ('cycle1','cycle2','cycle3'),'technical cycle not authorized')
    review=read(ROOT/f'audits/{mode.upper()}_SOURCE_REVIEW.json')
    require(review.get('accepted') is True,'independent cycle design/source audit missing')
    design=ROOT/'protocol'/review['design_file']
    require(sha(design)==review['design_sha256'],'stale design audit')
    require(review.get('source_hashes') and review['source_hashes']==hashes(),'incomplete or stale audited source closure')
    require(review.get('report_sha256')==sha(ROOT/f'audits/{mode.upper()}_SOURCE_REVIEW.md'),'audit report changed')
    return sha(design)
def require_science():
    require_supervised()
    require_runtime()
    lock=read(ROOT/'protocol/OFFICIAL_LOCK.json')
    require(lock['hashes']==hashes(),'official source drift')
    launch=read(ROOT/'audits/LAUNCH_ACCEPTED.json')
    require(launch.get('accepted') is True and launch['lock_sha256']==sha(ROOT/'protocol/OFFICIAL_LOCK.json'),'launch review missing')
    for rel,h in lock['acceptances'].items():require(sha(ROOT/rel)==h,'prior cycle acceptance or evidence drift')
    persisted=read(ROOT/'operations/PERSISTENCE_STATE.json')
    require(persisted.get('prelaunch_verified') is True and persisted.get('prelaunch_source_digest')==digest_map(hashes()),'operational persistence gate missing or stale')
    return sha(ROOT/'protocol/OFFICIAL_LOCK.json')
