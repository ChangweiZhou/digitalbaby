import copy,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest
import numpy as np
import audit_receipts as audit
import analyze
import locked_run
from assay import ROOT,ARMS

@pytest.fixture(scope='module')
def doc():return audit.load(ROOT/'results/cycle2/FE0/290001.json.gz')

@pytest.mark.parametrize('field',['world','params','source','runtime','row','target','accuracy','decomposition'])
def test_corrupt_receipt_rejected(doc,field):
    d=copy.deepcopy(doc)
    if field=='world':d['world']+=1
    elif field=='params':d['params']['j_gain']*=2
    elif field=='source':d['source_hashes']['src/assay.py']='0'*64
    elif field=='runtime':d['runtime']['pandas']='wrong'
    elif field=='row':d['records'].pop()
    elif field=='target':d['records'][0]['answer']=99
    elif field=='accuracy':d['probes']['final']['W']['old']['accuracy']+=.1
    elif field=='decomposition':d['probes']['final']['W']['old']['decomposition']['g12']+=1
    with pytest.raises((AssertionError,ValueError,KeyError)):
        audit.validate(d,expected_world=doc['world'],expected_arm=doc['arm'],expected_bouts=doc['bouts'],
            expected_sources=doc['source_hashes'],expected_params=doc['params'],expected_runtime=doc['runtime'])


def mock_final(tmp_path,monkeypatch):
    orig=audit.load(ROOT/'results/cycle2/FE0/290001.json.gz')
    lock=dict(worlds=[290001],arms=list(ARMS),bouts=4,params=orig['params'],source_hashes=orig['source_hashes'],
              expected_receipt_runtime=orig['runtime'],replay_jobs=[])
    for a in ARMS:
        d=audit.load(ROOT/'results/cycle2'/a/'290001.json.gz');d['lock_sha256']='test-lock'
        p=tmp_path/'results/final'/a/'290001.json.gz';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps(d).encode()))
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    monkeypatch.setattr(locked_run,'verify_lock',lambda:(lock,'test-lock'))
    return lock

@pytest.mark.parametrize('worlds',([], [290001,290001],[290002]))
def test_final_rejects_partial_duplicate_or_changed_roster(tmp_path,monkeypatch,worlds):
    mock_final(tmp_path,monkeypatch)
    with pytest.raises(AssertionError):audit.audit_suite('final',worlds)


def test_final_exact_roster_passes_then_extra_rejected(tmp_path,monkeypatch):
    mock_final(tmp_path,monkeypatch)
    assert audit.audit_suite('final',[290001])['pass_all']
    p=tmp_path/'results/final/FE0/290001.json.gz'
    (p.parent/'290099.json.gz').write_bytes(p.read_bytes())
    with pytest.raises(AssertionError):audit.audit_suite('final',[290001])


def test_analysis_roundtrip_and_holm():
    out=analyze.analyze('cycle2',[290001],bouts=4)
    loaded=json.loads(json.dumps(out,allow_nan=False))
    text=analyze.render(loaded)
    assert 'EXPLORATORY PILOT ONLY' in text and 'not' in text.lower()
    assert len(loaded['primary'])==4 and len(loaded['arms'])==7
    assert np.allclose(analyze.holm([.01,.03,.2,.8]),[.04,.09,.4,.8])


def test_final_job_authorization_rejects_before_fixture(monkeypatch):
    lock=dict(arms=list(ARMS),worlds=[300001],bouts=6,params=locked_run.EXPECTED_PARAMS)
    monkeypatch.setattr(locked_run,'verify_lock',lambda:(lock,'fake'))
    with pytest.raises(AssertionError):locked_run.authorize_final('T',300002,6,locked_run.EXPECTED_PARAMS)
    with pytest.raises(AssertionError):locked_run.authorize_final('T',300001,5,locked_run.EXPECTED_PARAMS)
    with pytest.raises(AssertionError):locked_run.authorize_final('T',300001,6,{**locked_run.EXPECTED_PARAMS,'j_gain':1})

def test_cap_termination_escalates_to_kill():
    import subprocess
    class Fake:
        stopped=False;killed=False
        def terminate(self):self.stopped=True
        def kill(self):self.killed=True
        def wait(self,timeout):
            if not self.killed:raise subprocess.TimeoutExpired('fake',timeout)
    p=Fake();locked_run.stop_process(p)
    assert p.stopped and p.killed

def test_sealed_final_path_fresh_process_smoke_without_final_worlds():
    """Run pilot, sealed-final path and replay on technical world290099 only."""
    import os,subprocess,tempfile
    scratch=ROOT/'scratch';scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sealed-final-smoke-',dir=scratch) as td:
        td=Path(td);paths=[]
        for i,kind in enumerate(('technical_smoke','final','final')):
            destination=td/f'{i}.json.gz';paths.append(destination)
            code=f"""
import sys
sys.path.insert(0,{str(ROOT/'src')!r})
import locked_run
from assay import run
params=locked_run.EXPECTED_PARAMS.copy()
if {kind!r}=='final':
    # Disposable synthetic lock; only a previously used technical world is allowed.
    locked_run.verify_lock=lambda: (dict(arms=['FE0'],worlds=[290099],bouts=1,params=params),'synthetic-technical-lock')
run('FE0',290099,1,params,kind={kind!r},destination={str(destination)!r})
"""
            subprocess.run([sys.executable,'-c',code],check=True,env=os.environ.copy(),timeout=180)
        pilot,final,replay=map(audit.load,paths)
        assert pilot['runtime']==final['runtime']==replay['runtime']
        audit.validate(final,expected_world=290099,expected_arm='FE0',expected_bouts=1,
            expected_sources=pilot['source_hashes'],expected_params=locked_run.EXPECTED_PARAMS,
            expected_runtime=pilot['runtime'],lock_sha256='synthetic-technical-lock')
        assert final['resource']['process_id']!=replay['resource']['process_id']
        final.pop('resource');replay.pop('resource')
        assert final==replay
