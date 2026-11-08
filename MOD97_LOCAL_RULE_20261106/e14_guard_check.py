"""Verify source and new-birth drift is rejected before a subprocess starts."""
from pathlib import Path
import tempfile,json
import e14_campaign as C

def main():
 root=Path(tempfile.mkdtemp(prefix='e14-guard-'));C.R=root;C.O=root/'exp45';C.O.mkdir()
 file=root/'frozen.py';file.write_text('original')
 (root/'E14_LOCK.json').write_text(json.dumps(dict(files={'frozen.py':C.sha(file)})))
 file.write_text('changed')
 job=dict(key='no_spawn',seed=C.FRESH[0],kind='E8',part='A',command=['-c',"raise RuntimeError('must not start')"],env={})
 try:C.run(job)
 except RuntimeError as e:assert 'Locked input changed' in str(e)
 else:raise AssertionError('Source drift accepted')
 assert not list(C.O.glob('E14_JOB*'))
 file.write_text('original');birth=C.O/'birth.pt';birth.write_text('original')
 (root/'E14_BIRTHS.json').write_text(json.dumps(dict(files={'one':{'birth.pt':C.sha(birth)}})))
 birth.write_text('changed')
 try:C.run(job)
 except RuntimeError as e:assert 'Fresh birth changed' in str(e)
 else:raise AssertionError('Birth drift accepted')
 assert not list(C.O.glob('E14_JOB*'))
 result=dict(passed=True,source_drift_rejected_before_spawn=True,fresh_birth_drift_rejected_before_spawn=True,synthetic_fixture_only=True)
 (Path(__file__).resolve().parent/'E14_GUARD_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
