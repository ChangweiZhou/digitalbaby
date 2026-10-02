import json
from pathlib import Path
import pytest
import recovery_local as r

@pytest.fixture
def env(tmp_path):
 root=tmp_path/'A';(root/'ops').mkdir(parents=True);(root/'ops/HOST_LOSS_RECOVERY_AMENDMENT.md').write_text('test')
 (root/'SOURCE_LOCK.json').write_text(json.dumps({'lock_digest':'L'}))
 e=r.RecoveryEnv(root);e.sci.mkdir(parents=True)
 ledger={'lock_digest':'L','active_wall_s':100,'worker_s':400,'failure':None,'cleared':[],'sessions':[]}
 r.write_json(e.sci/'RUN_LEDGER.json',ledger)
 r.write_json(e.ack,{'schema':'MINIFLY-A3-VERIFIED-BACKUP-v1','commit':'abc','verified_at_utc':'2026-10-01T00:00:00Z','lock_digest':'L','receipt_sha256':{}})
 e.bootstrapping=True
 return e

def add(e,n):
 for i in range(n):
  p=e.sci/'R0'/f'{190001+i}.json.gz';p.parent.mkdir(exist_ok=True);raw=f'synthetic{i}'.encode()
  if not p.exists():p.write_bytes(raw)
  e.validated[('R0',190001+i)]=r.sha(raw)
 r.write_json(e.sci/'RUN_STATUS.json',{'validated_receipts':n})

def test_write_once(tmp_path):
 p=tmp_path/'once.json';r.write_json(p,{'a':1},True)
 with pytest.raises(FileExistsError):r.write_json(p,{'a':2},True)
 assert json.loads(p.read_text())=={'a':1}

def test_absent_journal_refused(env):
 env.bootstrapping=False
 with pytest.raises(AssertionError):env.verify_remote()

def test_bootstrap_chain(env):
 add(env,1);env.persist('one');env.bootstrapping=False;env.persist('two');tip,doc=env.tip()
 assert tip['sequence']==2 and doc['validated_receipts']==1
 assert json.loads((env.root/'results/recovery/LOCAL_STATUS.json').read_text())['verified_remote_receipts']==0

def test_changed_receipt_refused(env):
 add(env,1);env.persist('one');(env.sci/'R0/190001.json.gz').write_bytes(b'different')
 with pytest.raises(AssertionError):env.persist('bad')

def test_prior_omission_refused(env):
 add(env,1);env.persist('one');env.validated.clear();r.write_json(env.sci/'RUN_STATUS.json',{'validated_receipts':0})
 with pytest.raises(AssertionError):env.persist('bad')

def test_durability_backlog_stops_at_eight(env):
 add(env,8)
 with pytest.raises(RuntimeError,match='Durability backlog'):env.persist('eight')
 assert env.tip()[1]['validated_receipts']==8

def test_ack_matching_bytes_allows_backlog_clear(env):
 add(env,8);ack=json.loads(env.ack.read_text());ack['receipt_sha256']={f'results/science/{a}/{w}.json.gz':h for (a,w),h in env.validated.items()};r.write_json(env.ack,ack);env.persist('backed')
 assert json.loads((env.root/'results/recovery/LOCAL_STATUS.json').read_text())['unbacked_receipts']==0

def test_false_ack_changed_hash_refused(env):
 add(env,1);ack=json.loads(env.ack.read_text());ack['receipt_sha256']={'results/science/R0/190001.json.gz':'wrong'};r.write_json(env.ack,ack)
 with pytest.raises(AssertionError):env.persist('bad')

def test_counters_never_decrease(env):
 add(env,1);env.persist('one');p=env.sci/'RUN_LEDGER.json';d=json.loads(p.read_text());d['worker_s']=399;r.write_json(p,d)
 with pytest.raises(AssertionError):env.persist('bad')

def test_process_local_frozen_audit_cache_keys(env,monkeypatch):
 calls=[];monkeypatch.setattr(r.frozen.Env,'validate',lambda *a:calls.append(a))
 a=('R0',190001,b'raw',{'lock_digest':'L'},{})
 env.validate(*a);env.validate(*a);assert len(calls)==1
 env.validate('R0',190001,b'raw',{'lock_digest':'L'},{'dep':b'changed'});assert len(calls)==2

def test_fs_failure_propagates(env,monkeypatch):
 add(env,1)
 def fail(fd):raise OSError('storage failure')
 monkeypatch.setattr(r.os,'fsync',fail)
 with pytest.raises(OSError):env.persist('bad')

def test_status_mismatch_refused(env):
 add(env,1);r.write_json(env.sci/'RUN_STATUS.json',{'validated_receipts':2})
 with pytest.raises(AssertionError):env.persist('bad')

def test_verified_private_backup_satisfies_offhost_limit(env):
 import recovery_with_private as private
 env.__class__=private.PrivateRecoveryEnv
 add(env,8);roster={f'results/science/{a}/{w}.json.gz':h for (a,w),h in env.validated.items()}
 r.write_json(env.root/'results/recovery/PRIVATE_BACKUP_STATUS.json',{'schema':'MINIFLY-A3-VERIFIED-PRIVATE-BACKUP-v1','library_file_id':'test-only','verified_at_utc':'now','lock_digest':'L','receipt_sha256':roster})
 env.persist('private backup verified')
 status=json.loads((env.root/'results/recovery/OFF_HOST_STATUS.json').read_text());assert status['verified_github_receipts']==0 and status['verified_private_recoverable_receipts']==8 and status['unprotected_accepted_receipts']==0

def test_missing_private_backup_does_not_bypass_limit(env):
 import recovery_with_private as private
 env.__class__=private.PrivateRecoveryEnv;add(env,8)
 with pytest.raises(RuntimeError,match='without either'):env.persist('no backup')

def test_false_private_backup_is_integrity_failure(env):
 import recovery_with_private as private
 env.__class__=private.PrivateRecoveryEnv;add(env,8)
 r.write_json(env.root/'results/recovery/PRIVATE_BACKUP_STATUS.json',{'schema':'MINIFLY-A3-VERIFIED-PRIVATE-BACKUP-v1','library_file_id':'test-only','verified_at_utc':'now','lock_digest':'L','receipt_sha256':{'results/science/R0/190001.json.gz':'wrong'}})
 with pytest.raises(AssertionError):env.persist('false backup')
