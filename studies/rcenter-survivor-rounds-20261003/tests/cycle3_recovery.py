# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure durability/recovery/analysis certificates; no learner import or native event."""
import fcntl,hashlib,importlib.util,json,math,os,resource,signal,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/runtime'))
from integrity import require_technical,hashes,runtime_versions
from durable import create,replace,journal,validate_journal,sha
from analyze import intervals,decide,analyze,CONTRASTS,N,M,FAMILY_ALPHA
import numpy as np
from scipy.stats import t
require_technical();start=time.monotonic();checks=[]
mock=ROOT/'operations/mock-tests'/uuid.uuid4().hex;mock.mkdir(parents=True)
def reject(fn):
 try:fn()
 except (AssertionError,RuntimeError,FileExistsError,FileNotFoundError,ValueError):return
 raise AssertionError('failure guard accepted corrupted/incomplete input')
# Atomic immutable creation, replacement and chained journal tamper detection.
p=mock/'immutable.json';create(p,{'v':1});reject(lambda:create(p,{'v':2}));assert json.loads(p.read_text())=={'v':1}
replace(mock/'mutable.json',{'generation':1});replace(mock/'mutable.json',{'generation':2});assert json.loads((mock/'mutable.json').read_text())['generation']==2
for i in range(3):journal(mock/'journal',{'kind':'test','value':i})
head=validate_journal(mock/'journal');q=mock/'journal/000001.json';b=q.read_bytes();q.write_text('{}');reject(lambda:validate_journal(mock/'journal'));q.write_bytes(b);assert validate_journal(mock/'journal')==head
checks+=['immutable_no_overwrite','atomic_replace','journal_tamper_rejected']
# Exclusive writer lock is tested by a distinct mock process.
with (mock/'writer.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 code="import fcntl,sys; f=open(sys.argv[1],'a');\ntry:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:raise SystemExit(73)\nraise SystemExit(1)"
 r=subprocess.run([sys.executable,'-c',code,str(mock/'writer.lock')]);assert r.returncode==73
checks.append('duplicate_writer_rejected')
# Exercise actual production reconciliation, WAL, reservation and admission functions.
sys.path.insert(0,str(ROOT/'tests'))
from recovery_production_checks import run_checks as recovery_checks
checks+=recovery_checks(mock/'production_recovery')
from transport_collector_checks import run_checks as transport_checks
checks+=transport_checks()
from host_shutdown_checks import run_checks as host_checks
checks+=host_checks(mock)
from runner_shutdown_checks import run_checks as runner_checks
checks+=runner_checks()
from receipt_semantic_checks import run_checks as semantic_checks
semantic_result=semantic_checks(mock/'semantic_receipts',hashes())
checks+=semantic_result['semantic_tampers_rejected']
# Semantic accounting and final-results changes trigger streaming snapshots; heartbeat alone does not.
spec=importlib.util.spec_from_file_location('checkpoint_under_test',ROOT/'operations/checkpoint.py');cp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cp)
cp.ROOT=mock/'checkpoint_fixture';(cp.ROOT/'source').mkdir(parents=True);(cp.ROOT/'operations').mkdir();(cp.ROOT/'source/fake.py').write_text('# synthetic source only\n')
(cp.ROOT/'operations/RUN_LEDGER.json').write_text(json.dumps({'worker_s':1.}))
first=cp.build();(cp.ROOT/'operations/RUN_LEDGER.json').write_text(json.dumps({'worker_s':901.}));charged=cp.build();assert first['identity']!=charged['identity']
with __import__('zipfile').ZipFile(charged['path']) as z:assert json.loads(z.read('operations/RUN_LEDGER.json'))['worker_s']==901.
(cp.ROOT/'results').mkdir();(cp.ROOT/'results/analysis.json').write_text('{"synthetic":true}');result_snapshot=cp.build();assert result_snapshot['identity']!=charged['identity']
(cp.ROOT/'operations/HOST_WALL.json').write_text(json.dumps({'effective_s':2.}));unchanged=cp.build();assert unchanged['identity']==result_snapshot['identity'] and unchanged['changed'] is False
# Historical accepted inputs must survive private restore, without entering public scope.
(cp.ROOT/'audits').mkdir();(cp.ROOT/'operations/cycle1.log').write_text('synthetic historical dependency\n')
(cp.ROOT/'audits/CYCLE1_ACCEPTED.json').write_text(json.dumps({'input_hashes':{'operations/cycle1.log':sha(cp.ROOT/'operations/cycle1.log')}}))
with __import__('zipfile').ZipFile(cp.build()['path']) as z:assert z.read('operations/cycle1.log')==b'synthetic historical dependency\n'
checks+=['historical_accepted_log_checkpointed','production_checkpoint_charged_ledger_change','production_checkpoint_results_included','heartbeat_no_archive_churn']
# A separate short-lived process tests bounded streaming, never a retained archive loop.
big=cp.ROOT/'source/generated_mock.bin'
with big.open('wb') as f:
 block=b'0'*(1024*1024)
 for _ in range(32):f.write(block)
stream_code="import sys,importlib.util,resource,json;from pathlib import Path;s=importlib.util.spec_from_file_location('cp',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ROOT=Path(sys.argv[2]);d=m.build();print(json.dumps({'rss':next(int(line.split()[1])*1024 for line in open('/proc/self/status') if line.startswith('VmHWM:')),'inherited_ru_maxrss_upper_bound':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'archive':d['sha256']}))"
streamed=subprocess.run([sys.executable,'-c',stream_code,str(ROOT/'operations/checkpoint.py'),str(cp.ROOT)],capture_output=True,text=True);assert streamed.returncode==0,streamed.stderr;stream_result=json.loads(streamed.stdout);assert stream_result['rss']<128*1024**2
checks.append('32MiB_payload_streamed_below128MiB_RSS')
# Exercise the exact supervisor parent-death wrapper with a disposable writer.
spec=importlib.util.spec_from_file_location('qualified_supervisor',ROOT/'operations/supervise.py');sup=importlib.util.module_from_spec(spec);spec.loader.exec_module(sup)
heartbeat=mock/'heartbeat.py';heartbeat.write_text("import os,time\nfrom pathlib import Path\np=Path(os.environ['HEARTBEAT'])\nwhile True:\n with p.open('ab') as f:f.write(b'x');f.flush()\n time.sleep(.03)\n")
parent=mock/'mock_parent.py';info=mock/'parent_info.json';admit=mock/'active.json';heart=mock/'heart'
parent.write_text("import os,sys,json,subprocess,time\nfrom pathlib import Path\nenv=dict(os.environ,SUPERVISOR_PID=str(os.getpid()),SURVIVOR_SOURCE_DIGEST='mock-source',SURVIVOR_RESERVATION_ID='mock-reservation',SURVIVOR_JOB_KEY='mock-key',SURVIVOR_HOST_TOKEN='mock-host',ENTRY="+repr(str(heartbeat))+",ADMISSION="+repr(str(admit))+",HEARTBEAT="+repr(str(heart))+")\nenv.pop('SUPERVISOR_LOCK_FD',None)\np=subprocess.Popen([sys.executable,'-c',"+repr(sup.WRAPPER)+"],env=env)\nPath("+repr(str(admit))+").write_text(json.dumps({'child_pid':p.pid,'source_digest':'mock-source','reservation_id':'mock-reservation','key':'mock-key','host_token':'mock-host'}))\nPath("+repr(str(info))+").write_text(json.dumps({'parent':os.getpid(),'child':p.pid}))\nwhile True:time.sleep(1)\n")
proc=subprocess.Popen([sys.executable,str(parent)],start_new_session=True);child=None
try:
 deadline=time.monotonic()+5
 while not heart.exists() or heart.stat().st_size<2:
  assert time.monotonic()<deadline,'mock child failed to start';time.sleep(.03)
 child=json.loads(info.read_text())['child'];proc.kill();proc.wait(timeout=3);time.sleep(.2);n=heart.stat().st_size;time.sleep(.25);assert heart.stat().st_size==n,'worker survived parent death'
finally:
 if proc.poll() is None:proc.kill();proc.wait(timeout=3)
 try:os.killpg(proc.pid,signal.SIGKILL)
 except ProcessLookupError:pass
checks.append('actual_parent_death_worker_guard')
# Registered inference formula and routing on synthetic bounded world rows.
zero=[dict.fromkeys(CONTRASTS,0.) for _ in range(N)];ci=intervals(zero)
assert all(v['method']=='Hoeffding_zero_variance_fallback' and v['lower']<0<v['upper'] for v in ci.values())
expected_half=2*math.sqrt(math.log(2*M/FAMILY_ALPHA)/(2*N));assert abs(ci['E3_1']['upper']-expected_half)<1e-12
mixed=[dict(zero[i],E3_1=1. if i<32 else -1.) for i in range(N)];mc=intervals(mixed)['E3_1'];radius=float(t.ppf(1-FAMILY_ALPHA/(2*M),N-1))/math.sqrt(63)
assert abs(mc['mean'])<1e-12 and abs(mc['upper']-radius)<1e-12
positive=[{k:(.75 if k.startswith('W_') else (0. if k.endswith('minus_1') else 1.)) for k in CONTRASTS} for _ in range(N)]
assert decide(intervals(positive))['route'].startswith('A_')
shared=[dict(r,E3_1=0.,W_heldout_1_minus_chance=-.25) for r in positive];assert decide(intervals(shared))['route'].startswith('B_')
assert decide(ci)['route'].startswith('STOP_');assert decide(intervals(positive))['later_round_launch_authorized'] is False
reject(lambda:intervals(zero[:-1]));empty=mock/'empty_receipts';empty.mkdir();reject(lambda:analyze(empty,{},mock/'invalid_analysis.json'))
checks+=['eleven_contrast_bonferroni_formulas','constant_sample_non_degenerate_fallback','fixed_round_routes','incomplete_sample_rejected']
assert 'survivor_frozen_learner' not in sys.modules and 'engine' not in sys.modules,'pure test imported learner'
d={'schema':'RC-SURVIVOR-CYCLE3-OPERATIONS-v1','passed':True,'checks':checks,'mock_only':True,'native_teaching_events':0,'source':hashes(),'runtime':runtime_versions(),**{k:json.loads((ROOT/'operations/ACTIVE_JOB.json').read_text())[k] for k in ('key','attempt','reservation_id')},'semantic_receipt_tests':semantic_result,'streaming_test':stream_result,'resources':{'wall_s':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
create(ROOT/'receipts/cycle3/operations.json',d);print(json.dumps({'passed':True,'checks':checks,'resources':d['resources']}))
