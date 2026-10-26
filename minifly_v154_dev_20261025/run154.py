"""One bounded V154 design/audit/test/revise exercise, no automatic retries."""
import os,sys,time,json,subprocess,pickle,traceback,resource
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
sys.dont_write_bytecode=True
import support154 as s
import native154 as n
import preflight154,checkpoints154,histories154,features154,probes154

def lock_sources():
 parent=s.load(s.ROOT/'minifly_v152_dev_20261025/SOURCE_LOCK.json');lock={}
 for path,digest in parent.items():
  if path.startswith(('minifly_a_v3/','minifly_cls_dev_20261017/')) or path in ['minifly_v152_dev_20261025/models152.py','minifly_v152_dev_20261025/support152.py']:
   assert s.sha(s.ROOT/path)==digest,path;lock[path]=digest
 for path in list(s.H.glob('*.py'))+[s.H/'AUTHORIZATION.json',s.D/'SPEC.json',s.D/'DESIGN.md',s.D/'ORIGIN.json']:
  lock[str(path.relative_to(s.ROOT))]=s.sha(path)
 s.save('SOURCE_LOCK.json',lock)

def main():
 assert not (s.H/'LEDGER.jsonl').exists(),'Do not replay an existing exercise'
 s.save('AUTHORIZATION.json',dict(version='V154',request='Do the 3-cycle exercise on V153 ... We call this V154 as Claude has been working on V153 already.',gradual_warmup=True,local_candidate_updates=True,no_arithmetic_rules=True,goal='own-byte/EOS unseen-addition generalization per sample and compute',bounded_test=True,formal_confirmation=False,github_push=False))
 lock_sources();s.log('CYCLE1_BEGIN');s.save('STATUS.json',dict(version='V154',status='CYCLE1_QUALIFICATION'))
 port_ready=preflight154.run();s.save('cycles/CYCLE1_REVISION.json',dict(status='PASS' if port_ready else 'PORT_NOT_QUALIFIED',revisions=['User-assigned identity V154; fresh V154 world/RNG namespace before feedback','Read-only prompt sharing qualified against the operative parent','Raw-tap collision audit distinguished from normalized-probe collisions'],scientific_roster_or_gates_changed=False));s.log('CYCLE1_COMPLETE',port_ready=port_ready)
 # Independent old-checkpoint lane cannot veto native learning.
 s.save('STATUS.json',dict(version='V154',status='CYCLE2_CHECKPOINTS'))
 cp=checkpoints154.run();data={w:s.dataset(w) for w in [202610255401,202610255402]};s.save('local/DATA_MANIFEST.json',data);s.save('DATA_LOCK.json',dict(sha256=s.sha(s.H/'local/DATA_MANIFEST.json'),worlds=list(data),before_new_feedback=True,RNG_namespace='V154'))
 s.log('CYCLE2_BEGIN');groups=[];configs=s.load(s.D/'SPEC.json')['native']['configurations']
 if port_ready:
  for world,dd in data.items():
   for config in configs:
    s.budget();s.save('STATUS.json',dict(version='V154',status='CYCLE2_NATIVE',group=f"{config['id']}_{world}"));groups.append(histories154.run_group(config,dd))
 held=histories154.score_held(groups,data)
 s.save('cycles/CYCLE2_REVISION.json',dict(status='COMPLETE',native_groups=len(groups),held_ready=sum(x['ready'] for x in groups),no_reference_veto=True,no_extra_training=True))
 s.log('CYCLE2_COMPLETE',groups=len(groups),ready=sum(x['ready'] for x in groups));s.log('CYCLE3_BEGIN');fits=[]
 for world,dd in data.items():
  for front in n.FRONTS:
   s.save('STATUS.json',dict(version='V154',status='CYCLE3_FEATURES_AND_PROBES',world=world,front=front));X,y,ids=features154.collect(dd,front)
   for tap,x in X.items():
    s.budget();key=f'{front}_{tap}_{world}';fits.append(probes154.fit(x,y,ids,dd,'OFF_257',key))
    if front in ['I_FAST','P_SLOW'] and tap in ['X','OBS']:
     for kind in ['LOC_257','OFF_C16','LOC_C16']:
      if kind.endswith('C16') and not port_ready:continue
      s.budget();fits.append(probes154.fit(x,y,ids,dd,kind,key))
 verdict='NO_NATIVE_ACQUISITION_READY' if not any(x['ready'] for x in groups) else 'DESCRIPTIVE_NATIVE_DEVELOPMENT_COMPLETE'
 summary=dict(version='V154',verdict=verdict,port_ready=port_ready,native_groups=len(groups),correlated_native_histories=3*len(groups),native_record_applications=sum(x['physical_records'] for x in groups),native_observed_targets=sum(x['targets'] for x in groups),native_readiness=[dict(config=x['config'],world=x['world'],ready=x['ready'],counts=x['readiness_counts'],immediate_exact=x['immediate_exact']) for x in groups],held=held,checkpoint_lane_status='COMPLETE' if cp else 'ASSET_UNAVAILABLE',diagnostic_fits=len(fits),qualified_fits=sum(x['status']=='QUALIFIED' for x in fits),optimizer_unresolved=sum(x['status']=='OPTIMIZER_UNRESOLVED' for x in fits),new_core_adopted=False)
 s.save('SUMMARY.json',summary);s.save('COUNTS.json',dict(counts=dict(s.COUNTS),timings=dict(s.TIMES),CPU=s.cpu(),wall_seconds=time.monotonic()-s.START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 result=subprocess.run([sys.executable,str(s.H/'audit154.py')],cwd=s.ROOT,capture_output=True,text=True);(s.H/'audit.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 s.log('CYCLE3_COMPLETE',verdict=verdict);s.save('STATUS.json',dict(version='V154',status='THREE_CYCLES_COMPLETE',verdict=verdict,audit='ACCEPTED',package_complete=False,new_core_adopted=False));print(json.dumps(summary),flush=True)

if __name__=='__main__':
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);s.save('LAUNCH.json',dict(pid=os.getpid(),caffeinate_pid=awake.pid,wall_start=s.WALL,workers=1,threads=1,python=sys.executable))
 try:main()
 except Exception as e:
  if s.ACTIVE_STATE is not None:s.checkpoint(s.ACTIVE_STATE,'INTERRUPTED_PROGRESS.pkl')
  s.save('FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=dict(s.COUNTS),CPU=s.cpu(),budget_pause=isinstance(e,s.BudgetStop),no_retry=True));s.save('STATUS.json',dict(version='V154',status=str(e) if isinstance(e,s.BudgetStop) else 'EXECUTION_FAILURE',error=repr(e)));print(traceback.format_exc(),flush=True);sys.exit(1)
 finally:
  s.save('COUNTS.json',dict(counts=dict(s.COUNTS),timings=dict(s.TIMES),CPU=s.cpu(),wall_seconds=time.monotonic()-s.START,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
  if awake.poll() is None:awake.terminate()
