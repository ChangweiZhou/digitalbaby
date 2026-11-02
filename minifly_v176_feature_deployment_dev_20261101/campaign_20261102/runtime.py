"""V176 sequential campaign accounting. Shared atomic receipts; no implicit retries."""
import os,time,json,hashlib,resource,sys,signal
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent;DEV=R.parent;ROOT=DEV.parent
AUTH=json.loads((R/'AUTHORIZATION.json').read_text());STAGE=os.environ.get('V176_STAGE','qualification')
COUNTS=Counter(json.loads((R/'COUNTERS.json').read_text()) if (R/'COUNTERS.json').exists() else {})
PRIOR_CPU=float(os.environ.get('V176_PRIOR_CPU',AUTH['base_CPU_allowance']))
def stopped(signum,frame):raise RuntimeError('EXTERNAL_STOP_SIGNAL:'+str(signum))
signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
def usage():
 r=resource.getrusage(resource.RUSAGE_SELF);return r.ru_utime+r.ru_stime
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def budget():
 limits={
 'CPU_CAP':PRIOR_CPU+usage()<=AUTH['CPU_cap'],
 'WALL_CAP':time.time()-AUTH['epoch']<=AUTH['wall_cap'],
 'RSS_CAP':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=AUTH['RSS_cap'],
 'RECORD_CAP':COUNTS['source_records']+COUNTS['probe_record_touches']+COUNTS['full_records']+COUNTS['technical_feature_touches']+COUNTS['technical_sensory_passes']+COUNTS['reference_records']+AUTH['prior_technical_records_upper']<=AUTH['full_record_or_touch_cap'],
 'PROBE_TOUCH_CAP':COUNTS['probe_record_touches']<=2048,
 'TARGET_CAP':COUNTS['source_targets']+COUNTS['probe_fit_targets']+COUNTS['qualification_targets']+COUNTS['native_targets']+COUNTS['reference_targets']+AUTH['prior_training_target_applications_upper']<=AUTH['training_target_cap'],
 'PROBABILITY_CAP':COUNTS['probability_evaluations']+5000<=AUTH['probability_eval_cap'],
 'READ_CAP':COUNTS['native_vector_reads']+AUTH['prior_native_vector_reads_upper']<=AUTH['native_read_cap'],
 'PAIR_CAP':COUNTS['own_TF_pairs']+AUTH['prior_own_TF_pairs']<=AUTH['own_TF_cap']}
 for key,ok in limits.items():
  if not ok:raise RuntimeError(key)
def pulse(state,**extra):
 save('COUNTERS.json',dict(COUNTS));save('ACTIVE.json',dict(stage=STAGE,status=state,pid=os.getpid(),epoch=time.time(),process_CPU=usage(),total_CPU=PRIOR_CPU+usage(),wall=time.time()-AUTH['epoch'],peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**extra));budget()
def receipt(kind,key,records,targets,files,**extra):
 entry=dict(kind=kind,key=key,records=records,targets=targets,files={str(Path(p).relative_to(R)):sha(p) for p in files},**extra)
 path='receipts/'+key.replace('/','_')+'.json'
 if (R/path).exists():raise RuntimeError('IMMUTABLE_RECEIPT_EXISTS:'+key)
 save(path,entry)
 index=json.loads((R/'COMMIT_INDEX.json').read_text()) if (R/'COMMIT_INDEX.json').exists() else []
 index.append(dict(path=path,sha256=sha(R/path),kind=kind,key=key,records=records,targets=targets));save('COMMIT_INDEX.json',index)
 return entry
def done(**extra):
 pulse('STAGE_COMPLETE',**extra);save(f'ops/{STAGE}_DONE.json',dict(status='COMPLETE',stage=STAGE,CPU=usage(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**extra))
def fail(e,partial=None):
 save('COUNTERS.json',dict(COUNTS));save('FAILURE.json',dict(stage=STAGE,error=repr(e),CPU=usage(),epoch=time.time(),partial_state=partial,counts=dict(COUNTS),automatic_retry=False));print(repr(e),file=sys.stderr)
