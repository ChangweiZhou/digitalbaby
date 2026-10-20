"""Bounded G1 sanity after failed G0: not a full G2 efficacy campaign."""
from v4_common import *
from core_v4 import Learner
from data_v4 import dataset,score
from geometry import Geometry
import resource,pickle,gzip,shutil,gc

def point(m,taught,held):
 return dict(taught=score(m,taught),held=score(m,held),stats=m.stats(),clock=m.engine.time)

def run():
 start=time.process_time();wall=time.monotonic();cfg=spec();g0=json.loads((HERE/'cycles/cycle2/AUDIT.json').read_text());assert g0['accepted']
 snapshot=HERE/'cycles/cycle3/SOURCE_EXECUTED';snapshot.mkdir(parents=True,exist_ok=True)
 for f in HERE.glob('*.py'):shutil.copy2(f,snapshot/f.name)
 before=lock();save(Path('cycles/cycle3/SOURCE_LOCK.json'),before)
 # New passive-input and on-policy read-only wrapper must resume exactly.
 a=Learner(256,403);a.train(b'\x00\xff',b'\x80');h=a.digest();a.generate(b'\x00\xff');assert a.digest()==h
 b=pickle.loads(pickle.dumps(a));a.train(b'\x89',b'\xff');b.train(b'\x89',b'\xff');assert a.digest()==b.digest()
 n=Learner(256,403,learning=False);n.train(b'\x00',b'\xff');assert np.count_nonzero(n.engine.bank.fast)==np.count_nonzero(n.engine.bank.slow)==np.count_nonzero(n.engine.confirm)==0
 assert np.array_equal(n.boundary.w,np.zeros(9))
 g=Geometry(256,403,True);r=Geometry(256,403,False);assert np.array_equal(g.weights,r.weights)
 del a,b,n,g,r;gc.collect()
 jobs=[];records_total=4
 for world in cfg['worlds']:
  train,held=dataset(world,cfg['G1_records'],cfg['G1_probe_records']);rng=np.random.default_rng(world+9)
  # Shuffling within equal-length answer bins preserves per-record clocks.
  shuffled=list(train)
  for length in sorted({len(a) for _,a in train}):
   ids=[i for i,(_,a) in enumerate(train) if len(a)==length];answers=[train[i][1] for i in ids];rng.shuffle(answers)
   for i,a in zip(ids,answers):shuffled[i]=(train[i][0],a)
  for arm,segmented,gated,topographic,learning in (
   ('B0',False,False,False,True),('B1',False,True,False,True),('B2',True,False,True,True),
   ('B3',True,True,True,True),('B3_RANDOM',True,True,False,True),('B3_N',True,True,True,False),('B3_SH',True,True,True,True)):
   job_start=time.process_time();job_wall=time.monotonic();m=Learner(65536,world,segmented,gated,topographic,learning)
   folder=Path(f'cycles/cycle3/jobs/{world}_{arm}');path=HERE/folder;path.mkdir(parents=True,exist_ok=True)
   save(folder/'ACTIVE.json',dict(world=world,arm=arm,completed_external_records=0,pid=os.getpid()))
   points={'birth':point(m,train[:16],held)};traces=[];stream=shuffled if arm=='B3_SH' else train
   for i,(p,a) in enumerate(stream,1):
    traces.append(m.train(p,a))
    if i%16==0:
     save(folder/'ACTIVE.json',dict(world=world,arm=arm,completed_external_records=i,pid=os.getpid()))
     print('PROGRESS',world,arm,i,flush=True)
   points['acquisition']=point(m,train[:16],held)
   m.rest(replay_count=0);points['rest_before_replay']=point(m,train[:16],held)
   for p,a in list(m.replay[-8:]):traces.append(m.train(p,a,replay=True))
   points['rest_after_replay']=point(m,train[:16],held)
   # Report threshold-policy mismatch without selecting a better scorer.
   greedy=[];d=m.digest()
   for p,a in held:
    out=m.generate(p,greedy_boundary=True);greedy.append(bool(out['payload']==a.hex() and out['ended'] and not out['truncated']))
   assert m.digest()==d
   freq={}
   for _,a in stream:freq[a.hex()]=freq.get(a.hex(),0)+1
   default=sorted(freq,key=lambda a:(-freq[a],a))[0]
   shortcuts=dict(frequency=sum(a.hex()==default for _,a in held)/len(held),exact_key_with_frequency_default=sum(a.hex()==default for _,a in held)/len(held),training_orbits_disjoint=True)
   # Minimal complete local checkpoint is intentionally explicit about pickle format.
   if arm=='B3':
    with gzip.open(path/'FINAL_LOCAL_CHECKPOINT.pkl.gz','wb') as f:pickle.dump(m,f,protocol=5)
   receipt=dict(world=world,arm=arm,technical_only=True,scientific_records=0,
    full_external_records=64,replay_record_applications=8,record_applications=m.record_applications,
    target_events=m.feedback_bytes,points=points,traces=traces,final_stats=m.stats(),
    total_worker_phase_CPU_seconds=time.process_time()-job_start,wall_seconds=time.monotonic()-job_wall,
    process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    secondary_greedy_held_exact=sum(greedy)/len(greedy),shortcuts=shortcuts,
    source_lock_sha256=sha(HERE/'cycles/cycle3/SOURCE_LOCK.json'),final_state_digest=m.digest())
   assert len(traces)==m.record_applications==72
   assert sum(t['target_events'] for t in traces)==m.feedback_bytes
   save(folder/'RECEIPT.json',receipt);jobs.append(dict(world=world,arm=arm,path=str(folder/'RECEIPT.json'),receipt_sha256=sha(path/'RECEIPT.json'),record_applications=72,held_exact=points['rest_after_replay']['held']['exact'],taught_exact=points['rest_after_replay']['taught']['exact']))
   records_total+=72;save(Path('cycles/cycle3/COMMIT_INDEX.json'),dict(jobs=jobs,accepted_record_applications=records_total,preflight_record_applications=4,scientific_records=0))
   print('ACCEPTED',world,arm,'taught',jobs[-1]['taught_exact'],'held',jobs[-1]['held_exact'],flush=True);del m,receipt,points;gc.collect()
 assert lock()==before
 result=dict(accepted=True,scope='G1 bounded sanity only; G0 failed, so no full G2 campaign',full_native_record_applications=records_total,
   jobs=jobs,correlated_histories=len(jobs),preflight_record_applications=4,
   G0_any_pass=g0['G0_any_pass'],full_P3_started=False,CPU_seconds=time.process_time()-start,
   wall_seconds=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   passive_chronology_preflight=True,checkpoint_resume_exact=True,paired_fixed_KC_weights=True,
   source_unchanged=True,scientific_records=0)
 save(Path('cycles/cycle3/AUDIT.json'),result)
if __name__=='__main__':run()
