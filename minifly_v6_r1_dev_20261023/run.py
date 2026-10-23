"""Single-process frozen engineering roster; evaluator never imported by learner."""
from core import *
import ops,v6_data as data,time,traceback,collections
RDIR=P/'results';rec=0;obs=0;private_events=0;queries=0

def save_probe(f,d,arm,stage,pools):
 global queries,private_events
 before=f.digest();rows=[];start=time.process_time()
 for pool,rs in pools.items():
  for i,r in enumerate(rs):
   for orientation,q in enumerate((r,data.flip(r))):
    for tape in range(3):
     ops.check(stage,world=d['seed'],arm=arm,records=rec,observed_events=obs,queries=queries) if queries%128==0 else ops.check(world=d['seed'],arm=arm,records=rec,observed_events=obs,queries=queries)
     seed=int(d['seed']+100000+10000*{'old':0,'new':1,'held':2}[pool]+i*10+orientation*3+tape)
     out,ended,trunc,marg=f.query(bytes(q['prompt']),seed);queries+=1;private_events+=len(q['prompt'])+len(marg)
     rows.append(dict(pool=pool,index=i,orientation=orientation,tape=tape,prompt=q['prompt'],target=q['answer'],output=out,EOS=ended,truncated=trunc,margins=marg,exact=bool(ended and out==q['answer']),payload_exact=out==q['answer']))
 assert before==f.digest()
 ops.save(f'results/probes/{d["seed"]}_{arm}_{stage}.json',dict(seed=d['seed'],arm=arm,stage=stage,records=f.c.n,CPU_seconds=ops.stats()['CPU_seconds'],read_CPU_seconds=time.process_time()-start,rows=rows,readonly_digest=before))
 return {pool:float(np.mean([r['exact'] for r in rows if r['pool']==pool])) for pool in pools}

def history(d,arm):
 global rec,obs,private_events
 ops.check('HISTORY_START',admission=True,world=d['seed'],arm=arm);t0=time.process_time();f=Fly(d['seed']);start=(f.c.Wv.copy(),f.c.Wg.copy());summaries=[];train_gen=collections.Counter();stsum=collections.Counter();maxgrad=0;maxtrace=0
 tag=f'{d["seed"]}_{arm}';private0=private_events;obs0=obs
 # Inspect stable-random-tape addresses on OLD-only prompts; never labels.
 def address_fingerprint():
  v=Fly(d['seed']);v.c.Wv=f.c.Wv.copy();v.c.Wg=f.c.Wg.copy();codes=[]
  for i,r in enumerate(d['old'][:8]):
   v.c.rng=np.random.default_rng(d['seed']+i+500);v.begin()
   for t in r['prompt']:v.c.step(t,False)
   v.c.step(257,False);codes.append(v.code().astype(np.uint8))
  return np.array(codes)
 initial_codes=address_fingerprint();initialW=[x.copy() for x in start]
 save_probe(f,d,arm,'BASELINE',{'old':d['old'][:8],'held':d['held'][:8]})
 def checkpoint(stage):
  f.checkpoint(P/'checkpoints'/f'{tag}_{stage}.npz')
  current=address_fingerprint();inter=(current*initial_codes).sum(1);union=np.maximum(current,initial_codes).sum(1)
  summaries.append(dict(stage=stage,records=f.c.n,CPU_seconds=ops.stats()['CPU_seconds'],native=f.e.stats(),controller_weight_delta=float(np.sqrt(np.sum((f.c.Wv-initialW[0])**2)+np.sum((f.c.Wg-initialW[1])**2))),controller_path_movement=f.c.movement,clipped_updates=f.c.clipped,updates=f.c.updates,baseline=f.c.b,initial_support_jaccard=float(np.mean(inter/np.maximum(1,union))),code_stats=dict(f.code_stats)))
 for phase in ('old','new'):
  stream=data.records(d,phase,arm=='SH_OLD');learning=not(arm=='NO_OLD' and phase=='old')
  for j,r in enumerate(stream,1):
   ops.check(world=d['seed'],arm=arm,phase=phase,record=j,records=rec,observed_events=obs,queries=queries)
   # Paired exogenous stochastic transition tape, not an arm/phase input feature.
   f.c.rng=np.random.default_rng(d['seed']+f.c.n*41+700000)
   gen,st=f.record(bytes(r['prompt']),bytes(r['answer']),native=learning,controller=learning and arm!='FROZEN',tape=d['seed']+f.c.n*31+900000)
   rec+=1;obs+=len(r['prompt'])+len(r['answer'])+1;private_events+=sum(len(g[3]) for g in gen)
   train_gen['exact']+=sum(g[1] and g[0]==r['answer'] for g in gen);train_gen['EOS']+=sum(g[1] for g in gen);train_gen['truncated']+=sum(g[2] for g in gen);train_gen['count']+=3
   for k in ('actual_loss','gradient_norm','trace_norm','gate_entropy'):stsum[k]+=st[k]
   maxgrad=max(maxgrad,st['gradient_norm']);maxtrace=max(maxtrace,st['trace_norm'])
   intermediate=(phase=='old' and j in (128,512,1024,2048)) or (phase=='new' and j==64)
   if intermediate:
    pools={'old':d['old'][:8],'held':d['held'][:8]}
    if phase=='new':pools['new']=d['new'][:8]
    name=f'{phase.upper()}_{j}';save_probe(f,d,arm,name,pools);checkpoint(name)
   if j%128==0:
    ops.save(f'results/progress/{tag}.json',dict(world=d['seed'],arm=arm,phase=phase,record=j,records=f.c.n,observed_events=f.e.events,teacher_loss_mean=stsum['actual_loss']/f.c.n,max_gradient=maxgrad,max_trace=maxtrace,training_generations=dict(train_gen),resources=ops.stats()))
  save_probe(f,d,arm,phase.upper()+'_END',{'old':d['old'],'new':d['new'],'held':d['held']});checkpoint(phase.upper()+'_END')
  f.e.rest(3600)
  stage='REST1' if phase=='old' else 'FINAL';save_probe(f,d,arm,stage,{'old':d['old'],'new':d['new'],'held':d['held']});checkpoint(stage)
 if arm=='FROZEN':assert np.array_equal(f.c.Wv,start[0]) and np.array_equal(f.c.Wg,start[1])
 result=dict(seed=d['seed'],arm=arm,records=f.c.n,observed_events=obs-obs0,private_query_events=private_events-private0,CPU_seconds=time.process_time()-t0,summary=summaries,teacher_statistics_mean={k:v/f.c.n for k,v in stsum.items()},max_gradient=maxgrad,max_trace=maxtrace,training_generations=dict(train_gen),birth_arrays={k:hashlib.sha256(getattr(f.e.bank,k).tobytes()).hexdigest() for k in ('Q','T','reader_Q','kc_side')})
 ops.save(f'results/histories/{tag}.json',result);ops.check('HISTORY_COMPLETE',history_CPU_seconds=result['CPU_seconds'],**{k:result[k] for k in ('seed','arm','records','observed_events')});return result

def lock():
 files=[*P.glob('*.py'),*P.glob('design/*'),*P.glob('assets/*'),*P.glob('manifests/*')]
 files+=[Path(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(getattr(m,'__file__')).startswith(str(R)) and str(getattr(m,'__file__')).endswith('.py')]
 # Birth loader recursively reads the canonical vendor package. Include its text/numeric assets.
 for base in [R/'minifly_a_v3',R/'r_center_core_v1/vendor']:
  files += [p for p in base.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz','.npy','.csv','.pkl') and 'scratch' not in p.parts and '__pycache__' not in p.parts]
 out={str(p.relative_to(R)):ops.sha(p) for p in set(files)};ops.save('SOURCE_LOCK.json',out)
 return out

def main():
 global rec,obs
 ops.begin();status='FAILED';completed=[]
 try:
  assert json.loads((P/'cycles/02_QUALIFICATION.json').read_text())['status']=='PASS'
  # Build all rosters before the first main feedback or held probe.
  ds=[data.build(seed) for seed in data.WORLDS]
  for d in ds:ops.save(f'manifests/{d["seed"]}.json',d)
  Fly(data.WORLDS[0]);locked=lock();ops.save('AUTHORIZATION.json',dict(user_request='Do the 3-cycle exercise on V6-R1: design, audit, test run, revise.',scope='bounded engineering harness only; no formal confirmation, push or adoption',worlds=data.WORLDS,arms=data.ARMS))
  for d in ds:
   for arm in data.ARMS:
    result=history(d,arm);completed.append(result['seed'].__str__()+'_'+arm)
    if len(completed)==1:
     st=ops.stats();estimate=st['CPU_seconds']+result['CPU_seconds']*1.5*7+180
     ops.save('results/PROFILE_ADMISSION.json',dict(first_history_CPU=result['CPU_seconds'],conservative_total_CPU=estimate,ceiling=7200,admit=estimate<7200))
     if estimate>=7200:status='BOUNDED_PROFILE_STOP';return
  for path,h in locked.items():assert ops.sha(R/path)==h,path
  status='MAIN_COMPLETE_AWAITING_AUDIT'
 except BaseException as e:
  ops.save('results/FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),records=rec,observed_events=obs,completed_histories=completed));raise
 finally:
  ops.save('results/SESSION_RESULT.json',dict(status=status,completed_histories=completed,main_full_records=rec,main_observed_native_events=obs,private_query_native_events=private_events,scored_queries=queries,resources=ops.stats()));ops.end()
if __name__=='__main__':main()
