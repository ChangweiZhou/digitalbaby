"""Cycle 3: ten fixed shared-controller fits on immutable native snapshots."""
import os,sys,time,json,pickle,subprocess,traceback
import sharedlib as s
import numpy as np
ACTORS={};TAPES={};CONTEXT={}
def run():
 global ACTORS,TAPES,CONTEXT
 assert not (s.H/'MAIN_STARTED.json').exists(),'No automatic restart/replay'
 q=json.loads((s.H/'QUALIFICATION.json').read_text());assert q['status']=='PASS' and q['main_admitted']
 prior=json.loads((s.H/'COUNTS.json').read_text());s.PRIOR_CPU=prior['CPU'];s.COUNTS.clear();s.COUNTS.update(prior['counts']);s.TIMES.update(prior['timings']);s.W0=time.monotonic()-max(0,time.time()-(s.H/'AUTHORIZATION.json').stat().st_mtime);s.TIER='main'
 for p,h in json.loads((s.H/'QUALIFICATION_SOURCE_LOCK.json').read_text()).items():assert s.sha(s.ROOT/p)==h
 source=s.lock();s.save('MAIN_SOURCE_LOCK.json',source);s.save('MAIN_STARTED.json',dict(CPU=s.cpu(),wall=s.wall(),source_sha256=s.sha(s.H/'MAIN_SOURCE_LOCK.json')))
 design=s.SPEC;datafile=s.ROOT/'minifly_v171_dev_20261026/local/TRAIN_DATA.json';assert s.sha(datafile)==design['inputs_sha256'][str(datafile.relative_to(s.ROOT))];data=json.loads(datafile.read_text());allresults={}
 for world in design['worlds']:
  w=data[str(world)];rows=w['OLD']+w['NEW'];byid={x['id']:x for x in rows};cp=s.ROOT/f'minifly_v171_dev_20261026/local/cal_{world}_final.pkl';assert s.sha(cp)==design['inputs_sha256'][str(cp.relative_to(s.ROOT))]
  with cp.open('rb') as f:model,_=pickle.load(f)
  before=s.sha(cp);cache={};norms=[];cache_t=s.cpu()
  for row in rows:
   r=s.collect(model,bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),row['id']);cache[row['id']]=r;norms.extend(np.linalg.norm(r['prompt_v'],axis=1).tolist())
  scale=float(np.median(norms)/4);fallback=scale==0
  if fallback:scale=1.
  carrier=np.full(16,scale);cache_receipt=s.checkpoint(cache,f'cache/{world}.pkl')
  for r in cache.values():
   for k in ('prompt_v','boundary_v','target_v'):r[k].flags.writeable=False
  ACTORS={arm:s.Shared(world,arm,carrier) for arm in s.ARMS};assert len({c.state_hash() for arm,c in ACTORS.items() if arm!='ADAM3-R0'})==1
  native=s.evaluate(model,{'NATIVE':None},rows,f'{world}_NATIVE')
  expected=4 if world==202610269711 else 11;assert native['NATIVE']['correct']==expected,('PARENT_AUTHENTICATION',world)
  panels={'BIRTH':s.evaluate(model,ACTORS,rows,f'{world}_BIRTH')};checkpoints=[s.checkpoint(ACTORS,f'checkpoints/{world}_BIRTH.pkl')];receipts=[];recordlog=s.H/'local'/f'{world}_RECORDS.jsonl'
  s.log('WORLD_START',world=world,carrier_scale=scale,cache_CPU=s.cpu()-cache_t,cache_receipt=cache_receipt)
  for phase in ('OLD','NEW'):
   for p in range(1,17):
    order=[w['OLD'][int(j)] for j in w['orders'][(p-1)%8]] if phase=='OLD' else w['NEW']
    TAPES={arm:s.parent_support.Tape(f'training/{world}_{phase}_{p:02d}_{arm}') for arm in ACTORS}
    for i,row in enumerate(order):
     CONTEXT=dict(world=world,phase=phase,pass_no=p,record_index=i,id=row['id'])
     updates={}
     for arm,c in ACTORS.items():
      CONTEXT['arm']=arm;out,sites=s.fit_record(c,cache[row['id']]);updates[arm]=out
      for md,arrays in sites:TAPES[arm].add(dict(CONTEXT,**md),**arrays)
     # Save current parameters/optimizer clocks before assertions or later panel calls.
     s.checkpoint(dict(context=CONTEXT,actors=ACTORS),f'{world}_LATEST.pkl')
     with recordlog.open('a') as f:f.write(json.dumps(s.plain(dict(CONTEXT,updates=updates)),allow_nan=False)+'\n')
     assert all(not c.h.any() and not c.frames for c in ACTORS.values())
     if i%16==0:s.save('ACTIVE.json',dict(CONTEXT,main_fit_records=s.COUNTS['main_fit_records'],CPU=s.cpu(),wall=s.wall()))
    for t in TAPES.values():receipts.append(t.flush())
    s.log('PASS_COMMIT',world=world,phase=phase,pass_no=p,main_records=s.COUNTS['main_fit_records'])
    if (phase,p) in [('OLD',8),('OLD',16),('NEW',8),('NEW',16)]:
     tag=f'{phase}{p}';panels[tag]=s.evaluate(model,ACTORS,rows,f'{world}_{tag}');checkpoints.append(s.checkpoint(ACTORS,f'checkpoints/{world}_{tag}.pkl'))
    assert all(s.sha(s.ROOT/f)==h for f,h in source.items()),'SOURCE_CHANGED'
    assert sum(p.stat().st_size for p in (s.H/'local').rglob('*') if p.is_file())<268435456,'OUTPUT_CAP'
    s.finish('CYCLE3_FIXED_FITS_RUNNING');s.budget()
  delayed=model.clone();delayed.rest(128);native_delay=s.evaluate(delayed,{'NATIVE':None},rows,f'{world}_NATIVE_DELAY');panels['POST_IDLE128']=s.evaluate(delayed,ACTORS,rows,f'{world}_POST_IDLE128')
  assert s.sha(cp)==before
  result=dict(world=world,native=native['NATIVE'],native_delayed=native_delay['NATIVE'],panels=panels,training_array_receipts=receipts,checkpoint_receipts=checkpoints,cache_receipt=cache_receipt,constant_scale=scale,constant_scale_fallback=fallback,controller_target_clocks={k:c.n_train for k,c in ACTORS.items()},Adam_record_clocks={k:c.opt_t for k,c in ACTORS.items()},parent_checkpoint_sha256=before)
  allresults[str(world)]=result;s.save(f'worlds/{world}.json',result);s.log('WORLD_COMPLETE',world=world,final_correct={k:v['correct'] for k,v in panels['NEW16'].items()});ACTORS={};TAPES={}
 assert s.COUNTS['main_fit_records']==12800 and s.COUNTS['main_fit_targets']==36800
 assert s.COUNTS['main_cache_episodes']==160 and s.COUNTS['main_cache_target_sites']==460 and s.COUNTS['main_cache_prompt_bytes']==810
 assert s.COUNTS['paired_queries']==5120+2
 s.save('RUN_RESULTS.json',dict(identity='SP-COND-V1',status='FIXED_FITS_COMPLETE_AWAITING_AUDIT',worlds=allresults,counts=dict(s.COUNTS),CPU=s.cpu(),wall=s.wall(),peak_RSS_bytes=s.rss(),main_native_teaching_records=0,held_assessment_decisions=0,main_arms_all_completed=True,new_core_adopted=False))
 s.finish('FIXED_FITS_COMPLETE_AWAITING_AUDIT');print(json.dumps(dict(status='COMPLETE',CPU=s.cpu(),final={w:{a:x['correct'] for a,x in r['panels']['NEW16'].items()} for w,r in allresults.items()})))
if __name__=='__main__':
 awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);s.save('LAUNCH.json',dict(pid=os.getpid(),caffeinate_pid=awake.pid,runtime=sys.executable))
 try:run()
 except Exception as e:
  try:s.checkpoint(dict(context=CONTEXT,actors=ACTORS,pending_tapes=TAPES),'FAILURE_STATE.pkl')
  finally:
   s.save('MAIN_FAILURE.json',dict(error=str(e),type=type(e).__name__,context=CONTEXT,counts=dict(s.COUNTS),CPU=s.cpu(),wall=s.wall(),traceback=traceback.format_exc().replace(str(s.ROOT),'$REPO').replace('/Users/pencilbard','$HOME')));s.finish('MAIN_FAILED_NO_AUTOMATIC_RETRY')
  raise
 finally:awake.terminate();awake.wait(timeout=5)
