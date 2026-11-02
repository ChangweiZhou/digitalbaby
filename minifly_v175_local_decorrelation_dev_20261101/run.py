import v175lib as V
from v175lib import W,np,H,ROOT,F,Model,COUNTS,TIMES
import json,time,sys,subprocess,traceback,hashlib,pickle,ast

def sourceaudit():
 tree=ast.parse((H/'v175lib.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='update_edges')
 assert [a.arg for a in fn.args.args]==['weights','i','j','z0'];assert not any(isinstance(n,ast.Name) and n.id in {'row','answer','target','phase','world','y','lp','coef'} for n in ast.walk(fn))
 old=json.loads((ROOT/'minifly_writer_interference_dev_v1_20261027/SOURCE_LOCK.json').read_text());assert all(W.sha(ROOT/p)==h for p,h in old.items())
 files=dict(old)
 for mod in list(sys.modules.values()):
  p=getattr(mod,'__file__',None)
  if p and __import__('pathlib').Path(p).resolve().is_relative_to(ROOT):
   p=__import__('pathlib').Path(p).resolve()
   if p.suffix=='.py':files[str(p.relative_to(ROOT))]=W.sha(p)
 for p in list(H.glob('*.py'))+[H/'AUTHORIZATION.json',H/'EXECUTION_SPEC.json',H/'MANIFEST.json']:
  files[str(p.relative_to(ROOT))]=W.sha(p)
 W.save('SOURCE_LOCK.json',files);W.save('SOURCE_AUDIT.json',dict(accepted=True,original_native_source_lock_preserved=True,update_function_has_only_cue_activity_and_local_edge_state=True,label_or_phase_arguments=False,encoder_has_no_target_argument=True,review_type='separately implemented structural/AST audit; not an independent human review'))

def c0():
 W.TIER='C0';manifest={str(w):W.data(w) for w in (202611017501,202611017511,202611017512)};W.save('MANIFEST.json',manifest);sourceaudit()
 a=Model(False);gprompts=V.rng(202611017501,'gamma-prompts').integers(0,256,(64,5));rows=[dict(id=f'gamma:{i}',prompt=bytes(x.tolist()).hex(),answer='',cohort='CAL') for i,x in enumerate(gprompts)]
 _,raw=V.codes(a,rows);side=a.bank[0].fly.m.kc_side;stds=[[float(np.std(u[(side==s)&(u>0)])) for u in raw] for s in (0,1)];gamma=np.median(stds,axis=1)/4;assert np.isfinite(gamma).all() and (gamma>0).all()
 W.save('GAMMA_LOCK.json',dict(gamma=gamma,positive_drive_stds=stds,unlabeled_prompt_hex=[r['prompt'] for r in rows],world=202611017501,targets_used=0,source_lock_sha256=W.sha(H/'SOURCE_LOCK.json')))
 feasibility=[]
 for w in (202611017511,202611017512):
  d=manifest[str(w)];X,_=V.codes(a,d['rows']);groups={}
  for x,r in zip(X,d['rows']):groups.setdefault(x.tobytes(),[]).append(r['answer'])
  ceiling=sum(max(__import__('collections').Counter(labels).values()) for labels in groups.values());floor=max(__import__('collections').Counter(r['answer'] for r in d['rows']).values());conf=V.conflicts(X,d['rows'])
  p=f'local/C0_codes_{w}.npz';(H/'local').mkdir(exist_ok=True);np.savez_compressed(H/p,codes=X);feasibility.append(dict(world=w,ceiling=ceiling,floor=floor,conflicts=conf,codes_path=p,sha256=W.sha(H/p)))
 W.save('CYCLE_0_TESTBED.json',dict(accepted=all(r['ceiling']>=72 and r['floor']<=8 and not r['conflicts'] for r in feasibility),worlds=feasibility,evaluator_only=True))
 if not all(r['ceiling']>=72 and r['floor']<=8 and not r['conflicts'] for r in feasibility):raise W.StopProtocol('CANNOT_TEST_CLAIM_AS_SPECIFIED')
 V.status('C0_PASSED')

def equal_native(a,b):
 assert a.time==b.time and a.n==b.n and a.records==b.records
 for s,t in zip(a.bank,b.bank):
  for attr in ('fast','slow','adapt'):assert np.array_equal(getattr(s.fly.m,attr),getattr(t.fly.m,attr))
  assert s.fly.m.elapsed==t.fly.m.elapsed
 assert a.fe is None and b.fe is None or pickle.dumps(a.fe)==pickle.dumps(b.fe)

def technical(resume=False):
 W.TIER='technical';gamma=json.loads((H/'GAMMA_LOCK.json').read_text())['gamma'];rows=W.data(202611017501)['rows'];checks={}
 if resume:
  assert COUNTS['technical_records']==8
  checks.update(BASE_literal_D_high_parity=True,L_zero_eta_zero_exact_parity=True)
 else:
  base=Model(False);ref=W.Arm()
  for row in rows[:2]:base.train_row(row);ref.train_row(row);equal_native(base,ref)
  checks['BASE_literal_D_high_parity']=True
  off=Model(True,gamma);off.suppress_updates=True;bas=Model(False)
  for row in rows[2:4]:off.train_row(row);bas.train_row(row);equal_native(off,bas)
  checks['L_zero_eta_zero_exact_parity']=True
 # Independent scalar edge update reference, no science fixtures involved.
 m=Model(True,gamma);side=m.bank[0].fly.m.kc_side;assert np.all(np.bincount(np.r_[m.i,m.j],minlength=len(side))==16)
 z=V.rng(202611017501,'technical-activity').integers(0,2,len(side)).astype(float);test=V.rng(202611017501,'technical-L').random(len(m.L)).astype(np.float32);expect=np.array([min(1.,(1-1/2048)*float(v)+(1/32)*z[int(i)]*z[int(j)]) for v,i,j in zip(test,m.i,m.j)],dtype=np.float32);V.update_edges(test,m.i,m.j,z);assert np.array_equal(test,expect)
 q=V.rng(202611017501,'gamma-prompts').integers(0,256,88).astype(float);m.L[:]=test;x,u=m.encode(q);z0,_=F.FastMemory.encode(m,q)
 assert all(x[side==s].sum()==z0[side==s].sum() for s in (0,1))
 inhibit=np.zeros(len(z0))
 for i,j,weight in zip(m.i,m.j,m.L):inhibit[i]+=float(weight)*z0[j];inhibit[j]+=float(weight)*z0[i]
 score=u-m.gamma[side]*inhibit;reference=np.zeros(len(z0))
 for ss in (0,1):
  ix=[int(i) for i in range(len(z0)) if side[i]==ss and u[i]>0];ix.sort(key=lambda i:(-score[i],i));reference[ix[:m.side_n[ss]]]=1
 assert np.array_equal(x,reference);checks['local_topology_and_one_pass_reference']=True
 # Same cue, different observed answers: the sole cue-derived L update is equal.
 la=Model(True,gamma);lb=la.clone();ra=dict(rows[4]);rb=dict(ra);rb['answer']=bytes([(bytes.fromhex(ra['answer'])[0]+1)%256]).hex();la.train_row(ra);lb.train_row(rb);assert np.array_equal(la.L,lb.L) and la.update_count==lb.update_count==1
 checks['labels_do_not_enter_L_update']=True
 # Inhibited teaching preserves the passive native event trajectory, not frozen values.
 a=la.clone();b=la.clone();row=rows[5];a.train_row(row,write=False);b._prompt(bytes.fromhex(row['prompt']));cue=b.cue_before_target()
 for y in [*bytes.fromhex(row['answer']),256]:b._event(y);b.n+=1
 b.after_observed_record(cue);b.records+=1;COUNTS['full_records']+=1;COUNTS['technical_records']+=1;COUNTS['technical_targets']+=2
 equal_native(a,b);assert np.array_equal(a.L,b.L);checks['inhibited_equals_passive_native_events']=True
 # Deterministic clone and complete mutable checkpoint with next-update parity.
 a=la.clone();b=a.clone();a.train_row(rows[6]);b.train_row(rows[6]);assert V.digest(a)==V.digest(b)
 receipt=W.checkpoint(a,'technical_roundtrip.pkl');b=W.restore(receipt);assert V.digest(a)==V.digest(b);a.train_row(rows[7]);b.train_row(rows[7]);assert V.digest(a)==V.digest(b)
 checks['clone_checkpoint_next_event_roundtrip']=True
 c=a.clone();c.L[0]=1-c.L[0];assert not np.array_equal(c.L,a.L);checks['clone_L_independence']=True
 # Empty prompt and empty output remain generic byte/EOS inputs.
 empty=dict(id='technical:empty',prompt='',answer='',cohort='TECH');a.train_row(empty);b.train_row(empty);assert V.digest(a)==V.digest(b)
 q=W.context(a,rows[8],0);x=W.querycode(q);v=q.values();vv=W.fixedread(q,x,q.time+F.DELTA);err=float(np.max(abs(v-vv)));assert err<1e-12
 p,lp=W.prob(v);y=bytes.fromhex(rows[8]['answer'])[0];c=W.warm(q.n)*W.direction(v,p,y,'D');zero=q.clone();written=q.clone();zero.teach_once(y,np.zeros(16),False);written.teach_once(y,c,True);COUNTS['shadow_targets']+=2;assert zero.time==written.time and np.array_equal(W.querycode(zero),W.querycode(written));assert W.nonvalue(zero)==W.nonvalue(written)
 x=W.querycode(zero);z=W.fixedread(zero,x,zero.time+F.DELTA);v=W.fixedread(written,x,written.time+F.DELTA);assert V.self_margin(v,y)>V.self_margin(z,y);checks['sign_fixed_condition_clock']=True
 before=V.digest(a);pt=V.cpu();W.panel(a,rows[:4],'technical');paircost=(V.cpu()-pt)/4;assert V.digest(a)==before;checks['readonly_full_digest']=True
 nc=COUNTS['technical_records'];assert nc<=32;checks['technical_records_within32']=True
 base_only=W.COUNTS['technical_records'];traincost=(TIMES['BASE_train_CPU']+TIMES['LPS_train_CPU']+TIMES['train_CPU'])/base_only
 remaining_estimate=4608*traincost+(1920+512)*paircost+100;admission=dict(accepted=V.cpu()+1.5*remaining_estimate<7200,conservative_total_CPU=V.cpu()+1.5*remaining_estimate,measured_full_record_CPU=traincost,measured_pair_CPU=paircost,multiplier=1.5,cap_CPU=7200)
 W.save('RESOURCE_ADMISSION.json',admission);W.save('CYCLE_1_QUALIFICATION.json',dict(accepted=all(checks.values()),checks=checks,technical_records=nc,reader_parity_error=err,roundtrip=receipt,graph_edges=len(m.L),L_bytes=m.L.nbytes,topology_bytes=m.i.nbytes+m.j.nbytes,gamma=gamma))
 if not admission['accepted']:raise W.StopProtocol('RESOURCE_ADMISSION_FAILED')
 V.status('C1_PASSED')

def snapshot(m,rows,world,arm,stage):
 rec=W.panel(m,rows,f'{world}_{arm}_{stage}');X,_=V.codes(m,rows);p=f'local/codes_{world}_{arm}_{stage}.npz';np.savez_compressed(H/p,codes=X);rec.update(code_path=p,code_sha256=W.sha(H/p),n=m.n,native_time=m.time,L_updates=m.update_count,L_nonzero=0 if m.L is None else int(np.count_nonzero(m.L)),L_max=0 if m.L is None else float(np.max(m.L)),conflicts=V.conflicts(X,rows));return rec

def mainrun():
 gamma=json.loads((H/'GAMMA_LOCK.json').read_text())['gamma'];manifest=json.loads((H/'MANIFEST.json').read_text());allworlds=[]
 for world in (202611017511,202611017512):
  d=manifest[str(world)];rows=d['rows'];sent=V.selected(rows[:64],world,'OLD-sentinel',8);worldresult=dict(world=world,arms={})
  for arm in ('BASE','LPS-16'):
   W.TIER='main_'+arm;m=Model(arm!='BASE',gamma);result=dict(panels={},shadows=[]);start=V.cpu();train0=TIMES[('LPS' if m.enabled else 'BASE')+'_train_CPU'];result['panels']['INITIAL']=snapshot(m,rows,world,arm,'INITIAL')
   for pas,order in enumerate(d['old_orders'],1):
    for ix in order:m.train_row(rows[ix])
    if pas in (8,16):
     stage='OLD'+str(pas);result['panels'][stage]=snapshot(m,rows,world,arm,stage);result[stage+'_checkpoint']=W.checkpoint(m,f'{world}_{arm}_{stage}.pkl');W.save(f'ACTIVE_{world}_{arm}.json',result)
    V.status('RUNNING_OLD',world=world,arm=arm,completed_pass=pas)
   for pas,order in enumerate(d['new_orders'],1):
    for ordinal,ix in enumerate(order,1):
     callback=None
     if pas==1 and ordinal in (1,3,5,7,9,11,13,15):
      def callback(pre,row,y,coef,ordinal=ordinal):
       r=V.shadow(pre,row,y,coef,sent,f'shadow_{world}_{arm}_{ordinal}');result['shadows'].append(r);W.save(f'ACTIVE_{world}_{arm}.json',result)
     m.train_row(rows[64+ix],shadow=callback)
    if pas in (4,8):
     stage='NEW4' if pas==4 else 'FINAL';result['panels'][stage]=snapshot(m,rows,world,arm,stage)
    V.status('RUNNING_NEW',world=world,arm=arm,completed_pass=pas)
   result['FINAL_checkpoint']=W.checkpoint(m,f'{world}_{arm}_FINAL.pkl');rest=m.clone();rest.rest(128*F.DELTA);result['panels']['REST128']=snapshot(rest,rows,world,arm,'REST128');result['life_train_CPU']=TIMES[('LPS' if m.enabled else 'BASE')+'_train_CPU']-train0;result['life_total_CPU']=V.cpu()-start
   worldresult['arms'][arm]=result;W.save(f'WORLD_{world}.json',worldresult);V.status('ARM_COMPLETED',world=world,arm=arm)
  allworlds.append(worldresult)
 W.save('RESULTS.json',allworlds);V.status('C2_COMPLETED')

def run(stage):
 cafe=subprocess.Popen(['caffeinate','-i','-w',str(__import__('os').getpid())])
 try:
  if stage=='qualify':c0();technical()
  elif stage=='continue_technical':sourceaudit();technical(resume=True)
  elif stage=='main':
   assert all(W.sha(ROOT/p)==h for p,h in json.loads((H/'SOURCE_LOCK.json').read_text()).items());assert json.loads((H/'CYCLE_1_QUALIFICATION.json').read_text())['accepted'];mainrun()
  else:raise ValueError(stage)
 except BaseException as e:
  rec=None
  if V.ACTIVE is not None:
   try:rec=W.checkpoint(V.ACTIVE,'FAILED_PARTIAL_STATE.pkl')
   except BaseException:pass
  W.save('FAILURE.json',dict(error=str(e),type=type(e).__name__,traceback=traceback.format_exc(),tier=W.TIER,preserved_partial_state=rec,counts=dict(COUNTS)));V.status('STOPPED',reason=str(e));print(traceback.format_exc(),flush=True)
 finally:
  cafe.terminate();cafe.wait();st=json.loads((H/'STATUS.json').read_text());V.status(st['status'],reason=st.get('reason'));print(json.dumps(json.loads((H/'COUNTS.json').read_text())),flush=True)
if __name__=='__main__':run(sys.argv[1])
