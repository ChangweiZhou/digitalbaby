import writerlib as L
from writerlib import np,H,ROOT,F,C,EOS,Arm
import json,time,sys,subprocess,traceback,hashlib,pickle

def qualify():
 L.TIER='qualification';a=Arm();tech=L.data(202610277391);rows=tech['rows'][:8];t0=L.cpu();original=L.digest(a)
 # Directions, API bounds and all byte/EOS vocabulary before learning.
 assert C.shape==(257,16) and len({tuple(r) for r in C})==257 and (C.sum(1)==0).all()
 assert set(range(256))|{EOS}==set(range(C.shape[0]))
 for v in (np.zeros(16),np.linspace(-100,100,16)):
  p,_=L.prob(v)
  for y in range(257):
   assert np.max(abs(2*L.direction(v,p,y,'N')))<=1+1e-14
   assert np.max(abs(L.direction(v,p,y,'D')))<=1+1e-14
 assert L.warm(0)==1/16 and L.warm(1024)==1 and L.warm(2048)==1
 for row in rows:a.train_row(row)
 train_elapsed=L.cpu()-t0
 receipt=L.checkpoint(a,'qualification.pkl');b=L.restore(receipt);assert L.digest(a)==L.digest(b)
 # Roundtrip next event checked without teaching/replaying any trained record.
 aa=a.clone();bb=b.clone();aa._event(0);bb._event(0);assert L.digest(aa)==L.digest(bb)
 row=rows[0];q=L.context(a,row,0);x=L.querycode(q);v=q.values();vv=L.fixedread(q,x,q.time+F.DELTA)
 reader_error=float(np.max(abs(v-vv)));assert reader_error<1e-12
 y=bytes.fromhex(row['answer'])[0]
 z=L.pulse(q,y,'D',0,keep=True);z2=L.pulse(q,y,'N',0,keep=True);zero_error=float(np.max(abs(z['v']-z2['v'])));assert L.digest(z['model'])==L.digest(z2['model'])
 # Check native API sign and synaptic linearity without applying arithmetic features.
 dn=L.pulse(q,y,'D',1,keep=True);nn=L.pulse(q,y,'N',.2,keep=True)
 assert z['time']==dn['time']==nn['time'];assert np.array_equal(z['x'],dn['x']) and np.array_equal(z['x'],nn['x']);assert z['nonvalue']==dn['nonvalue']==nn['nonvalue']
 pz,lz=L.prob(z['v']);pd,ld=L.prob(dn['v']);pn,ln=L.prob(nn['v']);assert ld[y]>lz[y] and ln[y]>lz[y], 'WRITER_SIGN_CHECK'
 # Zero coefficients enabled are precisely inhibited at same target/time.
 zz=q.clone();xx=zz.teach_once(y,np.zeros(16),True);L.COUNTS['shadow_targets']+=1;assert L.digest(zz)==L.digest(z['model'])
 # Clock-matched controls, targets genuinely observed under inhibition.
 aw=a.clone();ob=a.clone();aw.train_row(rows[1],gain=.3);ob.train_row(rows[1],gain=.3,write=False);assert aw.n==ob.n and L.nonvalue(aw)==L.nonvalue(ob)
 idle=a.clone();idle.rest(aw.time-a.time);assert abs(idle.time-aw.time)<1e-8 and idle.n==a.n
 ro=L.digest(a);tp=L.cpu()
 panel=L.panel(a,rows,'qualification');panel_time=L.cpu()-tp;assert L.digest(a)==ro
 # Scalar solver bookkeeping uses a synthetic linear fixture only, no science evidence.
 g,_,trace=L.solve(lambda g:(g,{}),.3,1,1e-12);assert abs(g-.3)<.003
 audit=dict(accepted=True,records=L.COUNTS['qualification_records'],targets=L.COUNTS['qualification_targets'],reader_error=reader_error,zero_error=zero_error,absolute_floor=max(1e-12,100*zero_error),sign_target_logp_change=dict(D=float(ld[y]-lz[y]),N=float(ln[y]-lz[y])),roundtrip=receipt,matching_clock_seconds=aw.time-a.time,measured_train_CPU_per_record=train_elapsed/8,measured_pair_CPU=panel_time/8,full_byte_codebook=True,homeostasis=False,expected_target_before_write=True,zero_inert=True,OBS_counter_and_nonvalue_match=True,IDLE_seconds=True)
 L.save('QUALIFICATION.json',audit);L.status('QUALIFIED')
 # Cost one native shadow pulse charged, not a new full record.
 t=L.cpu();L.pulse(q,y,'D',.5);pulse_time=L.cpu()-t
 estimate=(3472-L.COUNTS['full_records'])*audit['measured_train_CPU_per_record']+(5408-L.COUNTS['own_TF_pairs'])*audit['measured_pair_CPU']+(4096-L.COUNTS['shadow_targets'])*pulse_time
 admission=dict(CPU_spent=L.cpu(),shadow_pulse_CPU=pulse_time,remaining_roster_estimate_CPU=estimate,conservative_CPU_total=L.cpu()+1.5*estimate,cap_CPU=3600,accepted=L.cpu()+1.5*estimate<3600)
 L.save('RESOURCE_ADMISSION.json',admission)
 if not admission['accepted']:raise L.StopProtocol('RESOURCE_ADMISSION_FAILED')
 return a

def preflight(parent):
 L.TIER='address_preflight';results=[]
 for w in (202610277311,202610277312):
  d=L.data(w);codes=[];targets=[];ids=[]
  for row in d['rows']:
   q=parent.clone();q._prompt(bytes.fromhex(row['prompt']))
   for site,y in enumerate([*bytes.fromhex(row['answer']),EOS]):
    codes.append(L.querycode(q));targets.append(y);ids.append((row['id'],site))
    if site==0:q._event(y)
   L.COUNTS['address_episodes']+=1
  X=np.array(codes);G=X@X.T;sz=X.sum(1);den=sz[:,None]+sz[None,:]-G;J=np.divide(G,den,out=np.zeros_like(G),where=den!=0);upper=np.triu_indices(len(X),1);conflicts=[]
  for i,j in zip(*upper):
   if np.array_equal(X[i],X[j]) and targets[i]!=targets[j]:conflicts.append(dict(a=ids[i],b=ids[j],targets=[targets[i],targets[j]]))
  path=f'local/address_{w}.npz';np.savez_compressed(H/path,codes=X,targets=np.array(targets));results.append(dict(world=w,contexts=len(X),conflicts=conflicts,Jaccard_quantiles=np.quantile(J[upper],[0,.25,.5,.75,1]),path=path,sha256=L.sha(H/path)))
 L.save('ADDRESS_PREFLIGHT.json',results)
 if any(r['conflicts'] for r in results):raise L.StopProtocol('CANNOT_TEST_AS_SPECIFIED_ADDRESS_CONFLICTS')

def calibration():
 L.TIER='calibration';w=202610277390;d=L.data(w);rows=d['rows'][:32];labels=np.repeat(d['alphabet']['reply'],2);labels=L.rng(w,'calibration-labels').permutation(labels)
 for r,y in zip(rows,labels):r['answer']=bytes([int(y)]).hex();r['cohort']='CAL'
 L.save('CALIBRATION_DATA.json',rows);a=Arm()
 for pas in range(4):
  for ix in L.rng(w,f'CAL-order-{pas}').permutation(32):a.train_row(rows[int(ix)])
 receipt=L.checkpoint(a,'calibration.pkl');L.save('CALIBRATION_PARENT.json',receipt)
 contexts=[];floor=json.loads((H/'QUALIFICATION.json').read_text())['absolute_floor']
 for row in L.selected(rows,w,'calibration',8):
  for site,y in enumerate([*bytes.fromhex(row['answer']),EOS]):
   q=L.context(a,row,site);z=L.pulse(q,y,'D',0);contexts.append(dict(pre=q,y=y,zero=z,row=row['id'],site=site))
 cache={};distribution_rows=[]
 def doses(form,gain):
  key=(form,float(gain))
  if key not in cache:
   dd=[]
   for c in contexts:
    if gain==0:dd.append(0.);continue
    out=L.pulse(c['pre'],c['y'],form,gain)
    assert out['time']==c['zero']['time'] and np.array_equal(out['x'],c['zero']['x']) and out['nonvalue']==c['zero']['nonvalue']
    dd.append(L.displacement(out['v'],c['zero']['v']))
   cache[key]=dd;distribution_rows.append(dict(form=form,gain=gain,doses=dd,median=float(np.median(dd))))
   L.save('CALIBRATION_DOSE_TRACE.json',distribution_rows)
  return float(np.median(cache[key])),dict(doses=cache[key])
 low=doses('N',.2)[0];high=min(doses('N',2)[0],doses('D',1)[0]);L.save('CALIBRATION_ANCHORS.json',dict(low=low,high=high,floor=floor))
 if not low>floor or not high>low+max(floor,.01*low):raise L.StopProtocol('CALIBRATION_LEVELS_UNRESOLVED')
 gains={};solves={}
 for form,bound in [('N',2),('D',1)]:
  for level,target in [('low',low),('high',high)]:
   g,out,tr=L.solve(lambda gain:doses(form,gain),target,bound,floor);name=form+'-'+level;gains[name]=g;solves[name]=dict(gain=g,target=target,median=float(np.median(out['doses'])),doses=out['doses'],evaluations=tr)
 L.save('CALIBRATION_LOCK.json',dict(accepted=True,low=low,high=high,ratio=high/low,gains=gains,solves=solves,contexts=[dict(id=c['row'],site=c['site'],y=c['y']) for c in contexts],parent=receipt,data_sha256=L.sha(H/'CALIBRATION_DATA.json')));L.status('CALIBRATED');return a

def pulse_assay(parent,d,cal):
 L.TIER='matched_pulses';w=d['world'];sent=L.selected(d['rows'][:64],w,'OLD-sentinel',8);results=[];floor=json.loads((H/'QUALIFICATION.json').read_text())['absolute_floor']
 for row in L.selected(d['rows'][64:],w,'NEW-pulse',8):
  for site,y in enumerate([*bytes.fromhex(row['answer']),EOS]):
   pre=L.context(parent,row,site);z=L.pulse(pre,y,'D',0,keep=True);zpanel=L.panel(z['model'],sent,f'pulse_{w}_{row["id"].split(":")[-1]}_{site}_zero')
   maxes={};maxout={}
   for form,bound in [('N',2),('D',1)]:
    o=L.pulse(pre,y,form,bound,force_warm=L.warm(pre.n),keep=True);maxes[form]=L.displacement(o['v'],z['v']);maxout[form]=o
   A=min(maxes.values());scale=min(1,A/cal['high']);rr=dict(id=row['id'],site=site,y=y,attainable=A,scale=scale,max_d=maxes,zero=zpanel,conditions=dict(native_time=z['time'],code_hash=hashlib.sha256(z['x'].tobytes()).hexdigest()),interventions={})
   _,zero_lp=L.prob(z['v']);zero_margin=float(zero_lp[y]-np.max(np.delete(zero_lp,y)))
   if A<=floor:rr['unidentifiable']=True;results.append(rr);L.save(f'PULSES_{w}.json',results);continue
   for form,bound in [('N',2),('D',1)]:
    for level in ('low','high'):
     target=L.warm(pre.n)*scale*cal[level]
     def response(g):
      if g==0:return 0.,z
      o=L.pulse(pre,y,form,g,force_warm=L.warm(pre.n),keep=True);assert o['time']==z['time'] and np.array_equal(o['x'],z['x']) and o['nonvalue']==z['nonvalue'];return L.displacement(o['v'],z['v']),o
     try:g,out,tr=L.solve(response,target,bound,floor)
     except L.StopProtocol as e:
      rr['interventions'][form+'-'+level]=dict(matched=False,error=str(e));continue
     _,lp=L.prob(out['v']);name=form+'-'+level;rec=L.panel(out['model'],sent,f'pulse_{w}_{row["id"].split(":")[-1]}_{site}_{name}')
     rr['interventions'][name]=dict(matched=True,gain=g,target_d=target,actual_d=L.displacement(out['v'],z['v']),trace=tr,target_logp_change=float(lp[y]-zero_lp[y]),margin_change=float(lp[y]-np.max(np.delete(lp,y))-zero_margin),coef_RMS=float(np.sqrt(np.mean(out['coef']**2))),synaptic_L2=float(np.linalg.norm(L.synstate(out['model'])-L.synstate(z['model']))),sentinels=rec)
   results.append(rr);L.save(f'PULSES_{w}.json',results);L.log('pulse_context',world=w,id=row['id'],site=site)
 return results

def mainrun():
 cal=json.loads((H/'CALIBRATION_LOCK.json').read_text());allresults=[]
 for w in (202610277311,202610277312):
  L.TIER='OLD';d=L.data(w);a=Arm()
  for pas,order in enumerate(d['old_orders'],1):
   for ix in order:a.train_row(d['rows'][ix])
   L.status('RUNNING_OLD',world=w,completed_pass=pas)
  rec=L.checkpoint(a,f'OLD_{w}.pkl');L.save(f'OLD_{w}.json',rec);pre=L.panel(a,d['rows'],f'{w}_pre');oldtime=a.time;initial=set(pre['correct_ids'])&{r['id'] for r in d['rows'][:64]}
  L.save(f'PRE_{w}.json',dict(panel=pre,initial_OLD_correct=sorted(initial),native_time=oldtime,n=a.n,checkpoint=rec))
  pulse_assay(a,d,cal)
  wr=dict(world=w,pre=pre,initial_OLD_correct=sorted(initial),branches={});clock={}
  for arm in (*cal['gains'],'OBS'):
   L.TIER='NEW_'+arm;m=a.clone();form=arm[0] if arm!='OBS' else 'D';gain=cal['gains'][arm] if arm!='OBS' else 1.;checkpoints={};eventclock=[]
   for pas,order in enumerate(d['new_orders'],1):
    for i,ix in enumerate(order):
     row=d['rows'][64+ix];sites=m.train_row(row,form,gain,arm!='OBS',measure=pas in (1,2,4,8) and i==15 and arm!='OBS');eventclock.append(dict(id=row['id'],times=[s['time'] for s in sites],n=[s['n'] for s in sites],codes=[s['code_hash'] for s in sites]))
    if pas in (1,2,4,8):
     p=L.panel(m,d['rows'],f'{w}_{arm}_{pas}');p.update(retained=len(initial&set(p['correct_ids'])),lost=len(initial-set(p['correct_ids'])),native_time=m.time,delta_t=m.time-oldtime,n=m.n,nonvalue_hash=L.nonvalue(m));checkpoints[str(pas)]=p
     if arm==next(iter(cal['gains'])):clock[str(pas)]=p
     else:assert p['native_time']==clock[str(pas)]['native_time'] and p['n']==clock[str(pas)]['n'] and p['nonvalue_hash']==clock[str(pas)]['nonvalue_hash']
   L.save(f'EVENT_CLOCK_{w}_{arm}.json',eventclock)
   if arm!=next(iter(cal['gains'])):assert eventclock==json.loads((H/f'EVENT_CLOCK_{w}_{next(iter(cal["gains"]))}.json').read_text())
   wr['branches'][arm]=checkpoints;L.save(f'WORLD_{w}.json',wr);L.checkpoint(m,f'FINAL_{w}_{arm}.pkl');L.status('RUNNING_NEW',world=w,completed_arm=arm)
  L.TIER='IDLE';idlepanels={}
  for pas,p in clock.items():
   idle=a.clone();idle.rest(p['delta_t']);assert abs(idle.time-p['native_time'])<1e-8 and idle.n==a.n
   pp=L.panel(idle,d['rows'],f'{w}_IDLE_{pas}');pp.update(retained=len(initial&set(pp['correct_ids'])),lost=len(initial-set(pp['correct_ids'])),delta_t=p['delta_t'],native_time=idle.time,n=idle.n);idlepanels[pas]=pp
  wr['branches']['IDLE']=idlepanels;L.save(f'WORLD_{w}.json',wr);allresults.append(wr)
 L.save('RESULTS.json',allresults);L.status('MAIN_COMPLETED')

def run(stage):
 cafe=subprocess.Popen(['caffeinate','-i','-w',str(__import__('os').getpid())])
 try:
  if stage=='qualify':
   manifest={str(w):L.data(w) for w in (202610277311,202610277312,202610277390,202610277391)};L.save('MANIFEST.json',manifest);L.save('SOURCE_LOCK.json',L.lock());a=qualify();preflight(a);calibration();L.status('CYCLE_2_COMPLETE')
  elif stage=='main':
   assert all(L.sha(ROOT/f)==h for f,h in json.loads((H/'SOURCE_LOCK.json').read_text()).items()),'SOURCE_LOCK_CHANGED'
   mainrun()
  else:raise ValueError(stage)
 except BaseException as e:
  L.save('STOP.json',dict(reason=str(e),type=type(e).__name__,traceback=traceback.format_exc(),counts=dict(L.COUNTS),CPU=L.cpu(),tier=L.TIER));L.status('STOPPED',reason=str(e));print(traceback.format_exc(),flush=True)
 finally:
  cafe.terminate();cafe.wait();L.status(json.loads((H/'STATUS.json').read_text())['status'],reason=json.loads((H/'STATUS.json').read_text()).get('reason'));print(json.dumps(json.loads((H/'COUNTS.json').read_text())),flush=True)
if __name__=='__main__':run(sys.argv[1])
