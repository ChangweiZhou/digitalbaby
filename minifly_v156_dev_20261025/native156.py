"""Evaluator only. Native training equations are imported unchanged from V154/V152."""
import sys,math,time,pickle
import numpy as np
import support156 as s
sys.path.insert(0,str(s.ROOT/'minifly_v154_dev_20261025'))
import native154 as n
n.parent.COUNTS=s.COUNTS;n.parent.TIMES=s.TIMES
C=n.C;A=(C-C.mean(1,keepdims=True))/4;EOS=256
def load_state(front,world,state):
 j=s.load(s.ROOT/f'minifly_v154_dev_20261025/groups/N_2B_{front}_{world}/RESULT.json')['states'][state]
 p=s.ROOT/'minifly_v154_dev_20261025'/j['path'];assert s.sha(p)==j['sha256']
 with p.open('rb') as f:m=pickle.load(f)
 assert m.variant==front and m.arm=='N_2B' and np.array_equal(m.q,np.zeros(257))
 assert len(m.bank)==16 and not m.homeo and m.fresh_adapt
 return m
def read(m):
 """Exactly FastMemory.values, with clipping observability on the same read clones."""
 out=[];clips=np.zeros(4,dtype=np.int64);f=m.fe.clone();f.advance(m.time+n.DELTA);x=m.encode(f.read())[0]
 for bank in m.bank:
  fly=bank.fly.clone();fly.rest(n.DELTA);mm=fly.m;dx=n.fastmem.NATIVE.common.observed_activity(mm,x[None,:])
  g=(dx*(mm.k0[3]+mm.fast[:,0]))@mm.Q[:,:2]+35.2
  gamma=np.clip(g,0,71.66)-35.2
  assert mm.rep<2
  inp=(dx*(mm.k0[5]+mm.fast[:,1]+mm.projected_slow()))@mm.Q[:,2:]+gamma@mm.MM
  clips+=np.array([(g<0).sum(),(g>71.66).sum(),(inp<-11.2).sum(),(inp>19.96).sum()])
  out.append(float((mm.expression(dx)-fly.reader.predict(dx)).mean()))
 v=np.asarray(out);z=C@((v-v.mean())/4)+m.q;zz=z-z.max();lp=zz-np.log(np.exp(zz).sum())
 s.COUNTS['evaluator_distributions']+=1;s.COUNTS['native_evaluator_distributions']+=1
 return z,lp,clips
def pair(m,row,kind='native'):
 s.COUNTS[kind+'_pairs']+=1;s.budget();p=bytes.fromhex(row['prompt']);a=bytes.fromhex(row['answer'])
 pre=m.clone();pre._prompt(p);own=pre.clone();tf=pre.clone();choices=[];ended=False;payload=[]
 for _ in range(4):
  z,lp,cl=read(own);y=int(z.argmax());choices.append(y)
  if y==EOS:ended=True;break
  payload.append(y);own.step(y)
 zs=[];lps=[];cs=[];ys=list(a)+[EOS]
 for y in ys:
  z,lp,cl=read(tf);zs.append(z);lps.append(lp);cs.append(cl)
  if y!=EOS:tf.step(y)
 z=np.asarray(zs);lp=np.asarray(lps);ix=np.arange(len(ys));mask=z.copy();mask[ix,ys]=-np.inf;rival=mask.argmax(1);margin=z[ix,ys]-z[ix,rival]
 return dict(id=row['id'],output_hex=bytes(payload).hex(),EOS=ended,choices=choices,correct=ended and bytes(payload)==a,z=z,lp=lp,clipping=np.asarray(cs),y=np.asarray(ys),rival=rival,margin=margin)
def evaluate(m,rows,tag,kind='native',save_arrays=True):
 data=[];zs=[];lps=[];clips=[];ys=[];rs=[];ms=[];offset=[0];t=time.process_time()
 for row in rows:
  r=pair(m,row,kind);zs.append(r.pop('z'));lps.append(r.pop('lp'));clips.append(r.pop('clipping'));ys.append(r.pop('y'));rs.append(r.pop('rival'));ms.append(r.pop('margin'));offset.append(offset[-1]+len(ms[-1]));data.append(r)
 out=dict(tag=tag,rows=data,z=np.concatenate(zs),lp=np.concatenate(lps),clipping=np.concatenate(clips),y=np.concatenate(ys),rival=np.concatenate(rs),margin=np.concatenate(ms),offsets=np.asarray(offset),correct=np.asarray([r['correct'] for r in data],bool))
 first=out['z'][out['offsets'][:-1]];common=first.mean(0);eos=out['offsets'][1:]-1
 out['shared']=dict(common_logits=common,mean_payload_EOS_contrast=float(np.mean(first[np.arange(len(rows)),out['y'][out['offsets'][:-1]]]-first[:,EOS])),mean_true_EOS_margin=float(out['margin'][eos].mean()),common_EOS_argmax=int(common.argmax())==EOS)
 if save_arrays:
  path=s.H/'local'/f'{tag}.npz';path.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(path,**{k:out[k] for k in ['z','lp','clipping','y','rival','margin','offsets','correct']},choices=np.asarray([r['choices']+[ -1]*(4-len(r['choices'])) for r in data],dtype=np.int16),EOS=np.asarray([r['EOS'] for r in data]),common_logits=common)
  s.log('PANEL_COMMIT',tag=tag,records=len(rows),correct=int(out['correct'].sum()),targets=len(out['y']),sha256=s.sha(path),evaluation_cpu=time.process_time()-t)
 return out
def qualify_parity(out,front,world,state):
 """Compare existing saved outputs, not a second model trajectory."""
 acq=s.load(s.ROOT/f'minifly_v154_dev_20261025/groups/N_2B_{front}_{world}/ACQUISITION.json')
 tags=['OLD8','BEFORE_NEW'] if state=='OLD_W' else ['POST_IDLE']
 by={r['id']:r for c in acq['curves'] if c['tag'] in tags and c['branch']=='W' for r in c['rows']}
 largest=0.
 for i,r in enumerate(out['rows']):
  old=by[r['id']];assert all(r[k]==old[k] for k in ['output_hex','EOS','choices','correct'])
  lo,hi=out['offsets'][i:i+2]
  for j,t in enumerate(old['teacher_forced']):
   assert int(out['z'][lo+j].argmax())==t['argmax'];largest=max(largest,abs(float(out['lp'][lo+j,t['y']])-t['lp']))
 assert largest<1e-11
 return dict(status='PASS',records=len(out['rows']),max_log_probability_discrepancy=largest)
def countermodels(world,selected,up_to):
 """No arithmetic canonicalization enters a candidate. Exact-key table is an E1 countermodel."""
 from collections import Counter
 d=s.inputs()[str(world)];seen=d['OLD']*8+selected[:up_to];fav=Counter();pos=[Counter() for _ in range(4)];raw={}
 for row in seen:
  a=bytes.fromhex(row['answer']);raw[row['prompt']]=a;fav[a]+=1
  for k,y in enumerate(list(a)+[EOS]):pos[k][y]+=1
 favorite=min(fav,key=lambda a:(-fav[a],a));seq=[]
 for k in range(4):
  y=min(pos[k],key=lambda a:(-pos[k][a],a)) if pos[k] else EOS
  if y==EOS:break
  seq.append(y)
 panel=d['panel']
 return dict(presentations=len(seen),unique_keys=len(raw),raw_key_correct=sum(r['prompt'] in raw and raw[r['prompt']].hex()==r['answer'] for r in panel),favorite_correct=sum(r['answer']==favorite.hex() for r in panel),position_correct=sum(r['answer']==bytes(seq).hex() for r in panel),scope='taught OLD/selected NEW only; no held labels')
def continuation(front,world):
 key=f'{front}_{world}';d=s.inputs()[str(world)];panel=d['panel'];selected=d['selected_NEW'];model=load_state(front,world,'OLD_W');before=s.digest(model)
 s.ACTIVE_MODEL=model;s.ACTIVE_CASE=key
 baseline=evaluate(model,panel,key+'_initial');assert s.digest(model)==before
 parity=qualify_parity(baseline,front,world,'OLD_W');native_n0=model.n;native_t0=model.time
 initial_old=np.flatnonzero(baseline['correct'][:64]).tolist();events=[];prior=baseline
 for index,row in enumerate(selected):
  s.budget();idle=model.clone();observed=model.clone();s.record(row,'native_updating',True);t=time.process_time();sites=model.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),True);s.COUNTS['native_training_distributions']+=len(sites);s.TIMES['native_actual_training_cpu']+=time.process_time()-t
  s.record(row,'native_inhibited',False);t=time.process_time();sites0=observed.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),False);s.COUNTS['native_training_distributions']+=len(sites0);s.TIMES['native_inhibited_training_cpu']+=time.process_time()-t
  elapsed=model.time-idle.time;n.fastmem.FastMemory.rest(idle,elapsed)
  assert abs(model.time-observed.time)<1e-9 and abs(model.time-idle.time)<1e-9
  assert model.n==observed.n and idle.n==model.n-len(sites)
  assert np.array_equal(model.q,np.zeros(257))
  ia=evaluate(idle,panel,f'{key}_e{index:02d}_idle');ob=evaluate(observed,panel,f'{key}_e{index:02d}_observed');ac=evaluate(model,panel,f'{key}_e{index:02d}_actual')
  I=ia['margin']-prior['margin'];O=ob['margin']-ia['margin'];W=ac['margin']-ob['margin'];residual=I+O+W-(ac['margin']-prior['margin'])
  same=[i for i,r in enumerate(panel) if r['id']==row['id']];same_idx=np.concatenate([np.arange(ac['offsets'][i],ac['offsets'][i+1]) for i in same]);other=np.ones(len(W),bool);other[same_idx]=False
  event=dict(index=index,id=row['id'],elapsed_seconds=elapsed,n=model.n,sites=sites,matched_sites=sites0,correct=int(ac['correct'].sum()),old_correct=int(ac['correct'][:64].sum()),new_correct=int(ac['correct'][64:].sum()),initially_correct_old_retained=sum(bool(ac['correct'][i]) for i in initial_old),complete_lost=np.flatnonzero(prior['correct']&~ac['correct']).tolist(),complete_gained=np.flatnonzero(~prior['correct']&ac['correct']).tolist(),mean_I=float(I.mean()),mean_O=float(O.mean()),mean_W=float(W.mean()),same_record_W=float(W[same_idx].mean()),other_record_W=float(W[other].mean()),own_margin_negative_to_positive=int(np.sum((ob['margin'][same_idx]<=0)&(ac['margin'][same_idx]>0))),write_other_positive_to_negative=int(np.sum((ob['margin'][other]>0)&(ac['margin'][other]<=0))),idle_positive_to_negative=int(np.sum((prior['margin']>0)&(ia['margin']<=0))),observation_positive_to_negative=int(np.sum((ia['margin']>0)&(ob['margin']<=0))),telescoping_residual_max=float(np.max(abs(residual))),countermodels=countermodels(world,selected,index+1))
  events.append(event);prior=ac;s.log('NATIVE_RECORD_COMMIT',case=key,record_index=index,complete_correct=event['correct']);s.save('ACTIVE.json',dict(stage='M',case=key,next_record=index+1,**s.snapshot_counts()))
  del idle,observed,ia,ob
 end=s.checkpoint(model,key+'_continuation.pkl');model.rest(128);delay=evaluate(model,panel,key+'_final_idle');idle_end=s.checkpoint(model,key+'_post_idle.pkl');del model
 archived=load_state(front,world,'W');final=evaluate(archived,panel,key+'_archived_final');parity_final=qualify_parity(final,front,world,'W');del archived
 result=dict(case=key,front=front,world=world,evidence='E1 descriptive reused-world continuation',initial_n=native_n0,initial_time=native_t0,initial_old_correct=initial_old,retention_measurable=bool(initial_old),initial_complete=int(baseline['correct'].sum()),final_complete=int(prior['correct'].sum()),final_idle_complete=int(delay['correct'].sum()),archived_final_complete=int(final['correct'].sum()),parity=dict(OLD_W=parity,final_W=parity_final),events=events,end=end,post_idle=idle_end,initial_countermodels=countermodels(world,selected,0))
 s.save(f'cases/{key}/MARGIN_RESULT.json',result);s.ACTIVE_MODEL=None;s.ACTIVE_CASE=None;return result
