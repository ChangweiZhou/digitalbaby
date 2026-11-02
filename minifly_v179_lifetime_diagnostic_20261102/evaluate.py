"""Frozen full-vocabulary scoring and own-generation; no learner feedback."""
import util as U
import learner as M
import pickle,sys,time
T=M.torch;np=M.np
NATIVE=None

def native():
 global NATIVE
 if NATIVE is None:
  sys.path.append('/Users/pencilbard/Downloads/second_try_copy/V88/.venv-rce-v87-followup/lib/python3.11/site-packages')
  sys.path.insert(0,str(U.ROOT/'minifly_v176_feature_deployment_dev_20261101'));import native_stage as N
  NATIVE=(N,N.Model())
 return NATIVE

def panel(net,z,name,Aseen,allseen,old=None):
 net.eval();cols={k:[] for k in ['target_log2p','old_log2p','entropy','EOS_mass','A_absent_mass','all_absent_mass','pred']};psum=np.zeros(257);max_norm=0.;yy=z['y']
 with T.no_grad():
  for i in range(0,len(yy),512):
   lp=T.log_softmax(net(z['idx'][i:i+512]).double(),1).numpy()/np.log(2);prob=np.exp2(lp);y=yy[i:i+len(lp)];at=np.arange(len(y));cols['target_log2p'].append(lp[at,y]);cols['old_log2p'].append(lp[at,(old[i:i+len(y)] if old is not None else y)]);cols['entropy'].append(-(prob*lp).sum(1));cols['EOS_mass'].append(prob[:,256]);cols['A_absent_mass'].append(prob[:,~Aseen].sum(1));cols['all_absent_mass'].append(prob[:,~allseen].sum(1));cols['pred'].append(lp.argmax(1));psum+=prob.sum(0);max_norm=max(max_norm,float(abs(prob.sum(1)-1).max()))
 v={k:np.concatenate(a) for k,a in cols.items()};bits=-v['target_log2p'];ends=np.cumsum(np.r_[0,z['n']]);line=[float(bits[ends[i]:ends[i+1]].sum()) for i in range(len(z['n']))];eos=yy==256;n=len(bits);qs=[float(x.mean()) for x in np.array_split(bits,4)]
 path=U.H/'local/panels'/f'{name}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,y=yy,n=z['n'],**{k:x.astype(np.int16 if k=='pred' else np.float32) for k,x in v.items()})
 return dict(bits=float(bits.mean()),accuracy=float(np.mean(v['pred']==yy)),records=len(z['n']),targets=n,record_bit_sums=line,record_targets=z['n'].tolist(),payload_bits=float(bits[~eos].mean()),eos_bits=float(bits[eos].mean()),eos_top1=float(np.mean(v['pred'][eos]==256)),entropy_bits=float(v['entropy'].mean()),EOS_mass_mean=float(v['EOS_mass'].mean()),EOS_mass_on_payload=float(v['EOS_mass'][~eos].mean()),EOS_mass_on_EOS=float(v['EOS_mass'][eos].mean()),A_absent_mass=float(v['A_absent_mass'].mean()),all_absent_mass=float(v['all_absent_mass'].mean()),chronological_target_quartile_bits=qs,mean_probability=(psum/n).tolist(),predicted_class_counts=np.bincount(v['pred'],minlength=257).tolist(),target_class_counts=np.bincount(yy,minlength=257).tolist(),max_probability_normalization_error=max_norm,panel=str(path.relative_to(U.H)),sha256=U.sha(path))

def generation(net,recs,name):
 N,m=native();fe=[];times=[];expected=[];generated=[[] for _ in recs];active=list(range(len(recs)))
 for r in recs:
  f=m._new_front();t=m.time+2*N.F.DELTA;f.advance(t)
  for b in r[:3]:t+=N.F.DELTA;f.feed(b,t)
  fe.append(f);times.append(t);expected.append([*r[3:],256])
 before=M.state_hash(net);rng=T.get_rng_state().clone();net.eval()
 with T.no_grad():
  for step in range(32):
   if not active:break
   rows=[]
   for j in active:
    f=fe[j].clone();f.advance(times[j]+N.F.DELTA);x=np.flatnonzero(N.F.FastMemory.encode(m,f.read())[0]);row=np.full(185,5178,np.int64);row[:len(x) if len(x) else 1]=x if len(x) else 5177;rows.append(row)
   ys=net(T.from_numpy(np.array(rows))).argmax(1).tolist();next_active=[]
   for j,y in zip(active,ys):
    generated[j].append(y)
    if y!=256:times[j]+=N.F.DELTA;fe[j].feed(y,times[j]);next_active.append(j)
   active=next_active
 assert M.state_hash(net)==before and T.equal(T.get_rng_state(),rng)
 exact=sum(g==e for g,e in zip(generated,expected));stopped=sum(bool(g) and g[-1]==256 for g in generated)
 path=U.H/'local/panels'/f'{name}_generation.json';path.write_text(__import__('json').dumps(dict(generated=generated,expected=expected)))
 return dict(records=len(recs),complete_exact=exact,complete_exact_rate=exact/len(recs),own_EOS=stopped,truncated_without_EOS=len(active),max_generated=32,prompt_observed_bytes=3,output_targets=sum(map(len,generated)),raw=str(path.relative_to(U.H)),sha256=U.sha(path),no_learning_state_change=True)

def checkpoint(net,opt,world,label,data,records,results,save_state=True):
 t=time.process_time();snap=M.snap(net,opt);state=M.state_hash(net,opt,snap['rng']);sub=np.load(U.H/f'local/{world}_subsets.npz');Aseen=np.isin(np.arange(257),np.unique(data['A_train']['y']));allseen=np.isin(np.arange(257),np.unique(np.concatenate([data[k]['y'] for k in ['A_train','B_train','EDITED_train']])));scores={}
 for split in ('A_train','B_train','EDITED_train','NOEDIT_train','A_held','B_held','EDITED_held'):scores[split]=panel(net,data[split],world+'_'+label+'_'+split,Aseen,allseen,sub['old'] if split=='EDITED_held' else None)
 orig=np.load(U.H/scores['A_held']['panel']);edit=np.load(U.H/scores['EDITED_held']['panel']);ch=sub['changed'];stable=sub['stable'];newloss=-edit['target_log2p'];marg=edit['target_log2p']-edit['old_log2p']
 subsets=dict(A_stable_bits=float((-orig['target_log2p'])[stable].mean()),A_stable_positions=int(stable.sum()),changed_new_bits=float(newloss[ch].mean()),changed_margin=float(marg[ch].mean()),changed_accuracy=float(np.mean(edit['pred'][ch]==edit['y'][ch])),changed_positions=int(ch.sum()))
 gen={k:generation(net,records[k],world+'_'+label+'_'+k) for k in ('A_held','B_held')}
 assert M.state_hash(net,opt,T.get_rng_state())==state
 path=U.H/f'local/{world}_{label}.pt'
 parent=U.load('PARENT_RESULTS.json');oldrow=next((r for r in parent['checkpoints'] if r['world']==world and r['checkpoint']==label),None)
 if oldrow is not None and oldrow['state_hash']!=state:raise RuntimeError('INSTRUMENTATION_REPLAY_STATE_MISMATCH:'+world+':'+label)
 if oldrow is not None and oldrow['checkpoint_file']:
  original=U.ROOT/'minifly_v179_continuous_local_life_dev_20261102'/oldrow['checkpoint_file'];assert U.sha(original)==oldrow['checkpoint_sha256'];path.symlink_to(original)
 else:T.save(snap,path)
 entry=dict(world=world,checkpoint=label,state_hash=state,steps=max((float(v['step']) for v in opt.state.values()),default=0),scores=scores,subsets=subsets,generation=gen,checkpoint_file=str(path.relative_to(U.H)),checkpoint_sha256=U.sha(path),CPU=time.process_time()-t)
 results.append(entry);U.save('RESULTS.json',dict(status='RUNNING',checkpoints=results));U.budget();return snap
