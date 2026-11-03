"""Registered one-pair signal screen, no optimizer or scientific updates."""
import bootstrap as B
from learner import base_gradients,paired,hcredit,proposals,hnorm
from cycle2 import continuing_ledger,save
from ledger import H
import copy,time,json
T=B.T;np=B.np

def flat(g):return T.cat([g['H'].double().flatten(),g['Hb'].double().flatten()])
def cosine(a,b):return float(T.dot(a,b)/(a.norm()*b.norm()+1e-30))
def main():
 ledger=continuing_ledger();start=time.process_time();worlds={};proto=json.loads((H/'INSTRUMENT_PROVENANCE.json').read_text())
 for w in ('M1','M2'):
  z=np.load(H/'local'/f'{w}_frozen_sites.npz');x=T.from_numpy(z['x']);y=T.from_numpy(z['y']);ids=z['record_ids'];worlds[w]={}
  for stage in ('birth','A2'):
   net=B.P.Learner(w);net.restore(T.load(H/'local'/f'{w}_{stage}.pt',weights_only=True));before=B.statehash(net);rng=net.rng.get_state().clone();rows=[]
   for b in range(4):
    xb=x[b*256:(b+1)*256];yb=y[b*256:(b+1)*256];unit=len(set(ids[b*256:(b+1)*256].tolist()));ledger.add(f'{w} {stage} batch{b}: one base,4 auxiliary,1 H-movement loss',n=256,paths=6,base_records=unit,category='frozen_assay',loss_records=unit*6)
    grad,cache=base_gradients(net,xb,yb);draws=[];true=None;base=None;pcfirst=None;sigmafirst=None
    for draw in range(2):
     ns=f'V183|qualification|{w}|{stage}|{b}|{draw}|noise';ss=f'V183|qualification|{w}|{stage}|{b}|{draw}|shuffle';ng=T.Generator().manual_seed(B.P.seed(ns));sg=T.Generator().manual_seed(B.P.seed(ss));noise=T.randn((256,64),dtype=T.float64,generator=ng);q,sigma,losses=paired(net,cache,yb,noise);real=hcredit(cache[0],q,noise);shift=int(T.randint(1,256,(1,),generator=sg));shuf=hcredit(cache[0],T.roll(q,shift),noise)
     if true is None:
      a=cache[1].detach().double().requires_grad_();base=losses(a);delta=T.autograd.grad(base.mean(),a)[0];true=flat({'H':T.einsum('nj,njr->jr',delta,cache[0].double()),'Hb':delta.sum(0)});sigmafirst=sigma
     else:assert sigma==sigmafirst
     _,_,pc=proposals(net,{**grad,**real});_,_,sc=proposals(net,{**grad,**shuf});cn=hnorm(pc);sn=hnorm(sc)
     if sn<1e-20 and cn>=1e-20:raise RuntimeError('SHAM_STRENGTH_INVALID')
     scale=float(cn/sn) if sn>=1e-20 else 0.
     out={'draw':draw,'noise_namespace':ns,'shuffle_namespace':ss,'sigma':sigma,'raw_one_pair_cosine':cosine(flat(real),true),'raw_shuffled_cosine':cosine(flat(shuf),true),'fixed_FH_cosine':cosine(flat(grad),true),'HPC_proposed_descent_cosine':cosine(-flat(pc),true),'SHUFFLED_proposed_descent_cosine':cosine(-flat(sc),true),'HPC_H_step_norm':float(cn),'SHUFFLED_H_step_norm_before_match':float(sn),'sham_scale':scale,'sham_matched_norm_relative_error':abs(float(sn)*scale-float(cn))/max(float(cn),1e-30),'advisory_NORMAL_moment_history':stage=='A2'}
     draws.append(out)
     if draw==0:pcfirst=pc
    # Sixth and final loss evaluation: float32 proposed H/Hb stored movement.
    changed_a=(cache[0]*(net.p['H']+pcfirst['H'])).sum(2)+(net.p['Hb']+pcfirst['Hb']);newloss=losses(changed_a.double());rows.append({'batch':b,'source_record_batch_units':unit,'draws':draws,'H_only_counterfactual_loss_change_nats':float(newloss.mean()-base.detach().mean()),'counterfactual_applied_to_live_state':False})
   net.rng.set_state(rng);assert B.statehash(net)==before
   real=[d['raw_one_pair_cosine'] for r in rows for d in r['draws']];shuf=[d['raw_shuffled_cosine'] for r in rows for d in r['draws']];mean=float(np.mean(real));sm=float(np.mean(shuf));checks={'mean_cosine_ge_0_05':mean>=.05,'mean_advantage_ge_0_05':mean-sm>=.05,'positive_draws_ge_6_of_8':sum(v>0 for v in real)>=6}
   worlds[w][stage]={'batches':rows,'raw_one_pair_cosines':real,'raw_shuffled_cosines':shuf,'mean_cosine':mean,'mean_shuffled_cosine':sm,'mean_advantage':mean-sm,'positive_draws':sum(v>0 for v in real),'checks':checks,'stage_pass':all(checks.values()),'gating':w=='M1','mutable_state_unchanged':True}
   save('FROZEN_SIGNAL_RESULTS.json',{'worlds':worlds,'assay_state_origin':'V183-R1 authenticated V181 parent instruments; not V182 raw replication','source_provenance':'INSTRUMENT_PROVENANCE.json','new_optimizer_updates':0,'all_two_draws_reported_no_pair_averaged_update':True})
 ok=all(worlds['M1'][stage]['stage_pass'] for stage in ('birth','A2'))
 save('CYCLE3.json',{'status':'CREDIT_SIGNAL_QUALIFIED' if ok else 'CREDIT_SIGNAL_NOT_QUALIFIED','instrument_revision':'V183-R1','M1_stage_checks':{stage:worlds['M1'][stage]['checks'] for stage in ('birth','A2')},'M2_descriptive_only':True,'exact_D1_V182_archive_reproduced':False,'source_fixed_birth_reproduced':True,'new_optimizer_updates':0,'full_roster_opened':False,'counts':ledger.write(),'CPU_seconds':time.process_time()-start})
 print(json.dumps({'status':'QUALIFIED' if ok else 'CREDIT_SIGNAL_NOT_QUALIFIED','worlds':{w:{st:{k:v for k,v in obj.items() if k in ('mean_cosine','mean_shuffled_cosine','mean_advantage','positive_draws','stage_pass')} for st,obj in sts.items()} for w,sts in worlds.items()},'counts':ledger.write(),'CPU_seconds':time.process_time()-start}),flush=True)
if __name__=='__main__':main()
