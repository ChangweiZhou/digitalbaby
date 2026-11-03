"""Adversarial bounded engineering tests, not an independent scientific review."""
import bootstrap as B
from learner import Learner
from run import batches
from reduce import decide
from profile_and_review import fake
from cycle2 import continuing_ledger,save
from ledger import H
import copy,json
T=B.T;np=B.np

def main():
 l=continuing_ledger();s=B.sensor.Sensor();data,meta=B.D.prepare(s);sp=data['M1'];totrec=tottar=weightedrec=weightedtar=0;tails={}
 # Chronology, real source IDs and exact terminal counting; no model evaluations.
 l.add('full future roster batch/ID and EOS decision canaries',n=0,paths=1,base_records=1,loss_records=0)
 for arm,factor in [('NORMAL',1),('HPC',3),('SHUFFLED',3)]:
  for role,reps in [('A',2),('B',3),('EDIT',2),('NOEDIT',2)]:
   rs=sp[role+'_train'];codes=[list(range(len(r)+1)) for r in rs];sourceids=meta['M1']['correction']['selected_train_ids'] if role in ('EDIT','NOEDIT') else list(range(len(rs)));origrole='A' if role in ('EDIT','NOEDIT') else role
   for _ in range(reps):
    rows=list(batches(rs,codes,sourceids,origrole));nt=sum(len(v[1]) for v in rows);nr=sum(sum(v[3]) for v in rows);assert nt==sum(len(r)+1 for r in rs) and nr==len(rs);assert all(len(v[1])<=256 for v in rows);tails[role]=len(rows[-1][1]);totrec+=nr;tottar+=nt;weightedrec+=nr*factor;weightedtar+=nt*factor
 assert (totrec,tottar,weightedrec,weightedtar)==(1680,103092,3920,240548);assert tails=={'A':194,'B':52,'EDIT':7,'NOEDIT':7}
 f=fake();f['HPC']['EDIT2']['EDIT_held']['EOS_NLL']=2.;assert not decide(f,3.1)['checks']['EDIT2_EDIT_held_EOS_guard_NORMAL'];assert decide(f,3.1)['verdict']=='HPC_RELATIVE_GAIN_ONLY'
 f=fake();f['SHUFFLED']['A2']['A_held']['bits']=3.01;f['NORMAL']['A2']['A_held']['bits']=3.3;assert decide(f,3.1)['NO_INFORMATION_ASSIGNMENT_ADVANTAGE']
 # Actual moment-bearing state fork with all3 RNGs, at B's registered tail size.
 x=T.from_numpy(s.codes(b'012345'));y=T.tensor([*b'012345',256]);net=Learner();xx=x[T.arange(52)%7];yy=y[T.arange(52)%7];l.add('actual n52 update creates moment-bearing fork state',n=52,paths=3,base_records=1,updates=1);net.step(xx,yy,['fixture']*52,'HPC');state=copy.deepcopy(net.state());bhash=B.statehash(net);fork=[]
 for name in ('EDIT','NOEDIT'):
  n=Learner();n.restore(state);assert B.statehash(n)==bhash;l.add(name+' actual7-site fork from same weights,moments,clocks,RNG',n=7,paths=3,base_records=1,updates=1);n.step(x,y if name=='NOEDIT' else T.roll(y,1),['same-original-source-ID']*7,'HPC');fork.append(n)
 assert fork[0].clock==fork[1].clock
 for k in ('rng','credit_rng','shuffle_rng'):assert T.equal(getattr(fork[0],k).get_state(),getattr(fork[1],k).get_state())
 assert any(not T.equal(fork[0].p[k],fork[1].p[k]) for k in net.p)
 single=Learner();l.add('singleton shuffled H credit is exactly zero before decay',n=1,paths=3,base_records=1,updates=1);single.step(x[:1],y[:1],['singleton'],'SHUFFLED');assert not single.m['H'].any() and not single.m['Hb'].any() and single.m['Ob'].any()
 save('INTEGRATION_REVIEW.json',{'status':'PASS','mode':'EXECUTED_SAME_FAMILY_SELF_AUDIT','future_roster_conserved':{'optimizer_records':totrec,'optimizer_targets':tottar,'record_equivalents':weightedrec,'learning_target_evaluations':weightedtar},'all_registered_tail_sizes':tails,'moment_bearing_EDIT_NOEDIT_start_identical_all_RNGs':True,'different_feedback_changes_weights_but_preserves_RNG_continuation':True,'singleton_shuffled_raw_H_credit_zero':True,'EOS_and_information_assignment_false_passes_rejected':True,'cross_family_review':'PENDING','formal_MRK1':'UNAVAILABLE','main_science':'NOT_RUN','counts':l.write()})
 print('INTEGRATION_PASS',l.write(),flush=True)
if __name__=='__main__':main()
