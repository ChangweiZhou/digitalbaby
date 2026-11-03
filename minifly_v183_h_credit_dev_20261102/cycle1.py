"""Source/data audit and immutable instrument extraction, no training."""
import bootstrap as B
from pathlib import Path
import json,copy,time
from ledger import H,Ledger
T=B.T;np=B.np

def save(n,v):(H/n).write_text(json.dumps(v,indent=2)+'\n')
def main():
 ledger=Ledger();s=B.sensor.Sensor();assert s.digest=='7e0b78ae71cd9f508e9069018ec2343a88ab8bfd9263067506ee6eb3fae4962b';data,meta=B.D.prepare(s)
 original=json.loads((B.ROOT/'minifly_v181_mini_local_plasticity_dev_20261102/campaign_20261102/MANIFEST.json').read_text())
 packed={};t=time.process_time()
 for w,sp in data.items():
  for name,rs in sp.items():
   assert [B.hashlib.sha256(r).hexdigest() for r in rs]==original['worlds'][w][name]['record_sha256']
  rows=[];ys=[];ids=[];pos=[];seen=0
  for i,r in enumerate(sp['A_train']):
   for j,(code,y) in enumerate(zip(s.codes(r),[*r,256])):
    rows.append(code);ys.append(y);ids.append(i);pos.append(j)
    if len(rows)==1024:break
   if len(rows)==1024:break
  assert len(rows)==1024
  # Batch-record units count a record appearing in two adjacent batches twice,
  # matching the conservative source design's 21/32 units.
  units=sum(len(set(ids[i:i+256])) for i in range(0,1024,256));assert units=={'M1':21,'M2':32}[w]
  np.savez_compressed(H/'local'/f'{w}_frozen_sites.npz',x=np.array(rows),y=np.array(ys),record_ids=np.array(ids),positions=np.array(pos));packed[w]={'batch_record_units':units,'A_train_targets':sum(len(r)+1 for r in sp['A_train'])}
 instruments={};paramcount=0
 for w in ('M1','M2'):
  instruments[w]={}
  rpath=B.ROOT/'minifly_v181_mini_local_plasticity_dev_20261102/campaign_20261102/jobs'/f'{w}_NORMAL';receipt=json.loads((rpath/'RECEIPT.json').read_text())
  for stage,file in [('birth','birth.pt'),('A2','A2.pt')]:
   cp=rpath/file;expected=receipt['checkpoints'][stage]['checkpoint_sha256'];assert B.digest(cp)==expected;state=T.load(cp,weights_only=True);net=B.P.Learner(w);net.restore(state)
   if stage=='birth':
    newborn=B.P.Learner(w);assert all(T.equal(newborn.p[k],state['weights'][k]) for k in net.p);assert T.equal(newborn.rng.get_state(),state['dropout_rng'])
   assert sum(v.numel() for v in net.p.values())==37313;paramcount=37313
   savepoint=H/'local'/f'{w}_{stage}.pt';T.save(state,savepoint);instruments[w][stage]={'actual_source':'V181 authorized NORMAL checkpoint, not the unavailable V182 archive','path':str(cp),'source_sha256':expected,'working_copy_sha256':B.digest(savepoint),'mutable_state_hash':B.statehash(net),'clock':net.clock,'V182_published_scalar_parity_max_bits':2.3217854421275774e-6 if w=='M1' else 4.957880683775784e-8,'not_bitwise_V182_reproduction':True}
 tails={role:sum(len(r)+1 for r in data['M1'][role+'_train'])%256 for role in ('A','B','EDIT','NOEDIT')};assert tails=={'A':194,'B':52,'EDIT':7,'NOEDIT':7}
 net=B.P.Learner();count=np.bincount(net.indices.ravel(),minlength=5177);assert (count.min(),count.max())==(3,4)
 ledger.add('sensory byte legality, future-prefix and complete-EOS fixture',n=0,paths=1,base_records=2)
 for r in [bytes(range(128)),bytes(range(128,256))]:assert s.codes(r).shape==(129,5177)
 assert np.array_equal(s.codes(b'abcx')[:4],s.codes(b'abcy')[:4])
 save('INSTRUMENT_PROVENANCE.json',{'source_design_commit':'08809b4c3422c74f06faa6d18abb8c0ec11639e5','revision':'V183-R1 technical instrument provenance only','design_source_anchor':'e3cd3b368342a72e1acac4ae7d4ada8369a86a5d','instruments':instruments,'qualification_is_not_original_D1_state_reproduction':True})
 save('CYCLE1.json',{'status':'PASS_WITH_DECLARED_INSTRUMENT_REVISION','trainable_parameters':paramcount,'sensor_sha256':s.digest,'all_splits_and_original_C_IDs_match':True,'tail_sizes':tails,'frozen_site_panels':packed,'source_record_data_unchanged':True,'environment':{'numpy':np.__version__,'torch':T.__version__,'scipy':B.scipy.__version__,'platform':B.platform.platform()},'sensor_platform_adapter':'libSystem fma on ARM; exact same canonical sensor digest','runtime_recovery_failure':'first isolated pip attempt failed DNS; zero model updates; compatible runtime then downloaded under approved escalation','fixed_arrays_bytes':sum(a.nbytes for a in s.arrays),'source_asset_limits':'V182 private state archive unavailable; immutable V181 parent instruments explicitly substituted','new_optimizer_updates':0,'qualification_data_CPU_seconds':time.process_time()-t})
 print('CYCLE1_PASS',tails,flush=True)
if __name__=='__main__':main()
