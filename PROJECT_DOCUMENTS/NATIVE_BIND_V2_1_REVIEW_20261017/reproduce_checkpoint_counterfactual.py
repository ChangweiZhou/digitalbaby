"""Optional one-world posthoc forward-only reproduction; never trains or changes saved checkpoints."""
from pathlib import Path
import json,sys,copy
R=Path(__file__).resolve().parents[2];I=R/'native_context_binding_dev_v2_1_20261017';sys.path.insert(0,str(I))
import bootstrap
import torch
from bind_kernel import Learner
import bind_evaluator as e
p=I/'results/DEV_A/jobs/202610177101_F111'
before,_=Learner.restore(p/'W_OLD_END.pt');after,_=Learner.restore(p/'W_OLD_REST.pt')
digests=(before.digest(),after.digest());rows=[]
items=[(f'{a}+{b}='.encode(),str(a+b).encode()) for a,b in e.split(202610177101)['old']]
for mode in ['AFTER_REST','RESTORE_GAMMA_FAST','RESTORE_ALPHA_FAST','RESTORE_BOTH_FAST','RESTORE_ADAPT_ONLY','RESTORE_FAST_AND_ADAPT']:
 view=after.clone()
 with torch.no_grad():
  if mode=='RESTORE_GAMMA_FAST':view.state.fast[...,0].copy_(before.state.fast[...,0])
  if mode=='RESTORE_ALPHA_FAST':view.state.fast[...,1].copy_(before.state.fast[...,1])
  if mode in ('RESTORE_BOTH_FAST','RESTORE_FAST_AND_ADAPT'):view.state.fast.copy_(before.state.fast)
  if mode in ('RESTORE_ADAPT_ONLY','RESTORE_FAST_AND_ADAPT'):view.state.adapt.copy_(before.state.adapt)
 score=e.score(view,items)
 rows.append(dict(mode=mode,correct=sum(x['correct'] for x in score['rows']),n=len(items),immediate_EOS=sum(x['status']=='EOS' and x['emitted']=='' for x in score['rows']),rows=score['rows']))
assert digests==(before.digest(),after.digest())
print(json.dumps(dict(scope='One-world posthoc forward-only diagnosis;not efficacy or E3 confirmation',training_calls=0,conditions=rows),indent=2))
