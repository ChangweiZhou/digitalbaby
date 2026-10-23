"""Independent saved-state audit. Does not import core or call native events."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np,json,math
from pathlib import Path
P=Path(__file__).resolve().parent;checked=0;arrays=0;birthchecks=0
for p in sorted((P/'checkpoints').glob('*_FINAL.npz')):
 stem=p.stem;seed=int(stem.split('_')[0]);arm=stem[len(str(seed))+1:-len('_FINAL')];d=json.loads((P/f'manifests/{seed}.json').read_text());rng=np.random.default_rng(seed);Wv=rng.normal(0,.05/np.sqrt(348),(88,348));Wg=rng.normal(0,.05/np.sqrt(348),(88,348));Wv[:,-1]=0;Wg[:,-1]=np.log(.1/.9)
 def eventprefix(phase,n):
  total=0
  for e in d[phase+'_order'][:n]:
   r=d[phase][e['index']];total+=len(r['prompt'])+len(r['answer'])+1
  return total
 for q in sorted((P/'checkpoints').glob(f'{seed}_{arm}_*.npz')):
  with np.load(q,allow_pickle=False) as z:
   meta=json.loads(str(z['metadata']));n=meta['controller']['n'];assert 0<n<=4352
   expected=eventprefix('old',min(n,4096))+eventprefix('new',max(0,n-4096));assert meta['engine']['events']==expected
   stage=q.stem[len(f'{seed}_{arm}_'):];rests=2 if stage=='FINAL' else 1 if stage.startswith('NEW') or stage=='REST1' else 0
   assert math.isclose(meta['engine']['time'],expected*30/14+rests*3600,abs_tol=1e-5)
   assert z['fast'].shape==(9,5177,2) and z['slow'].shape==z['adapt'].shape==(9,5177)
   assert z['c_Wv'].shape==z['c_Wg'].shape==(88,348)
   for k in z.files:
    if k=='metadata':continue
    assert np.isfinite(z[k]).all(),(q,k);arrays+=1
   assert abs(z['c_h']).max()<=1 and z['adapt'].min()>=0 and z['adapt'].max()<=1
   if arm=='FROZEN' or (arm=='NO_OLD' and n<=4096):
    assert np.array_equal(z['c_Wv'],Wv) and np.array_equal(z['c_Wg'],Wg);birthchecks+=1
   if arm=='NO_OLD' and n<=4096:
    assert not np.any(z['fast']) and not np.any(z['slow']);assert meta['controller']['b']==meta['controller']['updates']==0
   checked+=1
assert checked==72,checked
out=dict(status='PASS',saved_checkpoints=checked,numeric_arrays_checked=arrays,initial_controller_equality_checks=birthchecks,NO_OLD_saved_native_values_exactly_zero_through_OLD=True,events_clocks_and_rest_counts_match_manifests=True,records_rerun=0,native_events=0)
(P/'results/CHECKPOINT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
