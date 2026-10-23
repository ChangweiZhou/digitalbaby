from rc_common import *
import rc_common as C
from decode_saved import infer,metric,evaluate

def verify():
 for name,h in json.loads((P/'PARENT_CLOSURE.json').read_text()).items():assert sha(R/name)==h,name
 for name,h in json.loads((P/'HEAD_FREEZE.json').read_text()).items():assert sha(P/'heads'/name)==h,name

def qualify():
 verify();e=Engine(5177,202610239701,False);xs,_=encode(e,keys(202610239701)[:1],True)
 for j in range(4):e.begin();e.tick(xs[0,j],residual(e.margins(xs[0,j])[0],j,.01));C.events+=1
 checkpoint(e,'qualification_restore.npz');other=Engine(5177,202610239701,False)
 with np.load(P/'checkpoints/qualification_restore.npz') as z:
  for k in ('fast','slow','adapt'):getattr(other.bank,k)[:]=z[k]
  for k in ('q','q_at','last_at','ring','ring_pos','signs'):getattr(other,k)[:]=z[k]
  other.time=float(z['time'][0]);other.events=int(z['time'][1]);other.write_l1=float(z['time'][2]);other.slow_write_l1=float(z['time'][3]);other.confirmations=int(z['time'][4])
 assert digest(e)==digest(other)
 before=digest(e);D=probe(e,xs);assert digest(e)==before;assert np.array_equal(D,probe(other,xs))
 rng=np.random.default_rng(202610239701)
 for std in (0,1):
  for hs in range(202610239401,202610239404):
   arbitrary=rng.normal(size=(257,9,9,2));z=infer(arbitrary,std,hs);assert np.array_equal(z[:,:,:8]>0,arbitrary.mean(-1)[:,:,:8]>0)
 for token in range(257):assert decode(targets(token))==token
 # Full imported local Python module closure, including dynamic birth modules.
 modules={str(Path(m.__file__).resolve()):sha(Path(m.__file__)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and str(Path(m.__file__).resolve()).startswith(str(R))}
 save('RUNTIME_MODULE_CLOSURE.json',modules)
 # Verify fresh keys against all previously evaluated association keys, before values.
 prior=set()
 for root in ('minifly_recoverability_v1_dev_20261022','minifly_recoverability_v1_r2_dev_20261023'):
  for f in (R/root/'traces').glob('*.json'):
   if f.stem.endswith(('_decoded','_outputs')):continue
   d=json.loads(f.read_text());prior.update(d.get('keys_hex',[]))
 for seed in range(202610239301,202610239305):prior.update(k.hex() for k in keys(seed))
 seen=set()
 for seed in json.loads((P/'PLAN.json').read_text())['pairs']:
  kk={k.hex() for k in keys(seed)};assert not kk&prior and not kk&seen;seen.update(kk)
 save('QUALIFICATION.json',dict(status='PASS_STOP_ONLY_SCOPE',partial_native_fixture_events=4,full_native_records=0,payload_signs_invariant_all_six_heads=True,codec257=True,all_mutable_checkpoint_fields_restored=True,readonly_exact=True,canonical_birth_verified=True,head_hashes_unchanged=True,source_birth_and_imported_module_closure_locked=True,unique_fresh_keys=len(seen),CREDIT_excluded=True,arithmetic_unchanged=True))

def main():
 qualify();start=time.process_time();base=C.events
 for seed in json.loads((P/'PLAN.json').read_text())['pairs']:
  for lid in (0,1):
   for std in (False,True):
    for arm in ('W','N','SH'):
     life(seed,lid,std,arm,False);evaluate(f'{seed}_{lid}_{int(std)}_{arm}')
  verify();check(f'completed pair {seed}')
  if seed==202610239601 and resource_stats()['CPU_seconds']+31*(time.process_time()-start)*1.25>1650:raise RuntimeError('PROFILE_CANNOT_FIT_LOCKED_ROSTER_WITH_AUDIT_RESERVE')
 assert C.records==225792 and C.events-base==1128960
 for name,h in json.loads((P/'RUNTIME_MODULE_CLOSURE.json').read_text()).items():assert sha(Path(name))==h
 save('EXECUTION_RESULT.json',dict(status='COMPLETE',main_records=C.records,main_events=C.events-base,E0_partial_events=base,pairs=32,value_lives=64,head_fits=0,resources=resource_stats()))

if __name__=='__main__':
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);C.awake=awake
 try:check('qualification start',admission=True);main()
 except BaseException as ex:save('FAILURE.json',dict(error=repr(ex),records=C.records,events=C.events,resources=resource_stats()));raise
 finally:awake.terminate();awake.wait();save('RESOURCE_LEDGER.json',resource_stats())
