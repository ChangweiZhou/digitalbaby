"""Freeze the complete campaign before any source/scientific fitting."""
import json,time,hashlib,os
from pathlib import Path
import runtime as C
def main():
 assert json.loads((C.R/'SOURCE_DRIVER_QUALIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((C.R/'NATIVE_DRIVER_QUALIFICATION.json').read_text())['status']=='PASS'
 assert not (C.R/'COMMIT_INDEX.json').exists()
 original=json.loads((C.DEV/'DESIGN_INPUT_LOCK.json').read_text())
 for p,d in original.items():assert C.sha(C.ROOT/p)==d,p
 native=json.loads((C.DEV/'IMPLEMENTATION_SOURCE_LOCK.json').read_text())['files']
 for p,d in native.items():
  if p.startswith('minifly_a_v3/'):assert C.sha(C.ROOT/p)==d,p
 audit=json.loads((C.DEV/'DATA_AUDIT.json').read_text());assert C.sha(C.DEV/'local/DATA_MANIFEST.json')==audit['manifest_sha256']
 for name,meta in audit['corpora'].items():assert C.sha(C.DEV/'assets/corpora'/name)==meta['sha256']
 exports=json.loads((C.DEV/'EXPORT.json').read_text())
 for name,meta in exports['splits'].items():assert C.sha(C.DEV/f'local/{name}.npz')==meta['sha256']
 # Same explicit PCG64 stream across all arms; passes are registered1-based.
 import numpy as np
 def rng(s):return np.random.default_rng(int.from_bytes(hashlib.sha256(s.encode()).digest()[:16],'big'))
 orders={pair:{str(k):rng(f'V176|{pair}|B-pass|{k}').permutation(128).tolist() for k in range(1,9)} for pair in ('P1','P2')};C.save('B_ORDERS.json',orders)
 C.save('COUNTERMODEL_LOCK.json',dict(classes=257,frequency_smoothing=1,output_site_separated=True,tie_break='lowest byte',actual_fit_records_per_pair=128,exact_key_fallback='frequency',counting_reference_status='COUNTING_REFERENCE_UNAVAILABLE',reason='Pinned ClaudeV169 counting code/config not authenticated locally; no substitute allowed.'))
 C.COUNTS['technical_sensory_passes']+=4;C.save('COUNTERS.json',dict(C.COUNTS))
 settings=dict(status='PASS',source_arms=6,source_passes=3,native_lives_conditional=12,native_records_conditional=12288,native_own_TF_pairs_conditional=10752,fixed_probe_fits=12,source_prediction_head_discarded_before_native=True,scientific_sources_frozen=True,local_candidate_updates_only=True,optimization='CPU float64 exact passive batching, scoped immutable checkpoints, shared evaluation prefix',no_automatic_retry=True,gate='TRUE lower probe NLL in both spaces and both pairs',H_float32_formula_tolerance=C.AUTH['H_float32_formula_tolerance'],counting_reference_optional_unavailable=True,qualification_CPU=sum(json.loads(p.read_text())['CPU'] for p in (C.R/'ops').glob('qual_*_DONE.json')))
 C.save('QUALIFICATION.json',settings)
 paths=set(C.R.glob('*.py'))|set((C.ROOT/'execution_optimization_20261101').glob('*.py'))
 paths|={C.DEV/'feature_stage.py',C.DEV/'native_stage.py',C.ROOT/'minifly_writer_interference_dev_v1_20261027/writerlib.py',C.ROOT/'minifly_cls_dev_20261017/fastmem.py',C.R/'AUTHORIZATION.json',C.R/'B_ORDERS.json',C.R/'COUNTERMODEL_LOCK.json',C.R/'BIRTH_LOCKS.json',C.R/'QUALIFICATION.json',C.R/'SOURCE_DRIVER_QUALIFICATION.json',C.R/'NATIVE_DRIVER_QUALIFICATION.json',C.DEV/'EXPORT.json',C.DEV/'DATA_AUDIT.json',C.DEV/'local/DATA_MANIFEST.json',C.DEV/'local/P1_train.npz',C.DEV/'local/P1_calib.npz',C.DEV/'local/P2_train.npz',C.DEV/'local/P2_calib.npz'}
 locks={str(p.relative_to(C.ROOT)):C.sha(p) for p in sorted(paths)};locks.update({p:d for p,d in native.items() if p.startswith('minifly_a_v3/')});locks.update(original)
 C.save('SOURCE_LOCK.json',dict(scope='Prospective complete scientific driver/data/source lock after engineering qualification; no main fit yet',files=locks))
 C.save('STATUS.json',dict(status='QUALIFIED_AWAITING_DETACHED_LAUNCH',scientific_training_started=False,epoch=time.time()))
 print(json.dumps(settings))
if __name__=='__main__':main()
