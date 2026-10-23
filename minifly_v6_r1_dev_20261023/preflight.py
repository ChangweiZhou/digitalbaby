from core import *
import ops,v6_data as data,collections,inspect,ast
ops.begin()
try:
 f=Fly(202610236000);base=f.e.clone();f.begin();x=f.code();m=f.e.margins(x)[0];base=f.e.clone();base.tick(x,None)
 assert np.max(abs(base.margins(x)[0]-m))<1e-12
 f.e.tick(x,residual(m,73,1/256))
 # These are diagnostic references, not a new fit or threshold search.
 out=dict(passive_margin_change=float(np.max(abs(base.margins(x)[0]-m))),write_margin_peak=float(np.max(abs(f.e.margins(x)[0]))),write_state_finite=bool(np.isfinite(f.e.bank.fast).all() and np.isfinite(f.e.bank.slow).all()),native_side_caps=f.e.bank.side_n,mutable_native_bytes=f.e.stats()['mutable_bytes'],canonical_anatomy_hashes={k:hashlib.sha256(getattr(f.e.bank,k).tobytes()).hexdigest() for k in ('Q','T','reader_Q','kc_side')},pn_projection_edges=int(f.e.bank.B.nnz),controller_allocated_array_bytes=sum(getattr(f.c,k).nbytes for k in ('Wv','Wg','Ev','Eg','Gv','Gg')),technical_full_records=0,technical_partial_native_events=2)
 # Static dependency boundary and cap checks. Original templates remain frozen.
 tree=ast.parse((P/'core.py').read_text());imps=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):imps += [a.name for a in n.names]
  if isinstance(n,ast.ImportFrom):imps.append(n.module)
 assert not any(s in imps for s in ['v6_data','torch','tensorflow','jax']);out['core_direct_imports']=imps
 assert inspect.signature(Fly.query).parameters.keys()=={'self','prompt','tape'}
 out['probe_signature_has_no_target_or_length']=True
 for seed in data.WORLDS:
  d=data.build(seed);old=data.records(d,'old');new=data.records(d,'new');count=sum(len(r['prompt'])+len(r['answer'])+1 for r in old+new)
  out[str(seed)]=dict(record_applications_per_arm=len(old+new),observed_native_events_per_arm=count,shuffle_singletons=d['shuffle_singletons'],same_answer_shuffled_pairs=sum(a['answer']==b['answer'] for a,b in zip(d['old'],d['shuffled'])))
 out['status']='PASS';ops.save('cycles/02_INTEGRATION_AUDIT.json',out)
finally:ops.end()
