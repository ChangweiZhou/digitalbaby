from v4_common import *
from native_engine import Engine
from geometry import Geometry,SegmentState,Boundary,BITS
from core_v4 import Learner,target_bits,token_from_values
from native import event_batch
import ast,resource,pickle

def run():
 start=time.process_time();birth=time.monotonic();save(Path('cycles/cycle1/SOURCE_BEFORE.json'),lock())
 e=Engine(5177,202610204001,False);ref=Engine(5177,202610204001,False);rng=np.random.default_rng(401);max_state=max_value=0.
 for step in range(12):
  x=np.zeros(5177);x[rng.choice(5177,103,replace=False)]=1
  c=rng.uniform(-.2,.2,9) if step%3 else None;dt=2.142857142857143
  a=e.values(x,dt);b=ref.values(x,dt);max_value=max(max_value,float(abs(a-b).max()));assert np.max(abs(a-b))<1e-10
  e.event(x,c,dt,step);m=ref.bank
  event_batch(m.fast,m.slow,m.adapt,x,dt,np.zeros(9) if c is None else c,c is not None,m.Q,m.T,m.F,m.MM,m.k0,m.fw0,m.fwd,m.ta,m.tg,m.tf,m.ts,m.fraction,m.alpha_scale,m.genes,m.sf,m.ss)
  ref.time+=dt
  for key in ('fast','slow','adapt'):
   err=float(abs(getattr(e.bank,key)-getattr(m,key)).max());max_state=max(max_state,err);assert err<1e-9
 e.rest(86400);ref.rest(86400)
 assert max(abs(e.bank.slow-ref.bank.slow).ravel())<1e-9
 g=Engine(64,401,True);x=np.ones(64);c=np.ones(9)*.1
 g.event(x,c,record_id=11);q=g.confirm.copy();assert np.count_nonzero(q)>0
 g.event(x,c,record_id=11);assert np.array_equal(q,g.confirm)
 g.event(x,c,record_id=12);idx=np.unravel_index(np.argmax(abs(g.confirm)),g.confirm.shape);assert abs(g.confirm[idx])>1.9
 unchanged=g.confirm.copy();g.event(x,record_id=13);assert np.array_equal(unchanged,g.confirm)
 s=SegmentState();s.feed(0,False);s.feed(255,False);s.feed(137,True)
 assert np.array_equal(s.closed[0,0,:8],BITS[255]) and np.array_equal(s.closed[0,1,:8],BITS[0])
 for b in range(256):s.feed(b,False)
 assert len(s.open)==16 and s.overflow==240
 b=Boundary(9);assert np.array_equal(b.w,np.zeros(9));assert not any(b.decide(i,False) for i in range(256))
 for i in range(256):assert token_from_values(target_bits(i))==i
 assert token_from_values(target_bits(256))==256
 m=Learner(256,401);traces=[]
 for p,a in ((b'\x00\xff',b'\x80'),(b'',b''),(b'\x89\x00',b'\xff\x00')):traces.append(m.train(p,a))
 before=m.digest();o=m.generate(b'\x00\xff');assert m.digest()==before
 clone=pickle.loads(pickle.dumps(m));m.train(b'\x00',b'\x80');clone.train(b'\x00',b'\x80');assert m.digest()==clone.digest()
 n=Learner(256,401,learning=False);w=n.boundary.w.copy();q=n.engine.confirm.copy();n.train(b'\x00',b'\xff');assert np.array_equal(w,n.boundary.w) and np.array_equal(q,n.engine.confirm)
 assert np.max(abs(n.engine.bank.fast))==0 and np.max(abs(n.engine.bank.slow))==0
 assert m.generate(b'',0)['truncated'] or m.generate(b'',0)['ended']
 for f in ('geometry.py','core_v4.py','native_engine.py'):
  tree=ast.parse((HERE/f).read_text());imports=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
   if isinstance(node,ast.ImportFrom):imports.append(node.module)
  assert not set(imports)&{'data_v4','torch','jax','tensorflow','sklearn'}
 result=dict(accepted=True,scope='E0 native/kernel and generic fixtures, not arithmetic learning',native_events=12,full_record_applications=6,partial_native_confirmation_events=4,max_native_state_error=max_state,max_native_value_error=max_value,all_256_payloads=True,explicit_native_query_reference=True,checkpoint_resume_exact=True,read_only_generation=True,no_learning_zero_writes=True,soft_gate_unconfirmed_slow_fraction=.5/(1+np.exp(4)),q_distinct_record_fixture=float(abs(g.confirm[idx])),CPU_seconds=time.process_time()-start,wall_seconds=time.monotonic()-birth,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_after=lock(),technical_only=True,scientific_records=0)
 save(Path('cycles/cycle1/AUDIT.json'),result);print(json.dumps({k:v for k,v in result.items() if k!='source_after'}),flush=True)
if __name__=='__main__':run()
