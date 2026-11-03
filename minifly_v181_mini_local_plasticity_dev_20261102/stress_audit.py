"""Full-batch synthetic arithmetic check; no trajectory update and no real labels."""
import sys,time,json,resource
from pathlib import Path
H=Path(__file__).resolve().parent
import learner as L
from sensor import Sensor
T=L.T;np=L.np;n=L.Learner();s=Sensor();g=T.Generator().manual_seed(181)
x=(T.rand((256,5177),generator=g)<184/5177).float();y=T.randint(257,(256,),generator=g);ids=[str(i//32) for i in range(256)]
start=time.process_time();grad,gains,info=n.gradients(x,y,ids);cpu=time.process_time()-start
assert n.clock==0 and all(T.count_nonzero(v)==0 for v in n.m.values());assert info['reliable']
assert all(not v.requires_grad for v in n.p.values())
# Bias finite difference for explicitly broadcast local linear objectives,
# using frozen error at the synthetic forward, in independent float64 numpy.
logits,(xx,ah,h,at,drop,z)=n.forward(x[:7]);e=logits.double().softmax(1);e[T.arange(7),y[:7]]-=1;e/=7;errors={}
for name,a,signal in [('H',ah,e@n.fh.double()),('T',at,(e@n.ft.double())*drop.double())]:
 a=a.numpy().astype(float);sig=signal.numpy();analytic=((a>0)*sig).sum(0);err=[]
 for j in range(16):
  eps=1e-5;off=np.zeros(64);off[j]=eps
  numeric=(np.sum(np.maximum(0,a+off)*sig)-np.sum(np.maximum(0,a-off)*sig))/(2*eps);err.append(abs(numeric-analytic[j]))
 errors[name]=max(err);assert max(err)<1e-6
# Literal float32 realized moves, after preconditioning and decay, same state.
before=T.randn((64,257),generator=g);delta=T.randn((64,257),generator=g)*.001;gain=gains['H'];en=delta.square().sum(1);scale=((en*gain.square()).sum()/en.sum()).sqrt();a=(before+delta*gain[:,None])-before;b=(before+delta*scale)-before;rel=float(abs(a.square().sum()-b.square().sum())/a.square().sum());assert rel<1e-5
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024);assert peak<=1073741824
(H/'STRESS_AUDIT.json').write_text(json.dumps({'status':'PASS','synthetic_targets':256,'real_training_records':0,'optimizer_updates':0,'full_batch_256_CPU_seconds':cpu,'peak_RSS_bytes':peak,'gradient_recombination':info,'bias_finite_difference_max_error':errors,'same_state_realized_float32_energy_relative_error':rel,'no_autograd_or_BPTT':True},indent=2)+'\n')
r=json.loads((H/'RESOURCE_ADMISSION.json').read_text());r['full_batch_stress_peak_RSS_bytes']=peak;r['full_batch_gradient_CPU_seconds']=cpu;r['full_batch_gradient_extrapolation_seconds']=cpu/256*r['scheduled_science_target_applications'];r['synthetic_full_batch_passed']=True
(H/'RESOURCE_ADMISSION.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'PASS','cpu':cpu,'rss':peak,'realized_sham_relative_error':rel,'bias_errors':errors}))
