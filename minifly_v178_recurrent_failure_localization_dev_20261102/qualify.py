"""C0 literal source and C1 acceleration parity, before substantive fitting."""
import copy,time,json,platform,hashlib,resource
import common as C
from model import Model,mask
torch=C.torch;np=C.np;L=C.L
def main():
 source=json.loads((C.H/'SOURCE_RECOVERY.json').read_text());assert all(v['matches'] for v in source['files'].values())
 tr=L.synth_split('S-COPY','train');raw=tr['idx'].numpy().reshape(100,120);assert np.array_equal(tr['y'].reshape(100,120)[:,8:],raw[:,:112]) and np.all(tr['y'].reshape(100,120)[:,:8]==-1)
 manifest={};seen=set()
 for guard in (False,True):
  sp=C.splits(guard);ss={};keys=set()
  for k,d in sp.items():
   x=d['idx'].numpy().reshape(-1,120);z={r.tobytes() for r in x};assert len(z)==len(x) and not keys&z;keys|=z;ss[k]=dict(records=len(x),sha256=hashlib.sha256(x.tobytes()).hexdigest(),labels_sha256=hashlib.sha256(d['y'].tobytes()).hexdigest())
  assert not seen&keys;seen|=keys;manifest['guard' if guard else 'historical']=ss
 C.save('DATA_MANIFEST.json',manifest)
 original=L.make_rnn(8,C.GAIN,1);fast=Model('A');fast.load_state_dict({**fast.state_dict(),**original.state_dict()})
 torch.set_default_dtype(torch.float64)
 try:
  original.double();fast.double();idx=torch.tensor([[[0],[3],[1],[7],[2],[6],[4],[5],[0],[7],[6],[1]]]);s=original.run(idx);f=fast.run(idx)
  assert torch.max(torch.abs(s-f))<=1e-9 and torch.max(torch.abs(original.out(s)-fast.out(f)))<=1e-9
  h=np.zeros((1,256));rows=[]
  for x in idx[0,:,0].tolist():h=np.tanh(original.Wrk[x].detach().numpy()+h@original.Wrr.detach().numpy().T+original.b.detach().numpy());rows.append(h.copy())
  independent=np.stack(rows,axis=1);assert np.max(np.abs(independent-s.detach().numpy()))<=1e-9
  target=torch.tensor([0,3,1,7]);loss=torch.nn.functional.cross_entropy(original.out(s[:,8:]).reshape(-1,8),target);loss.backward();lf=torch.nn.functional.cross_entropy(fast.out(f[:,8:]).reshape(-1,8),target);lf.backward()
  for (n,a),(m,b) in zip(original.named_parameters(),[(n,p) for n,p in fast.named_parameters() if n!='mask']):assert n==m and torch.allclose(a.grad,b.grad,rtol=1e-6,atol=1e-8)
  grad=original.Wrr.grad;flat=int(grad.abs().argmax());i,j=divmod(flat,256);base=float(original.Wrr[i,j].detach());eps=1e-6;values=[]
  with torch.no_grad():
   for v in (base+eps,base-eps):original.Wrr[i,j]=v;values.append(float(torch.nn.functional.cross_entropy(original.out(original.run(idx))[:,8:].reshape(-1,8),target)))
   original.Wrr[i,j]=base
  finite=(values[0]-values[1])/(2*eps);assert abs(finite-float(grad[i,j]))<=1e-8+1e-6*abs(float(grad[i,j]))
 finally:torch.set_default_dtype(torch.float32)
 # Time actual full training-record updates on identical copies, no stage learns here.
 timings=[];idx,mask0,pos=L.V.line_batches(tr,list(range(16)));ctr,valid=L.targets(tr,L.Synth(),pos)
 for name,net in [('literal',L.make_rnn(8,C.GAIN,1)),('optimized',Model('A'))]:
  opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);t=C.cpu();loss=torch.nn.functional.cross_entropy(net.out(net.run(idx))[valid],ctr[valid]);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step();C.charge(16);timings.append(dict(backend=name,batch16_CPU=C.cpu()-t,loss=float(loss.detach())))
 graph=mask();reached={0}
 while True:
  new=reached|set(torch.nonzero(graph[list(reached)],as_tuple=True)[1].tolist())
  if new==reached:break
  reached=new
 assert len(reached)==256
 births={};states={}
 for stage in ('A','B','C'):
  m=Model(stage);births[stage]={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()};states[stage]=m
 assert torch.equal(states['A'].Wrk,states['B'].Wrk) and torch.equal(states['A'].out.weight,states['B'].out.weight) and torch.equal(states['B'].Wrr,states['C'].Wrr)
 for stage in ('B','C'):
  m=states[stage];opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);loss=torch.nn.functional.cross_entropy(m.out(m.run(idx))[valid],ctr[valid]);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();m.enforce(opt);C.charge(16)
 # The guard has no engineering score access; only its manifest and fresh birth are frozen.
 C.save('PARAMETER_BIRTHS.json',births)
 C.save('ENVIRONMENT.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=C.scipy.__version__,historical_python='3.11.15',historical_torch='2.14.1+cu130',CPU_only=True,numerical_threads=1,isolated_compatibility_paths=True))
 forecast=1.5*(max(r['batch16_CPU'] for r in timings)*((5064+15)//16)*3+600+C.cpu())
 C.save('RESOURCE_ADMISSION.json',dict(status='PASS' if forecast<=7200 else 'RESOURCE_NOT_ADMITTED',full_route_forecast_CPU=forecast,CPU_cap=7200,margin=1.5,includes_training_eval_probe_sensitivity_and_import_reserve=True,technical_training_records=64,timings=timings))
 C.save('NUMERICAL_QUALIFICATION.json',dict(status='PASS',literal_source_imported_unchanged=True,float64_forward_and_logits_max_abs=1e-9,autograd_and_finite_difference=True,optimized_gradient_parity=True,masked_gradient_moments_preserved=True,strong_connectivity=True,scored_positions=112,historical_text_cache_not_needed=True,technical_records=64))
 print(json.dumps(timings));print('forecast',forecast)
if __name__=='__main__':
 try:main()
 except BaseException as e:C.save('FAILURE.json',dict(error=repr(e),stage='qualification'));raise
