"""Prospective source/runtime/data/parameter and optimizer freeze before A."""
import json,hashlib,time
from pathlib import Path
import common as C
from model import Model
torch=C.torch;np=C.np
def main():
 assert json.loads((C.H/'NUMERICAL_QUALIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((C.H/'RESOURCE_ADMISSION.json').read_text())['status']=='PASS'
 # A float64 fixture also qualifies the selected plain-loop gated package.
 torch.set_default_dtype(torch.float64)
 try:
  m=Model('C').double();idx=torch.tensor([[[0],[1],[2],[3],[4],[5],[6],[7],[1],[4],[2],[6]]]);actual=m.run(idx);h=torch.zeros(1,256);expected=[]
  for v in idx[0,:,0].tolist():h=.5*h+.5*torch.tanh(m.Wrk[v]+h@m.Wrr.T+m.b);expected.append(h)
  assert torch.allclose(actual,torch.stack(expected,1),atol=1e-9,rtol=0)
  loss=m.out(actual)[:,-1].sum();loss.backward();analytic=float(m.c.grad[0]);eps=1e-6;vals=[]
  with torch.no_grad():
   for value in (eps,-eps):m.c[0]=value;vals.append(float(m.out(m.run(idx))[:,-1].sum()))
   m.c[0]=0
  finite=(vals[0]-vals[1])/(2*eps);assert abs(finite-analytic)<=1e-8+1e-6*abs(analytic)
 finally:torch.set_default_dtype(torch.float32)
 tr=C.splits()['train'];measured,info=C.L.gain(tr,1)
 masks={}
 for guard in (False,True):
  for stage in ('A','B','C'):
   m=Model(stage,guard);masks[('GUARD_' if guard else '')+stage]=dict(spectral_norm=float(torch.linalg.matrix_norm(m.Wrr,ord=2).detach()),nonzero_recurrent_edges=int((m.Wrr.detach()!=0).sum()),input_sha256=hashlib.sha256(m.Wrk.detach().numpy().tobytes()).hexdigest(),output_sha256=hashlib.sha256(m.out.weight.detach().numpy().tobytes()).hexdigest())
 C.save('MASK_AND_GAIN_AUDIT.json',dict(historical_fixed_gain=C.GAIN,source_recomputed_gain=measured,recomputed_train_only_diagnostics=info,no_retuning=True,packages=masks,selected_gated_plain_loop_forward_gradient_parity=True))
 C.save('EXECUTION_SPEC.json',dict(stages=['A','B','C','conditional GUARD_A+GUARD_C'],first_failing_B_or_C_extension=10,epochs_matched=10,source_gain=C.GAIN,optimizer='Original PyTorch AdamW .001,decay.0001,global clip5,batch16,orderseed13',no_added_warmup='Instrumental literal historical reference; no new adopted/local candidate.',probe=dict(type='multinomial softmax',regularization=.01,objective='meanCE+.005*L2 excludingbias',training_only_normalization=True,optimizer='L-BFGS-B max200 ftol1e-10 gtol1e-7',no_grid=True),historical_test_10_if_calibration_pass_else_final_extension_20_only=True,compiled_loop_selected=False,technical_records_consumed=64,extra_acceleration_gradient_record_exposures=32,technical_examples_not_main_updates=True))
 locks={str(p.relative_to(C.ROOT)):C.sha(p) for p in [*C.H.glob('*.py'),C.H/'AUTHORIZATION.json',C.H/'EXECUTION_SPEC.json',C.H/'DATA_MANIFEST.json',C.H/'PARAMETER_BIRTHS.json',C.H/'MASK_AND_GAIN_AUDIT.json']}
 locks.update({str(p.relative_to(C.ROOT)):C.sha(p) for p in (C.H/'local/source').rglob('*') if p.is_file()})
 C.save('SOURCE_LOCK.json',dict(epoch=time.time(),scope='Prospective before any A/B/C or guard training; includes exact imported source closure',files=locks))
 C.save('STATUS.json',dict(status='QUALIFIED_SOURCE_LOCKED_READY_FOR_A',source_anchor='f4146856c3241d94f7fa948e0b7d30fa0b869bc1',reference_not_run=True))
 print('READY_FOR_A',C.cpu())
if __name__=='__main__':main()
