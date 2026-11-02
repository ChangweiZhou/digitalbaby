"""Read-only measurement around the unchanged source update."""
import util as U
import learner as M
import plasticity as P
import json,math
T=M.torch;F=M.F
COUNTER=0

def measured_step(net,opt,idx,y,context,filename='UPDATE_TRACE.jsonl',freeze=False):
 global COUNTER
 with T.no_grad():
  ah=F.embedding_bag(idx,net.W,mode='sum',padding_idx=M.V.PAD)+net.b;hh=ah.relu();at=net.top.first(hh)
  act=dict(H_active=float((ah>0).float().mean()),T_active=float((at>0).float().mean()),H_activation_RMS=float(hh.square().mean().sqrt()),T_activation_RMS=float(at.relu().square().mean().sqrt()))
  before={n:p.detach().clone() for n,p in net.named_parameters()}
 discarded=P.step(net,opt,idx,y,freeze)
 groups={}
 with T.no_grad():
  for n,p in net.named_parameters():
   group='H' if n in ('W','b') else ('T' if n.startswith('top.first') else 'O');g=groups.setdefault(group,dict(weight_sq=0.,gradient_sq=0.,update_sq=0.,moment_m_sq=0.,moment_v_sq=0.,elements=0));st=opt.state[p]
   g['weight_sq']+=float(before[n].square().sum());g['gradient_sq']+=float(p.grad.square().sum()) if p.grad is not None else 0.;g['update_sq']+=float((p-before[n]).square().sum());g['moment_m_sq']+=float(st['exp_avg'].square().sum());g['moment_v_sq']+=float(st['exp_avg_sq'].square().sum());g['elements']+=p.numel()
 for g in groups.values():
  for field in ('weight','gradient','update'):g[field+'_norm']=math.sqrt(g[field+'_sq'])
  g['update_weight_ratio']=g['update_norm']/max(g['weight_norm'],1e-30);g['moment_m_RMS']=math.sqrt(g['moment_m_sq']/g['elements']);g['moment_v_RMS']=math.sqrt(g['moment_v_sq']/g['elements'])
 COUNTER+=1;row=dict(step=COUNTER,context=context,targets=len(y),freeze_hidden=freeze,discarded_hidden_gradient_sq=discarded,activations=act,layers=groups)
 with (U.H/filename).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
