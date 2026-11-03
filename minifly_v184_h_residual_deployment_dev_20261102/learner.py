"""V184 literal residual; one physical output head, unchanged source-local teaching."""
import bootstrap as B
T=B.T;P=B.P
class ControlNotQualified(RuntimeError):pass
class Learner(P.Learner):
 def __init__(self,arm='NORMAL',world='M1',alpha=.5):
  super().__init__(world);assert arm in ('NORMAL','HRES','NORM_SHAM');self.arm=arm;self.alpha=alpha;self.last={}
 def forward(self,x,training=False):
  xx=x[:,self.ix];ah=(xx*self.p['H']).sum(2)+self.p['Hb'];h=ah.clamp_min(0);at=h@self.p['T'].T+self.p['Tb'];u=at.clamp_min(0);mask=(T.rand(at.shape,generator=self.rng)>=.1).float()/.9 if training else T.ones_like(at)
  base=mask*u;residual=mask*(u+self.alpha*h);bn=base.norm(dim=1);rn=residual.norm(dim=1);ratio=rn/bn.clamp_min(1e-6);gain=ratio.clamp(0,8);sham=base*gain[:,None]
  # Float64 is only the norm audit, not another learned route or loss path.
  actual=sham.double().norm(dim=1);target=residual.double().norm(dim=1);err=(actual-target).abs();bad=err>1e-7+1e-6*target
  self.last={'sites':len(x),'gain_min':float(gain.min()),'gain_max':float(gain.max()),'clipped_sites':int((ratio>8).sum()),'zero_denominator_sites':int((bn==0).sum()),'harmless_both_zero_sites':int(((bn==0)&(rn==0)).sum()),'material_norm_mismatch_sites':int(bad.sum()),'max_abs_norm_error':float(err.max()),'max_relative_norm_error':float((err/target.clamp_min(1e-30)).max())}
  if self.arm=='NORMAL':z=base
  elif self.arm=='HRES':z=residual
  else:
   if bad.any():raise ControlNotQualified('CONTROL_NOT_QUALIFIED:'+str(self.last))
   z=sham
  logits=z@self.p['O'].T+self.p['Ob'];return logits,(xx,ah,h,at,mask,z)
 def step(self,x,y,ids,arm=None):
  before={k:v.clone() for k,v in self.p.items()};info=super().step(x,y,ids,'NORMAL');info.update(arm=self.arm,readout_norm=dict(self.last),actual_post_Adam_squared_movements={layer:float((self.p[layer]-before[layer]).double().square().sum()+(self.p[layer+'b']-before[layer+'b']).double().square().sum()) for layer in ('H','T','O')});return info

 def state(self):return {**super().state(),'arm':self.arm,'alpha':self.alpha}
 def restore(self,state):
  if 'arm' in state:assert state['arm']==self.arm and state['alpha']==self.alpha,'Static forward configuration mismatch'
  super().restore(state)
