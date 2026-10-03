"""V77 diagnostic teaching references. Privileged NOFAST never edits live state."""
from common77 import *
class Learner(ParentLearner):
    def __init__(self,name,prestate=None):
        assert name in CFG['arms']
        super().__init__('REF' if name=='REF' else 'ERROR_NORM_SLOW',denominator=DEN,prestate=prestate)
        self.name77=name
    def references(self,codes):
        dx=observed_activity(self.m,codes);baseline=self.reader.predict(dx)
        current=float((self.m.expression(dx)-baseline).mean(1)@[1.,-1.])
        clone=self.m.clone();clone.fast[:,1]=0.
        nofast=float((clone.expression(dx)-baseline).mean(1)@[1.,-1.])
        return dx,current,nofast
    def pending(self,codes,label):
        assert label in (0,1)
        dx,total,nofast=self.references(codes);ref=nofast if self.name77.startswith('NOFAST') else total
        r=2*int(label)-1;target=r*CFG['target_Hz']
        signal=r*max(0.,CFG['target_Hz']-r*ref) if self.name77.endswith('MARGIN') else target-ref
        v,den=geometry(self.m,dx);used=float(np.clip(den,.1*self.den0,10*self.den0));gain=CFG['eta']/(1e-12+used)
        u=gain*signal*v if den>1e-12 else np.zeros_like(v)
        return u,dict(score=ref,signal=float(signal),denominator=den,gain=gain,clamped=int(used!=den),
            total_score=total,nofast_score=nofast,alpha_fast_exclusion_difference=total-nofast,
            adequate_total=float(r*total>=CFG['target_Hz']),adequate_reference=float(r*ref>=CFG['target_Hz']),
            opposing_label_update=float(r*signal<0),pending_L1=float(abs(u).sum()),pending_L2=float(np.linalg.norm(u)))
