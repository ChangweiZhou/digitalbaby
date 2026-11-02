"""Source-equivalent dense tanh and explicitly masked architecture packages."""
import copy
import common as C
torch=C.torch;np=C.np;F=torch.nn.functional
@torch.jit.script
def recurrence(u:torch.Tensor,w:torch.Tensor,b:torch.Tensor,r:torch.Tensor):
 out=torch.jit.annotate(list[torch.Tensor],[])
 for t in range(u.shape[1]):
  r=torch.tanh(u[:,t]+r@w.T+b);out.append(r)
 return torch.stack(out,1)
def mask():
 m=torch.zeros(256,256)
 for j in range(256):
  mod,k=divmod(j,8);ix=[mod*8+i for i in range(8)]+[((mod+o)%32)*8+(k+o+d)%8 for o in (-2,-1,1,2) for d in (0,3)];m[j,ix]=1
 assert int(m.sum())==4096 and bool((m.sum(1)==16).all());return m
class Model(torch.nn.Module):
 def __init__(self,stage='A',guard=False):
  super().__init__()
  # Historical birth is literal. Guard uses a separately locked named seed.
  if guard:
   seed=int.from_bytes(C.hashlib.sha256(b'V178|202611027811|birth').digest()[:8],'big')&((1<<63)-1)
   old=C.L.V.SET['lane_C']['init_seed'];C.L.V.SET['lane_C']['init_seed']=seed
   try:base=C.L.make_rnn(8,C.GAIN,1)
   finally:C.L.V.SET['lane_C']['init_seed']=old
  else:base=C.L.make_rnn(8,C.GAIN,1)
  self.Wrk=base.Wrk;self.Wrr=base.Wrr;self.b=base.b;self.out=base.out;self.stage=stage;self.register_buffer('mask',mask())
  if stage!='A':
   with torch.no_grad():
    seed=seed if guard else C.L.V.SET['lane_C']['init_seed'];gen=torch.Generator().manual_seed(seed)
    torch.randn(C.L.V.NIN+1,256,generator=gen);raw=torch.randn(256,256,generator=gen)
    assert torch.equal(base.Wrr,C.GAIN*(raw/torch.linalg.matrix_norm(raw,ord=2)))
    w=raw*self.mask;self.Wrr.copy_(C.GAIN*w/torch.linalg.matrix_norm(w,ord=2))
  if stage=='C':
   self.Vin=torch.nn.Parameter(torch.zeros_like(self.Wrk));self.Srr=torch.nn.Parameter(torch.zeros_like(self.Wrr));self.c=torch.nn.Parameter(torch.zeros_like(self.b))
  self.optimized=True
  self.use_compiled=False
 def run(self,idx,truncate=False):
  if truncate:raise RuntimeError('Only full BPTT is permitted')
  if self.stage=='A' and not self.use_compiled:return C.L.V.RNN.run(self,idx,False)
  B,T,A=idx.shape
  u=F.embedding(idx[:,:,0],self.Wrk,padding_idx=C.L.V.PAD) if A==1 else F.embedding_bag(idx.reshape(B*T,A),self.Wrk,mode='sum',padding_idx=C.L.V.PAD).view(B,T,-1)
  w=self.Wrr if self.stage=='A' else self.Wrr*self.mask;r=torch.zeros(B,256,dtype=self.Wrk.dtype)
  if self.stage!='C':
   if self.use_compiled:return recurrence(u,w,self.b,r)
   out=[]
   for t in range(T):r=torch.tanh(u[:,t]+r@w.T+self.b);out.append(r)
   return torch.stack(out,1)
  q=F.embedding(idx[:,:,0],self.Vin,padding_idx=C.L.V.PAD);s=self.Srr*self.mask;out=[]
  for t in range(T):
   a=torch.tanh(u[:,t]+r@w.T+self.b);g=torch.sigmoid(q[:,t]+r@s.T+self.c);r=(1-g)*r+g*a;out.append(r)
  return torch.stack(out,1)
 def enforce(self,opt):
  if self.stage=='A':return
  for p in [self.Wrr]+([self.Srr] if self.stage=='C' else []):
   assert bool((p[~self.mask.bool()]==0).all())
   if p in opt.state:
    for k in ('exp_avg','exp_avg_sq'):assert bool((opt.state[p][k][~self.mask.bool()]==0).all())
