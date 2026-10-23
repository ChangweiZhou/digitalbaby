from rc_common import *
import head as H
def infer(D,std,hs):
 scale=np.load(P/'heads'/f'scale_{int(std)}.npy');X,base=H.features(D,scale,'MEAN')
 with np.load(P/'heads'/f'{int(std)}_{hs}.npz') as f:p={k:f[k] for k in H.NAMES}
 z=base.copy();z[:,:,8:9]=H.forward(p,X,base)[0];return z

def metric(z,vals):
 out=[];payload=[];stop=[];correct=[]
 for q,v in zip(z,vals):
  L=len(v);bs=[];ended=False
  for j in range(9):
   if q[j,8]>0:ended=True;break
   if j==8:break
   bs.append(sum(1<<b for b in range(8) if q[j,b]>0))
  out.append(dict(bytes=bs,EOS=ended));correct.append(ended and bs==v)
  payload.append(all(sum(1<<b for b in range(8) if q[j,b]>0)==t for j,t in enumerate(v)))
  stop.append(bool(np.all(q[:L,8]<=0) and q[L,8]>0))
 ix=[i for i,v in enumerate(vals) if v];eligible=[i for i in ix if payload[i]]
 return dict(nonempty_exact=float(np.mean([correct[i] for i in ix])) if ix else None,payload_exact=float(np.mean([payload[i] for i in ix])) if ix else None,conditional_n=len(eligible),conditional_stop=float(np.mean([stop[i] for i in eligible])) if eligible else None,exact=float(np.mean(correct)),outputs=out)

def evaluate(tag):
 meta=json.loads((P/'traces'/f'{tag}.json').read_text());rows=[]
 with np.load(P/'traces'/f'{tag}.npz') as f:
  for i,(pool,vals) in enumerate(zip(meta['pools'],meta['labels'])):
   D=f[f'D{i}'];rows.append(dict(pool=pool,head='DIRECT',lesion='NONE',metrics=metric(D.mean(-1),vals)))
   for hs in range(202610239401,202610239404):
    for lesion in (('NONE','ZERO','FOREIGN') if meta['arm']=='W' else ('NONE',)):
     src=D if lesion=='NONE' else np.zeros_like(D) if lesion=='ZERO' else np.roll(D,1,axis=0)
     z=infer(src,meta['STD'],hs);assert np.array_equal(z[:,:,:8]>0,src.mean(-1)[:,:,:8]>0)
     rows.append(dict(pool=pool,head=hs,lesion=lesion,metrics=metric(z,vals)))
 save(f'traces/{tag}_outputs.json',dict(**meta,results=rows))

