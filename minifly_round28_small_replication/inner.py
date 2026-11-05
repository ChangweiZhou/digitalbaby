"""Generic plastic recurrent byte learner (no backprop, no attention). Genome parameterizes the local rules.
state: h_t = kWTA( Ein[x_t] + g_rec*Ws h + g_fast*Wf h )   (binary sparse, k active of N)
predict: score = softplus-free positive mixture: p = normalize( exp( (Ro + g_ro*Rf) h ) )
learning, all local:
  Ro  += lr_o * (y - p) h^T                                (slow delta rule readout)
  Rf   = lam_r*Rf + eta_r * m * y h^T                       (fast Hebbian state->next-byte association)
  Wf   = lam_f*Wf + eta_f * m * (A post pre^T + B pre + C post + D)   (fast recurrent plasticity, ABCD)
  m    = sigmoid(g0 + g1 * surprise)                         (neuromodulator from local prediction error)"""
import numpy as np, math
N=256; K=16
GENES=['lr_o','g_rec','g_fast','g_ro','eta_r','lam_r','eta_f','lam_f','A','B','C','D','g0','g1','g_in']
def decode(z):
    d=dict(zip(GENES,z))
    sg=lambda v:1/(1+math.exp(-v))
    return dict(lr_o=0.1*math.exp(d['lr_o']), g_rec=d['g_rec'], g_fast=d['g_fast'], g_ro=math.exp(d['g_ro']),
                eta_r=0.5*math.exp(d['eta_r']), lam_r=sg(3+d['lam_r']), eta_f=0.05*math.exp(d['eta_f']), lam_f=sg(3+d['lam_f']),
                A=d['A'],B=0.1*d['B'],C=0.1*d['C'],D=0.01*d['D'],g0=d['g0'],g1=d['g1'],g_in=math.exp(d['g_in']))
def life(z, data, ans_spans, score_from, seed=0, plastic=True):
    P=decode(z); rng=np.random.default_rng(seed)
    Ein=(rng.random((256,N))<0.05).astype(np.float32)*P['g_in']
    Ws=(rng.standard_normal((N,N))*(rng.random((N,N))<0.05)).astype(np.float32)
    Wf=np.zeros((N,N),np.float32); Ro=np.zeros((256,N),np.float32); Rf=np.zeros((256,N),np.float32)
    h=np.zeros(N,np.float32); bits=np.zeros(len(data))
    for t in range(len(data)):
        logit=(Ro+P['g_ro']*Rf)@h; logit-=logit.max(); p=np.exp(logit); p/=p.sum()
        y=data[t]; py=max(p[y],1e-6); bits[t]=-math.log2(py)
        if plastic:
            s=-math.log(py); m=1/(1+math.exp(-(P['g0']+P['g1']*s)))
            Ro[y]+=P['lr_o']*h; Ro-=P['lr_o']*np.outer(p,h)
            Rf*=P['lam_r']; Rf[y]+=P['eta_r']*m*h
        pre=h
        u=Ein[y]+P['g_rec']*(Ws@h)+P['g_fast']*(Wf@h)
        idx=np.argpartition(-u,K)[:K]; h=np.zeros(N,np.float32); h[idx]=1.0
        if plastic:
            Wf*=P['lam_f']
            if P['eta_f']*m>1e-6:
                Wf[idx]+=P['eta_f']*m*(P['A']*pre+P['C'])[None,:] if True else 0
                Wf+=P['eta_f']*m*(P['B']*pre[None,:]+P['D'])
    tot=bits[score_from:].sum()/max(1,len(data)-score_from)
    ab=[bits[a:a+l].sum() for a,l in ans_spans if a>=score_from]
    return tot, (sum(ab)/len(ab) if ab else 0.0)
