"""Evaluator-only exact teacher Jacobian; never passed to cue geometry."""
import math
import numpy as np
from geometry import *

def finite(world,centered):
    d,rr=history(world);g=Geometry(centered);w=np.zeros((32,7744));direct=np.zeros((4,7744));last=0.;out={};clips=0
    labels=np.array(d['old_fact']['labels']+d['new_fact']['labels']);target=2*np.eye(4)[labels]-1
    def snapshot(name,at):
        f=g.panel([bytes.fromhex(c) for c in d['old_fact']['cues_hex']],at)
        decay=np.exp(-(at+12*DT-last)/TAU)
        l=f@w.T*decay;pred=l@target;got=f@direct.T*decay
        a=l[:,:16];q=int_basis();comp=q.T@a@q;ys=q@(q.T@target[:16]);formed=a@ys
        htarget=components(target[:16])[3].reshape(16,4)
        hformed=components(pred)[3].reshape(16,4)
        projections={}
        for key,c,y in zip(('B','F','G','H'),components(pred),components(target[:16])):
            projections[key]=float(np.sum(c*y)/np.sum(y*y)) if np.sum(y*y)>1e-15 else None
        j=np.ones((4,4))/4;c=np.eye(4)-j
        projectors={'B':np.kron(j,j),'F':np.kron(c,j),'G':np.kron(j,c),'H':np.kron(c,c)}
        cross={o:{i:float(np.linalg.norm(po@a@pi)) for i,pi in projectors.items()} for o,po in projectors.items()}
        error=float(np.max(np.abs(got-pred)))
        if clips==0 and error>1e-10:raise AssertionError('direct/operator mismatch')
        out[name]={'full_target_aligned_gain':float(np.sum(pred*target[:16])/np.sum(target[:16]**2)),'cross_component_operator_norms':cross,'time':at,'operator':l.tolist(),'symmetric_interaction_min':float(np.linalg.eigvalsh((comp+comp.T)/2)[0]),'interaction_singular':np.linalg.svd(comp,compute_uv=False).tolist(),'interaction_residual_fro':float(np.linalg.norm(comp-np.eye(9))),'old_only_target_aligned_gain':float(np.sum(ys*formed)/np.sum(ys*ys)),'full_target_component_gains':projections,'new_teacher_interaction_norm':float(np.linalg.norm(q.T@l[:,16:])),'added_score_forecast':(GAIN*pred).tolist(),'direct_replay_max_error':float(np.max(np.abs(got-pred))),'direct_clipping_coordinates':clips,'feature_spectrum':spectrum(f)}
    for r in rr:
        phi=g.feed_cue(bytes.fromhex(r['cue_hex']),r['at']);t=r['at']+12*DT
        dec=math.exp(-(t-last)/TAU);w*=dec;direct*=dec;last=t
        item=r['item']+(16 if r['stage']=='new' else 0);e=np.zeros(32);e[item]=1
        den=max(1.,phi@phi)
        w+=ETA*(e-w@phi)[:,None]*phi/den
        direct+=ETA*(target[item]-direct@phi)[:,None]*phi/den
        clips+=int(np.sum(np.abs(direct)>16));np.clip(direct,-16,16,out=direct)
        if r['index']==95:
            snapshot('old_end',96*RECORD_SECONDS);snapshot('old_day',96*RECORD_SECONDS+86400)
        if r['index']==191:
            snapshot('new_end',192*RECORD_SECONDS+86400);snapshot('final',192*RECORD_SECONDS+172800)
    return out,g

def kernel_metrics(native,candidate):
    n=len(native);h=np.eye(n)-np.ones((n,n))/n
    x=[]
    for a in range(6):
        for b in range(6):
            if a!=b:
                r=np.zeros(12);r[a]=1;r[6+b]=1;x.append(r)
    u,s,_=np.linalg.svd(h@np.array(x),full_matrices=False);q=u[:,s>1e-10];p=q@q.T
    def gram(f):
        k=f@f.T;g=h@k@h;tr=np.trace(g)
        if tr<1e-15:raise ValueError('zero centered kernel')
        return g/tr
    a=gram(native);b=gram(candidate)
    ra=a-h/(n-1);rb=b-h/(n-1);off=~np.eye(n,dtype=bool)
    norma=np.linalg.norm(ra);normb=np.linalg.norm(rb)
    cosine=float(np.dot(ra[off],rb[off])/(np.linalg.norm(ra[off])*np.linalg.norm(rb[off]))) if min(norma,normb)>1e-12 else None
    mass_a=float(np.trace(p@a));mass_b=float(np.trace(p@b))
    strength_a=float(norma/np.linalg.norm(a));strength_b=float(normb/np.linalg.norm(b))
    passed=mass_b>=.5*mass_a and min(norma,normb)>1e-12 and cosine is not None and cosine>=.5 and strength_b>=.5*strength_a
    return dict(shared_rank=int(len(q.T)),native_shared_mass=mass_a,candidate_shared_mass=mass_b,residual_cosine=cosine,native_residual_strength=strength_a,candidate_residual_strength=strength_b,pass_gate=bool(passed))

def native_panel(cues,at):
    import stores
    s,birth=stores.birth('native');out=[]
    for cue in cues:
        sensor=bb.bc.FE0();sensor.t=at
        for k,b in enumerate(cue):sensor.feed(b,at+k*DT)
        # Exact association_value read-only sensor clock advancement.
        sensor.advance(at+12*DT)
        out.append(s.model.encode_sparse(s.fly.m,sensor.read()))
    return np.array(out),{k:v for k,v in birth.items() if k!='fly_id'}
