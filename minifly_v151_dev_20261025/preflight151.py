"""Bounded technical qualification, including a separately paid native resume fixture."""
import os, sys, pickle, subprocess, time
import numpy as np
from support151 import H,R,D,COUNTS,TIMES,save,load,sha,clone,digest,prompt,values,intervene,functional_error,ledger,cpu
from assay151 import native, gradients, probabilities, project, warmup, teaching, ASSETS,SITES,source

def technical_step(m, y, ident):
    v=values(m);dirs,raw=teaching(m,v,y);dc=project(m.C+raw)-m.C
    return intervene(m,dirs['B_COMMON'],observed=True,delta_C=dc,ident=ident)

def child():
    m,y=pickle.loads((H/'TECH_RESUME_INPUT.pkl').read_bytes())
    assert m.target_n==2
    for k in range(2): m=technical_step(m,y,f'PREFLIGHT_RESUME_CHILD_{k}')
    (H/'TECH_RESUME_OUTPUT.pkl').write_bytes(pickle.dumps(m,protocol=5))
    save('TECH_CHILD_COUNTS.json',dict(COUNTS));save('TECH_CHILD_TIMES.json',dict(TIMES))

def run():
    t=time.process_time();C=native.fastmem.CODE.copy();v=np.linspace(-.4,.5,16);g,gc,_,_=gradients(C,v,256);eps=1e-6;err=[]
    f=lambda cc,vv:-probabilities(cc,vv)[1][256]
    for j in range(16):
        a=v.copy();b=v.copy();a[j]+=eps;b[j]-=eps
        err.append(abs((f(C,a)-f(C,b))/(2*eps)-g[j]))
    for i,j in [(0,0),(31,7),(256,15),(128,3)]:
        a=C.copy();b=C.copy();a[i,j]+=eps;b[i,j]-=eps
        err.append(abs((f(a,v)-f(b,v))/(2*eps)-gc[i,j]))
    assert max(err)<1e-6
    assert warmup(0)==1/16 and warmup(512)==17/32 and warmup(1024)==1
    assert np.linalg.matrix_rank(C)==15
    for y in range(257):assert int(np.argmax(probabilities(C,4*C[y])[0]))==y
    info=next(x for x in ASSETS if x['id']=='bryant_BIRTH');site=SITES['bryant']['sites'][0]
    root=native.Native('C',.1);m=source(root,info,site);original=digest(m);copy=clone(m)
    copy.C[0,0]+=.1;copy.theta[0]+=.1;copy.bank[0].fly.m.fast[0,0]+=.1
    assert digest(m)==original
    z=intervene(m,np.zeros(16),ident='PREFLIGHT_ZERO')
    p=intervene(m,np.zeros(16),passive=True,ident='PREFLIGHT_PASSIVE')
    vz=values(z);vp=values(p);assert functional_error(z,p)<1e-12 and np.max(abs(vz-vp))<1e-12
    # No target byte is fed by the event, only elapsed-time FE decay.
    expected=m.fe.clone();expected.advance(m.time+native.fastmem.DELTA)
    assert np.array_equal(z.fe.p,expected.p) and z.fe.t==expected.t
    response=[]
    for sign in [-1,1]:
        a=np.zeros(16);a[0]=sign*.001
        out=intervene(m,a,ident=f'PREFLIGHT_SIGN_{sign}');response.append(values(out)-vz)
    assert response[0][0]>0 and response[1][0]<0
    assert max(np.max(abs(r[1:])) for r in response)<1e-12
    for k in range(2):m=technical_step(m,site['source']['target'],f'PREFLIGHT_PREFIX_{k}')
    assert m.target_n==2
    (H/'TECH_RESUME_INPUT.pkl').write_bytes(pickle.dumps((m,site['source']['target']),protocol=5))
    parent=pickle.loads((H/'TECH_RESUME_INPUT.pkl').read_bytes())[0]
    for k in range(2):parent=technical_step(parent,site['source']['target'],f'PREFLIGHT_PARENT_{k}')
    env=os.environ.copy();env['V151_LEDGER']='TECH_CHILD_LEDGER.jsonl';env['V151_PRIOR_CPU']=str(cpu())
    ret=subprocess.run([sys.executable,str(H/'run.py'),'resume_child'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert ret.returncode==0,ret.stderr.decode()
    other=pickle.loads((H/'TECH_RESUME_OUTPUT.pkl').read_bytes())
    assert digest(parent)==digest(other) and parent.target_n==other.target_n==4
    COUNTS.update(load(H/'TECH_CHILD_COUNTS.json'));TIMES.update(load(H/'TECH_CHILD_TIMES.json'))
    result={'status':'PASS','finite_difference_max_error':max(err),'rank':15,'all257_reachable':True,
            'zero_passive_max_value_error':float(np.max(abs(vz-vp))),'zero_passive_functional_error':functional_error(z,p),
            'native_pulse_signs_correct':True,'no_target_byte_fed':True,'clone_independent':True,
            'cross_process_resume_equal':True,'resume_from_n':2,'resume_final_n':4,
            'preflight_target_applications':COUNTS['target_applications'],'preflight_branch_events':COUNTS['branch_events'],
            'preflight_nonzero_compartments':COUNTS['nonzero_compartment_increments'],
            'preflight_readout_vectors':COUNTS['native_vector_readouts'],'counts':dict(COUNTS),'CPU_parent':time.process_time()-t}
    assert result['preflight_target_applications']<=8 and result['preflight_branch_events']<=16 and result['preflight_nonzero_compartments']<=128 and result['preflight_readout_vectors']<=256
    save('CYCLE1_PREFLIGHT.json',result);ledger('PREFLIGHT_PASS',targets=result['preflight_target_applications'])
    return result
