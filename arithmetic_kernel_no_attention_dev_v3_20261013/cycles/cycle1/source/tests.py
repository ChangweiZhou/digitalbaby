"""Generic interface, derivatives and state contracts; fixtures are not science."""
import ast, copy, json, math, time
from pathlib import Path
import torch
from torch import nn
from core import Learner, EOS, BEGIN, REPLY
from runner import ARMS, model_seed

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);checks=[];details={};t=time.monotonic()
    def check(name,truth):
        if not truth:raise AssertionError(name)
        checks.append(name)
    tree=ast.parse(Path('core.py').read_text())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    check('core imports no evaluator',not any(x in ('worlds','runner','auditor') for x in imports))
    check('core has no forbidden operator calls',not any(isinstance(n,ast.Attribute) and n.attr in ('MultiheadAttention','Transformer','scaled_dot_product_attention','einsum','bmm') for n in ast.walk(tree)))
    whitelist=(nn.Embedding,nn.Linear,nn.GRU,nn.GRUCell,nn.Conv1d,nn.ModuleList,nn.Sequential,nn.Tanh)
    for arm in ARMS:
        c=Learner(arm,model_seed(9001,arm));pcount=sum(p.numel() for p in c.net.parameters())
        check(arm+' module whitelist',all(type(m) in whitelist for name,m in c.net.named_modules() if name))
        before=c.digest();clone=c.clone()
        check(arm+' clone state parity',clone.digest()==before)
        check(arm+' independent parameters',all(a.data_ptr()!=b.data_ptr() for a,b in zip(c.net.parameters(),clone.net.parameters())))
        for k in range(16):
            rows=c.observe(bytes(range(k*16,k*16+16)),bytes([k*16,k*16+15]),False)
            check(arm+f' arbitrary payload block {k}',[r['target'] for r in rows]==[k*16,k*16+15,EOS])
        check(arm+' all-byte no-write freeze',c.digest()==before)
        c.generate(b'\x00\xff\n= ',cap=3)
        check(arm+' generation state purity',c.digest()==before)
        # Generic empty answer/EOS and bounded output fixtures.
        with torch.no_grad():c.net.head.weight.zero_();c.net.head.bias.fill_(-10);c.net.head.bias[EOS]=10
        check(arm+' own EOS',c.generate(b'anything')[0:2]==(b'','EOS'))
        with torch.no_grad():c.net.head.bias[EOS]=-10;c.net.head.bias[0]=10
        check(arm+' own truncation',c.generate(b'anything',cap=2)==(b'\x00\x00','TRUNCATED'))
        c=Learner(arm,model_seed(9001,arm));loss,_=c.loss_record(b'\x82\x07',b'\xab\x11',True)
        if c.config.kind=='P':
            later,_=c.loss_record(b'\x22',b'\x99',True);loss=(loss+later)/2
        loss.backward()
        group={name:float(p.grad.norm()) if p.grad is not None else 0. for name,p in c.net.named_parameters()}
        for prefix in ('E.','head.'):
            check(arm+' nonzero gradient '+prefix,any(v>0 for k,v in group.items() if k.startswith(prefix)))
        if arm=='P':check('P future error trains modulation policy',any(v>0 for k,v in group.items() if k.startswith('policy.')))
        if arm=='P_STOP':check('P_STOP severs policy credit',all(v==0 for k,v in group.items() if k.startswith('policy.')))
        if arm in ('G0','R1','C'):
            # Independent central differences on embedding, recurrence, output coordinates.
            for prefix in ('E.','head.', 'cell.' if arm=='C' else 'scan.'):
                name,p=next((name,p) for name,p in c.net.named_parameters() if name.startswith(prefix) and p.grad is not None)
                index=int(p.grad.abs().reshape(-1).argmax());analytical=float(p.grad.reshape(-1)[index]);eps=.002
                old=float(p.detach().reshape(-1)[index]);vals=[]
                for sign in (1,-1):
                    with torch.no_grad():p.reshape(-1)[index]=old+sign*eps
                    new,_=c.loss_record(b'\x82\x07',b'\xab\x11',False);vals.append(float(new.detach()))
                with torch.no_grad():p.reshape(-1)[index]=old
                numerical=(vals[0]-vals[1])/(2*eps)
                check(arm+' finite difference '+prefix,abs(numerical-analytical)<.002+abs(analytical)*.04)
        c=Learner(arm,model_seed(9001,arm))
        for _ in range(c.config.cadence):c.observe(b'\x01\x02',b'\x03\x04',True)
        c.flush();check(arm+' update bounded',c.last_update['applied_RMS']<=.00100001)
        before=c.digest();c.observe(b'other',b'answer',False)
        check(arm+' trained no-write freeze',c.digest()==before)
        # Same public BEGIN must reset transient computation; no teacher future access.
        with torch.no_grad():
            m=torch.zeros_like(c.m) if c.config.reset_modulation else c.m
            logits,_,_=c.net.decision([BEGIN,1,2,REPLY],None,m)
        generated,_=c.generate(b'\x01\x02',cap=0)
        # Compare first-decision logits directly with actual target rows.
        rows=c.observe(b'\x01\x02',b'\x07',False)
        check(arm+' prediction before first feedback',rows[0]['choice']==int(logits.argmax()))
        details[arm]={'parameters':pcount,'gradient_component_norms':group,'state_bytes':c.state_bytes()}
    shared=Learner('R8',model_seed(9001,'R8'));untied=Learner('RU8',model_seed(9001,'RU8'))
    check('shared and untied initial output parity',torch.equal(shared.net.decision([BEGIN,4,REPLY])[0],untied.net.decision([BEGIN,4,REPLY])[0]))
    check('GRU baseline parameter formula',details['G0']['parameters']==36593)
    result={'accepted':True,'checks':checks,'count':len(checks),'models':details,'elapsed':time.monotonic()-t,'scientific_jobs':0,'fixtures_not_science':True,'known_scope':'finite-difference coordinates, not all Jacobian entries; model source routes additionally manually reviewed'}
    (out/'CONTRACT_TESTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'accepted':True,'checks':len(checks),'seconds':result['elapsed']}),flush=True)
    return result
if __name__=='__main__':
    import sys
    run(sys.argv[1])
