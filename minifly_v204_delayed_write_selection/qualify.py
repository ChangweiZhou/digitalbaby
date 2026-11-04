"""Development-only checks for causal delayed write selection."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
import importlib.util
spec=importlib.util.spec_from_file_location('qa_base',ROOT.parent/'minifly-qa-generalization'/'qualify.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ARMS=('immediate','literal','left','arrived','position')
VIEWS=('original','missing','misleading','distractor')

def source(arm):
    s=base.program()
    s=s.replace("FOLLOW[0]=meta['follow']", "FOLLOW[0]=meta['follow']; NWR[:]=meta['write_counts']")
    s=s.replace("follow=FOLLOW[0],answer_keys=", "follow=FOLLOW[0],write_counts=NWR[:],answer_keys=")
    s=s.replace("location_start=b.rfind(b' ',0,len(b)-2)+1", """location_start=b.rfind(b' ',0,len(b)-2)+1
                distractor_start=-1
                if b' and then to the ' in b:
                    pivot=b.index(b' and then to the ')
                    distractor_start=b.rfind(b' ',0,pivot)+1
                if b', not the ' in b:
                    pivot=b.index(b', not the ')
                    distractor_start=location_start
                    location_start=b.rfind(b' ',0,pivot)+1""")
    s=s.replace('false_write=0,non_target=0','false_write=0,distractor_hits=0,non_target=0')
    s=s.replace("                    diag[field]+=1", "                    diag[field]+=1\n                    diag['distractor_hits']+=int(pos==distractor_start and gw>GTH)")
    if arm in ('immediate','literal','position'):
        if arm=='literal':
            s=s.replace("if arm=='oracle': gw=float(target)",
                        "if arm=='literal': gw=float(hist.endswith(b'the '))")
        return s
    s=s.replace('        for pos,y in enumerate(b):','        pending=[]\n        for pos,y in enumerate(b):')
    start=s.index("            if store and STOREMODE in")
    end=s.index("            hist.append(y);",start)
    s=s[:start]+'''            if store:
                pending.append((bytes(hist),y,len(STREAM)))
                if len(pending)==3:
                    ctx,_,index=pending[0]
                    arrived=bytes(item[1] for item in pending)
                    assert pending[-1][2]==index+2 and len(arrived)==3
                    if os.environ['DELAYED']=='arrived':
                        ids=np.unique(np.array([(hash((b'D',ctx[-k:],arrived))*2654435761+j*40503)%N
                                                 for k in (3,4,5) for j in range(2)],np.int64))
                    else:
                        ids=gate_ix(ctx)
                    gw=float(np.clip(G[ids].mean(),0,1))
                    if not FROZ[0]:
                        TAGS.append((ids,index,None))
                        del TAGS[:-300]
                    target=FROZ[0] and pos-2==location_start
                    if FROZ[0]:
                        field=('hit' if gw>GTH else 'miss') if target else ('false_write' if gw>GTH else 'non_target')
                        diag[field]+=1
                        diag['distractor_hits']+=int(pos-2==distractor_start and gw>GTH)
                    NWR[1]+=1
                    if gw>GTH:
                        NWR[0]+=1
                        for step,value in enumerate(arrived):
                            ix,v=bag(ctx,step=step)
                            if len(ix):
                                M[ix]=0
                                M[ix,value]=1
                                if FROZ[0]:
                                    diag['row_writes']+=len(ix)
                                    diag['follow']+=int(step>0)
                    pending.pop(0)
'''+s[end:]
    return s

def run(task):
    arm,view,folder,*checkpoint=task
    save,load,seed,batches=(*checkpoint,1,1) if len(checkpoint)==2 else checkpoint or (None,None,1,1)
    env=dict(os.environ,PYTHONHASHSEED=str(seed),QA_GOLD='0',QA_SAVE=str(save or ''),QA_LOAD=str(load or ''),
             QA_CONTROL={'position':'oracle','literal':'literal'}.get(arm,'learned'),
             DELAYED=arm,FREEZE='1',BIN='1',VALID='0',STORE='gated',EPOCHS='1',
             GORD='3,4,5',GTH='.15',QNOSTORE='1',STEPN='2',MATCHN='3',UNIT='1',
             SEG='1',WIN='32',HABT='0',CREDIT='match',RATE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=json.loads(subprocess.check_output(
        [sys.executable,'-B','-c',source(arm),str(base.TRAIN),str(folder/f'{view}.txt')],
        env=env,text=True))
    assert result['n']==100*batches and sum(result['gate_audit'][k] for k in ('hit','miss'))==200*batches
    return dict(arm=arm,view=view,seed=seed,**result)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        base.materialize(302,folder)
        canonical=(folder/'original.txt').read_text()
        (folder/'missing.txt').write_text(canonical.replace(' to the ',' toward that '))
        (folder/'misleading.txt').write_text(canonical.replace(' the ',' the quiet '))
        lines=[]
        for line in canonical.splitlines():
            if '\t' not in line:
                prefix,location=line.rsplit(' ',1)
                distractor=next(p for p in base.PLACES if p!=location[:-1])
                line=prefix+' '+distractor+' and then to the '+location
            lines.append(line)
        (folder/'distractor.txt').write_text('\n'.join(lines)+'\n')
        tasks=[(a,v,folder) for a in ARMS for v in VIEWS]
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(run,tasks))
        selected=tasks.index(('arrived','misleading',folder))
        checkpoint=folder/'trained.npz'
        assert run((*tasks[selected],checkpoint,None))==rows[selected]
        assert run((*tasks[selected],None,checkpoint))==rows[selected]
        assert all(r['gate_audit']['hit']==200 for r in rows if r['arm']=='position')
    report=dict(status='ENGINEERING_DEVELOPMENT_ONLY',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cycles=['Causal three-byte arrival and original-address writes checked inside the runner',
                        'Full development batch across five arms and four renderings',
                        'Fresh-process restore from a persisted training-boundary snapshot reproduced all scored bytes and diagnostics'],
                rows=rows)
    (ROOT/'ENGINEERING.json').write_text(json.dumps(report,indent=2)+'\n')
    print([(r['arm'],r['view'],r['exact_answer_with_own_newline'],r['gate_audit']['hit']) for r in rows])
