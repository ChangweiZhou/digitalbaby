"""Independent byte/coordinate reducer; never imports the learner or runner update."""
import math
from collections import Counter,defaultdict
import numpy as np
import settings,data
from representation import SharedEncoder
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def close(x,y,msg,tol=3e-10):
    a=np.asarray(x);b=np.asarray(y)
    require(a.shape==b.shape and np.isfinite(a).all() and np.max(np.abs(a-b),initial=0)<=tol,msg)
def supports(enc):
    features=[32*r+j for r in range(2) for j in list(range(5))+list(range(8,12))+list(range(16,20))+[20]]
    chosen=np.random.default_rng(20261007).permutation(np.flatnonzero(enc.mask))[:420]
    result={}
    for pair in data.ALL:
        selected=[]
        for role,d in enumerate(pair):
            active=[32*role+d,32*role+20]+[32*role+(8+t if d>t else 16+t) for t in range(4)]
            for pn in active:
                i=features.index(pn);selected.extend(chosen[15*i:15*(i+1)])
        ids=np.sort(np.array(selected));require(len(ids)==180,'independent active support')
        actual=np.flatnonzero((enc.code(pair)>0)&enc.mask)
        require(np.array_equal(ids,actual),'fixed projection changed')
        result[pair]=ids
    return result
def numeric_update(arr,arm,ids,y,value,t,allowed):
    # Separate equations and temporaries from operative learner implementation.
    f=arr['fast'][ids];s=arr['slow'][ids];vis=arr['visits'][ids].copy()
    r=float(y-48-value);info={'content_l1':0.,'auxiliary_l1':0.,'attenuation_l1':0.,'transfer_l1':0.,'commits':0,'clipped':0,'gate_counts':[0,0,0],'raw_residual':r}
    if allowed:
        steps=(max(-1.,min(1.,r))/math.sqrt(len(ids)))/np.sqrt(1+vis)
        if arm=='STABLE':
            a=f+.2*steps;b=s+.8*steps;info['clipped']=int(((np.abs(a)>2)|(np.abs(b)>2)).sum());a=np.maximum(-2,np.minimum(2,a));b=np.maximum(-2,np.minimum(2,b))
        elif arm=='U042-WG1':
            old=arr['summary'][ids].copy();gain=np.full(len(ids),.25)
            gain[(vis<8)|(np.maximum(1,2*old)>=abs(r))]=1
            if abs(r)<=.1:gain[:]=0
            a=f+.2*steps*gain;b=s+.8*steps*gain
            info['clipped']=int(((np.abs(a)>2)|(np.abs(b)>2)).sum());a=np.maximum(-2,np.minimum(2,a));b=np.maximum(-2,np.minimum(2,b))
            nxt=.875*old+.125*min(abs(r),8);arr['summary'][ids]=nxt
            info['auxiliary_l1']=float(np.abs(nxt-old).sum());info['gate_counts']=[int((gain==x).sum()) for x in (0,.25,1)]
        else:
            old=arr['run'][ids].copy();candidate=f/2+steps;info['clipped']=int((np.abs(candidate)>2).sum());a=np.maximum(-2,np.minimum(2,candidate));b=s.copy()
            sign=0 if abs(r)<=.1 else (1 if r>0 else -1)
            counts=np.zeros(len(ids),np.int8)
            if sign:
                counts[:]=sign
                same=np.sign(old)==sign;counts[same]=sign*np.minimum(np.abs(old[same])+1,2)
            ready=(abs(counts)==2)&(counts*a>0);proposal=s[ready]+a[ready]
            info['clipped']+=int((np.abs(proposal)>2).sum())
            b[ready]=np.maximum(-2,np.minimum(2,proposal));carry=b[ready]-s[ready];a[ready]=a[ready]-carry
            counts[ready]=0;arr['run'][ids]=counts
            info.update(auxiliary_l1=float(np.abs(counts.astype(float)-old).sum()),attenuation_l1=float(np.abs(f/2).sum()),transfer_l1=float(np.abs(carry).sum()),commits=int(ready.sum()))
        info['content_l1']=float(np.abs(a-f).sum()+np.abs(b-s).sum());arr['fast'][ids]=a;arr['slow'][ids]=b
        vv=np.minimum(vis+1,65535);arr['visits'][ids]=vv;info['auxiliary_l1']+=float(np.abs(vv-vis).sum())
    arr['last'][ids]=t
    return info
def probabilities(value):
    logits=np.array([-.5*(k-value)**2 for k in range(9)])
    return np.exp(logits-np.logaddexp.reduce(logits))
def audit(receipt,check_source=True):
    settings.require_world(receipt['world']);arm=receipt['arm'];dose=receipt['dose']
    require(receipt['schema']=='ADDITION_EFFICIENCY_TECH_JOB' and receipt['technical'] and receipt['science_worlds']==0,'technical-only admission')
    require(arm in settings.ARMS and dose in settings.DOSES,'job roster')
    if check_source:
        from runner import source_lock
        require(receipt['source']==source_lock(),'source lock identity')
    fixture=receipt['master'];require(fixture==data.generate(receipt['world']),'master source provenance')
    held={(0,4),(1,1),(1,2),(2,1),(2,3),(3,2),(3,3),(4,0)}
    require(set(data.HELD)==held and len(data.OLD)==13 and len(data.NEW)==4,'partition')
    for phase,blocks in fixture['blocks'].items():
        keys=set(data.OLD if phase=='OLD' else data.NEW)
        require(len(blocks)==4,'four blocks')
        for entries in blocks:
            c=defaultdict(Counter)
            for r in entries:
                pair=tuple(r['pair']);a,b=pair
                require(pair in keys and pair not in held,'held exposure')
                require(r['truth']==48+a+b,'environment truth')
                c[pair][r['kind']]+=1
                require((r['target'] is None)==(r['kind']==2),'missing category')
                require(r['kind']!=0 or r['target']==r['truth'],'correct byte provenance')
                require(r['kind']!=1 or r['target'] in settings.ALPHABET and r['target']!=r['truth'],'wrong byte provenance')
            require(set(c)==keys and all(v==Counter({0:8,1:1,2:1}) for v in c.values()),'block counts')
            require(Counter(r['target'] for r in entries)==Counter(r['shuffled_target'] for r in entries),'shuffled histogram')
    enc=SharedEncoder();support=supports(enc);n=enc.n
    require(set(receipt['branches'])==set(settings.BRANCHES),'three histories')
    require(len({o['initial_digest'] for o in receipt['branches'].values()})==1,'identical branch prestates')
    expected_birth={k:v for k,v in enc.birth.items() if k!='fly_id'}
    require(all({k:v for k,v in o['birth'].items() if k!='fly_id'}==expected_birth for o in receipt['branches'].values()),'canonical birth values')
    totalbytes=records=checked=0;maxerr=0.
    for branch,out in receipt['branches'].items():
        arr={k:np.zeros(n) for k in ('fast','slow','last','visits')}
        if arm=='U042-WG1':arr['summary']=np.zeros(n)
        if arm=='U021-SF1':arr['run']=np.zeros(n)
        tick=seen=0
        for phase,end,day in (('OLD','OLD_END','DAY1'),('NEW','NEW_END','FINAL')):
            entries=[(b,r) for b,block in enumerate(fixture['blocks'][phase]) if b<dose//10 for r in block]
            require(len(out['training'][phase])==len(entries),'training row dropped')
            allowed=not(branch=='N_OLD' and phase=='OLD');phasebytes=0
            for (block,r),actual in zip(entries,out['training'][phase]):
                raw=data.expression(r['pair'])
                target=r['shuffled_target'] if branch=='SHUFFLED' and phase=='OLD' else r['target']
                raw+=(bytes([target]) if target is not None else b'')+b'\n'
                require(bytes(row['byte'] for row in actual['trace'])==raw,'delivered target provenance')
                require(actual['block']==block and actual['padding_ticks']==6-len(raw),'slot padding')
                ids=support[tuple(r['pair'])];line=b'';value=None
                for i,row in enumerate(actual['trace']):
                    tick+=1;seen+=1;phasebytes+=1;checked+=1
                    require(abs(row['t']-tick*settings.DT)<1e-7,'byte clock')
                    require(row['plastic']==allowed,'permitted branch')
                    require(row['pre_line']==line.hex(),'sensor prestate')
                    b=row['byte'];line=b'' if b==10 else line+bytes([b]);require(row['post_line']==line.hex(),'sensor poststate')
                    answer=target if target is not None and i==4 else None
                    require(row['observed_target']==answer and row['update_allowed']==bool(answer is not None and allowed),'actual write decision')
                    if b==61:
                        value=float(np.sum(arr['fast'][ids]+arr['slow'][ids])/math.sqrt(len(ids)))
                        pred=row['prediction'];require(pred is not None,'pre-feedback output absent')
                        close(pred['numeric_value'],value,'actual F+S readout')
                        p=probabilities(value);close(pred['probabilities'],p,'ordinal probabilities');require(pred['emitted']==48+int(np.argmax(p)),'model own output')
                    else:require(row['prediction'] is None,'unexpected output timing')
                    if answer is not None:
                        expected=numeric_update(arr,arm,ids,answer,value,tick*settings.DT,allowed)
                        require(row['update'] is not None,'missing update ledger')
                        require(set(row['update'])==set(expected),'update ledger schema')
                        for k in expected:close(row['update'][k],expected[k],'independent update '+k,tol=3e-8)
                    else:require(row['update'] is None,'missing/nonanswer update')
                tick+=actual['padding_ticks'];records+=1
            inactive=4-dose//10;perblock=len(data.OLD if phase=='OLD' else data.NEW)*10
            tick+=inactive*perblock*6
            costs=out['costs'][phase]
            require(costs['records']==len(entries) and costs['training_bytes']==phasebytes and costs['core_cpu_seconds']>=0,'cost/bytes')
            totalbytes+=phasebytes
            for cp,isday in ((end,False),(day,True)):
                if isday:tick+=settings.DAY_TICKS
                state=out['states'][cp]
                require(state['ticks']==tick and abs(state['time']-tick*settings.DT)<1e-7 and state['bytes_seen']==seen,'checkpoint clock/byte count')
                require(set(state['arrays'])==set(arr),'all operative arrays required')
                for key,a in arr.items():
                    close(state['arrays'][key],a,'actual checkpoint '+key,tol=1e-7 if key=='last' else 3e-9)
                    maxerr=max(maxerr,float(np.max(np.abs(np.asarray(state['arrays'][key])-a))))
                    if branch=='N_OLD' and phase=='OLD' and key!='last':require(not np.any(state['arrays'][key]),'forbidden auxiliary/content old write')
                require(state['payload_bytes']==(36 if arm=='U042-WG1' else 29 if arm=='U021-SF1' else 28)*n,'state payload')
                probe=out['probes'][cp];require(probe['parent_digest']==state['digest'],'query parent identity')
                order=np.random.default_rng(receipt['world']+1000+settings.PHASES.index(cp)).permutation(25)
                require([r['pair'] for r in probe['rows']]==[list(data.ALL[i]) for i in order],'query roster')
                qt=tick
                for row in probe['rows']:
                    pair=tuple(row['pair']);ids=support[pair];val=float(np.sum(arr['fast'][ids]+arr['slow'][ids])/math.sqrt(len(ids)));p=probabilities(val)
                    close(row['numeric_value'],val,'probe operative numeric value');close(row['probabilities'],p,'probe operative probabilities')
                    require(row['emitted']==48+int(np.argmax(p)) and row['exact']==(row['emitted']==48+sum(pair)) and row['truth']==48+sum(pair),'probe own exact answer')
                    require(bytes(r['byte'] for r in row['trace'])==data.expression(pair)+b'\n','probe feedback absent')
                    for j,event in enumerate(row['trace']):
                        qt+=1;checked+=1
                        require(abs(event['t']-qt*settings.DT)<1e-7 and not event['plastic'] and not event['update_allowed'] and event['observed_target'] is None and event['update'] is None,'query clock or mutation')
                        if j==3:
                            require(event['prediction'] is not None,'query output absent');close(event['prediction']['numeric_value'],val,'query trace/value link');close(event['prediction']['probabilities'],p,'query trace/probability link');require(event['prediction']['emitted']==row['emitted'],'query trace/output link')
                        else:require(event['prediction'] is None,'query extra output')
                for name,pairs in (('old',data.OLD),('new',data.NEW),('held',data.HELD)):
                    close(probe['groups'][name],np.mean([r['exact'] for r in probe['rows'] if tuple(r['pair']) in pairs]),'probe group arithmetic')
        require(tick==84720,'final clock parity')
        require(out['erased']['erase'],'erasure required')
        for row in out['erased']['rows']:
            close(row['probabilities'],probabilities(0),'erase prior');require(row['emitted']==48,'erase emitted prior')
            require(all(not e['update_allowed'] and e['update'] is None for e in row['trace']),'erase query mutated');checked+=len(row['trace'])
        require(len(out['erased']['rows'])==25,'erase roster')
    require(receipt['training_histories']==3 and receipt['training_bytes']==totalbytes and receipt['training_records']==records,'committed job counts')
    return dict(verdict='PASS',independent_rows=checked,training_bytes=totalbytes,training_records=records,technical_histories=3,science_worlds=0,max_checkpoint_difference=maxerr)
def countermodels(master,dose):
    counts=Counter();left=defaultdict(Counter);right=defaultdict(Counter)
    for blocks in master['blocks'].values():
        for entries in blocks[:dose//10]:
            for r in entries:
                if r['target'] is not None:
                    a,b=r['pair'];y=r['target']-48;counts[y]+=1;left[a][y]+=1;right[b][y]+=1
    mode=lambda c:max(range(9),key=lambda y:(c[y],-y))
    predictions={'frequency':lambda a,b:mode(counts),'left_only':lambda a,b:mode(left[a]),'right_only':lambda a,b:mode(right[b]),'exact_key_frequency_fallback':lambda a,b:mode(counts),'commutative_key_frequency_fallback':lambda a,b:mode(counts),'zero_content_prior':lambda a,b:0}
    return {name:float(np.mean([fn(a,b)==a+b for a,b in data.HELD])) for name,fn in predictions.items()}
