"""Independent raw-row/branch/output audit. Never calls a learning routine."""
import copy,math
from collections import Counter,defaultdict
import numpy as np
import settings,data

def check(condition,message):
    if not condition:raise AssertionError(message)
def probs(p):
    a=np.asarray(p,dtype=float)
    check(a.shape==(9,) and np.isfinite(a).all() and (a>0).all() and abs(a.sum()-1)<1e-12,'probability schema')
    return a

def audit(receipt,check_source=True):
    settings.require_dev(receipt['world'])
    check(receipt['schema']=='ADDITION_DEV_RECEIPT' and receipt['science_worlds']==0,'DEV scope')
    if check_source:
        from runner import source_lock
        check(receipt['source']==source_lock(),'source changed during run')
    world=receipt['world'];fixture=receipt['fixture'];regenerated=data.generate(world,receipt['quick'])
    check(fixture==regenerated,'fixture altered')
    # Check the mathematical partition independently of the generator's truth field.
    held={(1,1),(1,2),(2,1),(0,4),(4,0),(2,3),(3,2),(3,3)}
    check(set(map(tuple,fixture['held_keys']))==held,'held orbit partition')
    train=set(map(tuple,fixture['old_keys']))|set(map(tuple,fixture['new_keys']))
    check(len(train)==17 and not train&held and train|held==set((a,b) for a in range(5) for b in range(5)),'exposure partition')
    check(set(a+b for a,b in map(tuple,fixture['old_keys']))==set(range(9)),'old output coverage')
    for phase,records in fixture['records'].items():
        counts=defaultdict(Counter)
        for r in records:
            a,b=r['pair'];check(r['truth']==48+a+b and (a,b) not in held,'truth/exposure')
            counts[(a,b)][r['noise_kind']]+=1
            check((r['target'] is None)==(r['noise_kind']==2),'missing provenance')
            check(r['noise_kind']!=0 or r['target']==r['truth'],'correct observation')
            check(r['noise_kind']!=1 or r['target']!=r['truth'],'corruption provenance')
        n=10 if receipt['quick'] else 40
        check(all(c==Counter({0:n*8//10,1:n//10,2:n//10}) for c in counts.values()),'noise budget')
        check(Counter(r['target'] for r in records)==Counter(r['shuffled_target'] for r in records),'shuffle frequency budget')
    phases=('old','day1') if receipt['quick'] else ('old','day1','new','day2')
    check(set(receipt['arms'])=={'NATIVE','RESIDUAL'},'arm roster')
    total_rows=total_training_bytes=0
    for arm,out in receipt['arms'].items():
        check(set(out['training'])==set(p for p in phases if not p.startswith('day')),'training phases')
        clocks={b:0. for b in settings.BRANCHES};bytes_seen={b:0 for b in settings.BRANCHES}
        for phase in phases:
            if phase.startswith('day'):
                for b in clocks:clocks[b]+=settings.DAY
            else:
                check(set(out['training'][phase])==set(settings.BRANCHES),'branch roster')
                for branch,traces in out['training'][phase].items():
                    records=fixture['records'][phase]
                    check(len(traces)==len(records),'training record count')
                    previous=b''
                    for record,trace in zip(records,traces):
                        raw=bytes.fromhex(record['shuffled_raw'] if branch=='SHUFFLED' else record['raw'])
                        check(bytes(r['byte'] for r in trace)==raw,'raw training bytes')
                        target=(record['shuffled_target'] if branch=='SHUFFLED' and phase=='old' else record['target'])
                        prompt_predictions=[];actual_targets=[]
                        for row in trace:
                            clocks[branch]+=settings.DT;bytes_seen[branch]+=1;total_training_bytes+=1;total_rows+=1
                            check(abs(row['t']-clocks[branch])<1e-7,'training clock')
                            check(row['plastic']==data.write_allowed(branch,phase),'branch decision')
                            check(bytes.fromhex(row['pre_line'])==previous,'pre-arrival sensor history')
                            byte=row['byte']
                            if byte==10:previous=b''
                            elif byte!=32:previous+=bytes([byte])
                            check(bytes.fromhex(row['post_line'])==previous,'post-arrival sensor history')
                            dose=np.asarray(row['write_l1'],float)
                            width=1 if arm=='RESIDUAL' and 'V3_ORDINAL' in receipt['version'] else 9
                            check(dose.shape==(width,) and np.isfinite(dose).all() and (dose>=0).all(),'write dose')
                            expected_target=target if byte in settings.ALPHABET and len(previous)==5 else None
                            check(row['observed_target']==expected_target,'target came from actual answer byte')
                            expected_update=expected_target is not None and data.write_allowed(branch,phase)
                            check(row['answer_update']==expected_update,'actual answer-write decision')
                            check(expected_update or not np.any(dose),'forbidden or missing-answer write')
                            if row['observed_target'] is not None:actual_targets.append(row['observed_target'])
                            pred=row['prediction']
                            if pred is not None:
                                check(byte==61 and len(previous)==4,'prediction timing')
                                pp=probs(pred['probabilities']);check(pred['emitted']==48+int(pp.argmax()),'own output selection')
                                check(len(pred['values'])==9 and np.isfinite(pred['values']).all(),'native value schema')
                                prompt_predictions.append(pred)
                        check(len(prompt_predictions)==1 and actual_targets==([] if target is None else [target]),'one pre-feedback prediction')
                    cost=out['costs'][phase+'/'+branch]
                    check(cost['raw_bytes']==sum(map(len,traces)) and cost['records']==len(records) and cost['cpu_seconds']>=0,'training cost')
            for branch in settings.BRANCHES:
                state=out['states'][phase][branch]
                check(abs(state['model_time']-clocks[branch])<1e-7 and state['bytes_seen']==bytes_seen[branch],'operative clock/counter')
                if 'V3_ORDINAL' in receipt['version']:
                    cert=state['actual_content_certificate'];check(all(math.isfinite(v) and v>=0 for v in cert.values()),'actual content certificate')
                    if branch=='N_OLD' and phase in ('old','day1'):check(not any(cert.values()),'actual forbidden old content')
                probe=out['probes'][phase][branch];check(probe['operative_digest']==state['digest'],'probe state unchanged')
                check(len(probe['rows'])==25 and len({tuple(r['pair']) for r in probe['rows']})==25,'probe roster')
                order=np.random.default_rng(world+1000+settings.PHASES.index(phase)).permutation(25)
                expected_pairs=[(int(i)//5,int(i)%5) for i in order]
                check([tuple(r['pair']) for r in probe['rows']]==expected_pairs,'prospective query order')
                probe_time=clocks[branch]
                computed=defaultdict(list)
                for r in probe['rows']:
                    a,b=r['pair'];truth=48+a+b;p=probs(r['probabilities'])
                    check(r['truth']==truth and r['emitted']==48+int(p.argmax()),'probe truth/choice')
                    check(r['exact']==(r['emitted']==truth) and r['error']==abs(r['emitted']-truth),'probe exact/MAE')
                    check(abs(r['loss_bits']+math.log2(p[truth-48]))<1e-12,'probe loss')
                    tr=r['trace'];check(bytes(x['byte'] for x in tr)==data.expression((a,b))+b'\n','probe supplies no answer')
                    for x in tr:
                        probe_time+=settings.DT;check(abs(x['t']-probe_time)<1e-7,'probe clock')
                    check(all(not x['answer_update'] and not any(x['write_l1']) and not x['plastic'] for x in tr),'read-only test')
                    check(sum(x['prediction'] is not None for x in tr)==1,'probe one emitted answer')
                    actual=next(x['prediction'] for x in tr if x['prediction'] is not None)
                    check(actual['emitted']==r['emitted'] and actual['probabilities']==r['probabilities'],'trace/readout linkage')
                    total_rows+=len(tr)
                    for name,pairs in (('old',data.OLD),('new',data.NEW),('held',data.HELD),('all',data.ALL)):
                        if (a,b) in pairs:computed[name].append(r)
                for name,selected in computed.items():
                    g=probe['groups'][name]
                    check(g['questions']==len(selected),'group count')
                    for key,field in (('exact','exact'),('mae','error'),('bpb','loss_bits')):
                        check(abs(g[key]-sum(x[field] for x in selected)/len(selected))<1e-12,'group arithmetic')
        erased=out['erased_final']
        if arm=='RESIDUAL' and 'V3_ORDINAL' in receipt['version']:
            logits=-.5*np.arange(9,dtype=float)**2;expected=np.exp(logits-np.logaddexp.reduce(logits))
        else:expected=np.ones(9)/9
        check(erased['erase'] and all(np.max(np.abs(probs(r['probabilities'])-expected))<1e-11 for r in erased['rows']),'content erase causal control')
        total_rows+=sum(len(r['trace']) for r in erased['rows'])
    check(receipt['training_histories']==6 and receipt['training_bytes']==total_training_bytes,'committed counts')
    return dict(verdict='PASS',audited_rows=total_rows,training_bytes=total_training_bytes,training_histories=6,science_worlds=0)

def countermodels(fixture):
    counts=Counter();per_left=defaultdict(Counter);per_right=defaultdict(Counter)
    xs=[];ys=[]
    for records in fixture['records'].values():
        for r in records:
            if r['target'] is not None:
                a,b=r['pair'];y=r['target']-48;counts[y]+=1;per_left[a][y]+=1;per_right[b][y]+=1;xs.append([1,a,b]);ys.append(y)
    def mode(c):return max(range(9),key=lambda y:(c[y],-y))
    frequency=mode(counts);beta=np.linalg.lstsq(np.asarray(xs),np.asarray(ys),rcond=None)[0]
    scores={'frequency':[],'left_only':[],'right_only':[],'exact_key_frequency_fallback':[],'commutative_key_frequency_fallback':[],'learned_linear_solvability_witness':[]}
    for a,b in data.HELD:
        truth=a+b
        predictions={'frequency':frequency,'left_only':mode(per_left[a]),'right_only':mode(per_right[b]),'exact_key_frequency_fallback':frequency,
            'commutative_key_frequency_fallback':frequency,'learned_linear_solvability_witness':int(np.clip(np.rint(np.dot([1,a,b],beta)),0,8))}
        for name,pred in predictions.items():scores[name].append(pred==truth)
    return {name:sum(v)/len(v) for name,v in scores.items()}
