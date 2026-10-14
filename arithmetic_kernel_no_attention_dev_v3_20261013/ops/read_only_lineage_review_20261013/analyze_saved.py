"""Read-only old/new lineage review. No model imports, forward passes or training."""
import ast,collections,gzip,hashlib,json,math,random
from pathlib import Path

OUT=Path(__file__).parent
ROOT=OUT.parents[1]
def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)
def decode(hex_value,seed):
    pi=list(range(256));random.Random(seed+918273).shuffle(pi);inv={v:k for k,v in enumerate(pi)}
    raw=bytes(inv[b] for b in bytes.fromhex(hex_value))
    try:return raw.decode('ascii')
    except UnicodeDecodeError:return repr(raw)
def rate(correct,total):return correct/total if total else None

def training(rows):
    out={}
    for op in sorted({x['operation'] for x in rows}):
        rr=[x for x in rows if x['operation']==op];freq=collections.Counter(x['prompt'] for x in rr)
        last=rr[-min(256,len(rr)):];p=collections.Counter();loss=collections.defaultdict(list)
        for x in last:
            for i,q in enumerate(x['predictions']):
                key='EOS' if q['target']==256 else 'first_payload' if i==0 else 'later_payload'
                p[key+'_total']+=1;p[key+'_correct']+=q['choice']==q['target'];loss[key].append(q['loss'])
        out[op]={'records':len(rr),'unique_literal_prompts':len(freq),'unique_groups':len({x['group'] for x in rr}),
                 'exposures_per_literal_mean':len(rr)/len(freq),'exposures_per_literal_min':min(freq.values()),'exposures_per_literal_max':max(freq.values()),
                 'last_up_to_256_online_teacher_prefix_records':len(last),'online_accuracy':{k:rate(p[k+'_correct'],p[k+'_total']) for k in ('first_payload','later_payload','EOS')},
                 'online_mean_CE':{k:sum(v)/len(v) for k,v in loss.items()},
                 'warning':'Actual record-time predictions before its update, using earlier actual reply bytes; not evaluation of final saved weights'}
    return out

def output_analysis(probe,seed):
    rows=probe['rows'];errors=collections.Counter();correct_zero=correct_other=target_zero=target_other=0;outputs=collections.Counter()
    for x in rows:
        actual=decode(x['actual'],seed);expected=decode(x['expected'],seed);outputs[(actual,x['status'])]+=1
        zero=expected=='0';target_zero+=zero;target_other+=not zero
        correct_zero+=zero and x['exact'];correct_other+=(not zero) and x['exact']
        a,b=bytes.fromhex(x['actual']),bytes.fromhex(x['expected'])
        if x['exact']:errors['exact']+=1
        elif not a and x['status']=='EOS':errors['empty_EOS']+=1
        elif not a or a[0]!=b[0]:errors['wrong_first_byte']+=1
        elif x['status']=='TRUNCATED':errors['first_correct_truncated']+=1
        else:errors['first_correct_wrong_suffix_or_end']+=1
    return {'queries':len(rows),'errors':dict(errors),'zero_target_correct':correct_zero,'zero_target_count':target_zero,
            'nonzero_target_correct':correct_other,'nonzero_target_count':target_other,
            'most_common_outputs':[{'actual':a,'status':s,'count':n} for (a,s),n in outputs.most_common(6)],
            'posthoc_partition_not_new_gate':True}

def main():
    files=list((ROOT/'cycles/cycle3/receipts').glob('*.json.gz'))
    full=load(next(p for p in files if '202610139954_G0' in p.name))
    short=[load(p) for p in files if '202610139953_' in p.name]
    report={'scope':'Saved receipts, source and full inventory review only','new_forward_passes':0,'new_training_records':0,'scientific_jobs':0,
            'full_G0_training':{},'full_G0_final':{},'short_controls':{},'inventory_coverage':{},'source_hashes':{}}
    for segment in full['branches']['W']['segments']:report['full_G0_training'][segment['label']]=training(segment['records'])
    for name,b in full['branches'].items():report['full_G0_final'][name]={q['label']:output_analysis(q,full['world']) for q in b['final']}
    for r in short:
        b=r['branches']['W'];policy={k:v for k,v in b['last_gradient'].items() if k.startswith('policy.')}
        other={k:v for k,v in b['last_gradient'].items() if not k.startswith('policy.')}
        report['short_controls'][r['arm']]={'records':r['binary_n']+r['mixed_n'],'optimizer_steps':b['resource']['optimizer_steps'],
              'fast_state_bytes':b['state_bytes']['fast_modulation'],'last_policy_gradient_norm':math.sqrt(sum(v*v for v in policy.values())),
              'last_nonpolicy_gradient_norm':math.sqrt(sum(v*v for v in other.values())),
              'generation':{q['label']:output_analysis(q,r['world']) for q in b['final']},
              'gradient_measurement_scope':'Last window only; cannot infer the complete training history from it'}
    inv=ROOT.parents[0]/'arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.json'
    review=json.loads(inv.read_text());report['inventory_coverage']={k:len(v['rows']) for k,v in review['views'].items()}
    selected={'M001','M007','M010','M012','M022','M024','M027','M031','M033','M034','M035','M041','M050','M053','M059','M061','M076','M094','M096','M102','M106'}
    report['selected_original_mechanisms']=[x for x in review['views']['historical108']['rows'] if x['id'] in selected]
    relevant=['core.py','runner.py','tests.py','PREPARATION_DESIGN.md','REVISED_EXPERIMENT.md']
    for n in relevant:report['source_hashes'][str(ROOT/n)]=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
    report['source_hashes'][str(inv)]=hashlib.sha256(inv.read_bytes()).hexdigest()
    old=ROOT.parent/'arithmetic_stream_transfer_dev_v1_20261013/design/DESIGN.json';d=json.loads(old.read_text())
    report['old_generic_baseline']={'source':str(old),'architecture':d['learner'],'task':d['arithmetic'],
            'evidence':'Four final technical recipes reached100% taught + and * after17920 records per operation; retained held-case reuse remained weak. Not same task/dose/optimizer as V3.'}
    (OUT/'SAVED_RECEIPT_FINDINGS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'full_training':report['full_G0_training'],'full_final_W':report['full_G0_final']['W'],
                      'short_steps':{a:x['optimizer_steps'] for a,x in report['short_controls'].items()},
                      'short_policy_gradients':{a:x['last_policy_gradient_norm'] for a,x in report['short_controls'].items() if a.startswith('P')},
                      'inventory_coverage':report['inventory_coverage']},indent=2))
if __name__=='__main__':main()
