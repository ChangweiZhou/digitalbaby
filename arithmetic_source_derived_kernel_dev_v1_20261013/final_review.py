"""Independent saved-receipt review; no learner import or trajectory replay."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
from auditor import audit
from reducer import metrics

ROOT=Path(__file__).resolve().parent


def independent_area(curve,end,compute=False):
    area=0.;last_x=0.;last_y=0.
    for point in curve:
        x=point['production_cpu'] if compute else point['external_W_coordinates']
        values=[]
        for probe in point['probes']:
            groups={}
            for row in probe['rows']:groups.setdefault(row['operation'],[]).append(int(row['actual']==row['expected'] and row['status']=='EOS'))
            values.append(sum(sum(v)/len(v) for v in groups.values())/len(groups))
        y=(values[0]+values[1])/2
        area+=(min(x,end)-last_x)*last_y
        if x>=end:return area/end
        last_x=x;last_y=y
    return (area+(end-last_x)*last_y)/end


def decoded_errors(rows,world):
    import random
    pi=list(range(256));random.Random(world+918273).shuffle(pi);inv={v:k for k,v in enumerate(pi)}
    decode=lambda value:bytes(inv[b] for b in bytes.fromhex(value)).decode(errors='replace')
    outputs=collections.Counter((decode(r['actual']),r['status']) for r in rows)
    zero=[r for r in rows if decode(r['expected'])=='0'];nonzero=[r for r in rows if decode(r['expected'])!='0']
    return dict(top_outputs=[dict(output=a,status=b,count=n) for (a,b),n in outputs.most_common(6)],
                zero_correct=sum(r['exact'] for r in zero),zero_queries=len(zero),
                nonzero_correct=sum(r['exact'] for r in nonzero),nonzero_queries=len(nonzero),
                first_byte_errors=sum(bytes.fromhex(r['actual'])[:1]!=bytes.fromhex(r['expected'])[:1] for r in rows),
                first_correct_suffix_or_end_wrong=sum(not r['exact'] and bytes.fromhex(r['actual'])[:1]==bytes.fromhex(r['expected'])[:1] for r in rows),
                empty_EOS=sum(r['status']=='EOS' and not r['actual'] for r in rows),
                truncation=sum(r['status']=='TRUNCATED' for r in rows),posthoc_only=True)


def run():
    receipts={};certificates=[];files=[]
    for cycle in (2,3):
        folder=ROOT/f'cycles/cycle{cycle}';expected=json.loads((folder/'SOURCE_LOCK.json').read_text())['source']
        # Verify original producer files against its lock, not current sources.
        for name,digest in expected.items():
            path=ROOT.parent/name[len('external:'):] if name.startswith('external:') else folder/'source'/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(cycle,name)
        for p in sorted((folder/'receipts').glob('*.json.gz')):
            r=json.load(gzip.open(p,'rt'));cert=audit(r,expected);certificates.append(cert)
            receipts[(cycle,r['arm'])]=r
            files.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),external_records=r['executed_external_records']))
    cstar=receipts[(3,'B69')]['branches']['W']['mixed_curve'][-1]['production_cpu']
    comparisons=[]
    for (cycle,arm),r in sorted(receipts.items()):
        reduced=metrics(r,cstar if cycle==3 else None)
        for name,branch in r['branches'].items():
            prefix=r['branches']['W']['binary_curve'] if name=='N_MIXED' else branch['binary_curve']
            curve=prefix+branch['mixed_curve']
            expected_area=independent_area(curve,r['binary_n']+r['mixed_n'])
            assert abs(expected_area-reduced[name]['J_sample'])<1e-12
            if cycle==3:assert abs(independent_area(curve,cstar,True)-reduced[name]['J_compute'])<1e-12
            # Behavioral invariance complements whole-state no-write hashes.
            if name=='N_BINARY':
                reference=[(x['actual'],x['status']) for p in branch['binary_curve'][0]['probes'] for x in p['rows']]
                for point in branch['binary_curve']:
                    assert [(x['actual'],x['status']) for p in point['probes'] for x in p['rows']]==reference
            if name=='N_MIXED':
                reference=[(x['actual'],x['status']) for p in r['branches']['W']['binary_curve'][-1]['probes'] for x in p['rows']]
                for point in branch['mixed_curve']:
                    assert [(x['actual'],x['status']) for p in point['probes'] for x in p['rows']]==reference
        w=reduced['W'];final=r['branches']['W']['mixed_curve'][-1]
        average=lambda d:sum(d.values())/len(d)
        comparisons.append(dict(cycle=cycle,arm=arm,trainable_parameters=r['trainable_parameters'],
                                J_sample=w['J_sample'],J_compute=w['J_compute'],production_CPU=w['production_cpu_end'],
                                job_body_CPU=r['total_process_cpu'],job_body_wall=r['total_wall'],
                                old_held=w['final']['old_held'],mixed_held=average(w['final']['new_held']),
                                old_taught=average(w['taught']['old_taught']),mixed_taught=average(w['taught']['new_taught']),
                                metrics_all_branches=reduced,
                                mixed_error_structure=decoded_errors(final['probes'][1]['rows'],r['world']),
                                peak_process_rss_bytes=r.get('peak_process_rss_job_bytes')))
    result=dict(scope='three-cycle technical preparation, not formal DEV/confirmation',scientific_jobs=0,
                native_technical_jobs=len(receipts),native_external_records=sum(r['executed_external_records'] for r in receipts.values()),
                correlated_final_histories=4*len(receipts),native_worlds=[202610138902,202610138903],
                generic_Torch_fixture_record_applications=65,paired_NumPy_oracle_fixture_record_applications=15,
                C_star=cstar,comparisons=comparisons,
                verdict='NO_TECHNICAL_EFFICACY_ADVANTAGE_FOR_ELM_NO_ADOPTION',
                no_family_wide_exclusion=True,final_weights_missing_for_completed_technical_jobs=True,
                checkpoint_limitation='Native job receipts and gradients preserved, but learned arrays were not saved. Birth arrays can be reproduced exactly; final feature geometry cannot be inspected without replay, which was not done.')
    (ROOT/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    audit_result=dict(accepted=True,scope='source, independent receipt/cost/count and reducer audit only;not core adoption',
                      native_jobs=len(receipts),native_external_records=result['native_external_records'],scientific_jobs=0,
                      original_producer_sources_checked=True,strengthened_current_auditor_used=True,
                      frozen_receipts_unchanged=True,independent_area_reduction=True,
                      no_write_behavioral_invariance=True,certificates=certificates,receipt_files=files,
                      limits=['one short andonefulltechnicalworld, not statistical confirmation','tensor allocator peaknotisolated','finallearnedweightsnotarchivedforalreadycompletedjobs'])
    (ROOT/'FINAL_AUDIT.json').write_text(json.dumps(audit_result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('native_technical_jobs','native_external_records','scientific_jobs','C_star','verdict')}))


if __name__=='__main__':run()
