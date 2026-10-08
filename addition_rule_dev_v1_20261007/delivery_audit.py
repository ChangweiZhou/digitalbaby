"""Independent postrun delivery checks and named receipt-tamper fixtures."""
import argparse,copy,gzip,json,math
from pathlib import Path
import numpy as np
import settings,data
from audit import audit,countermodels

def load(path):return json.loads(gzip.decompress(Path(path).read_bytes()))
def verify(receipt):
    result=audit(receipt)
    for arm,out in receipt['arms'].items():
        for bs in out['probes'].values():
            for probe in bs.values():
                previous=b''
                for row in probe['rows']:
                    for x in row['trace']:
                        if x['observed_target'] is not None:raise AssertionError('probe hidden target')
                        if bytes.fromhex(x['pre_line'])!=previous:raise AssertionError('probe prehistory')
                        if x['byte']==10:previous=b''
                        elif x['byte']!=32:previous+=bytes([x['byte']])
                        if bytes.fromhex(x['post_line'])!=previous:raise AssertionError('probe posthistory')
    return result

def tamper_suite(receipt):
    cases=[]
    def add(name,change):
        altered=copy.deepcopy(receipt);change(altered)
        try:verify(altered)
        except (AssertionError,ValueError,KeyError,IndexError):cases.append(name)
        else:raise AssertionError('accepted tamper: '+name)
    def first_target(r,branch='W'):
        return next(x for tr in r['arms']['RESIDUAL']['training']['old'][branch] for x in tr if x['observed_target'] is not None)
    def missing(r):
        return next(x for tr in r['arms']['RESIDUAL']['training']['old']['W'] for x in tr if x['byte']==10 and len(bytes.fromhex(x['pre_line']))==4)
    add('forbidden_old_branch',lambda r:first_target(r,'N_OLD').update(plastic=True))
    add('forbidden_write_dose',lambda r:first_target(r,'N_OLD').update(write_l1=[1.]))
    add('missing_answer_write',lambda r:missing(r).update(answer_update=True,write_l1=[1.]))
    add('wrong_observed_byte',lambda r:first_target(r).update(observed_target=57))
    add('dropped_byte',lambda r:r['arms']['RESIDUAL']['training']['old']['W'][0].pop())
    add('source_identity',lambda r:r['source']['files'].update({'learners.py':'wrong'}))
    add('truth_metadata',lambda r:r['fixture']['records']['old'][0].update(truth=100))
    add('heldout_exposure',lambda r:r['fixture']['records']['old'][0].update(pair=[1,1]))
    add('probe_hidden_target',lambda r:r['arms']['RESIDUAL']['probes']['day2']['W']['rows'][0]['trace'][0].update(observed_target=50))
    add('query_clock',lambda r:r['arms']['RESIDUAL']['probes']['day2']['W']['rows'][0]['trace'][0].update(t=0))
    add('operative_actual_content',lambda r:r['arms']['RESIDUAL']['states']['old']['N_OLD']['actual_content_certificate'].update(fast_l1=1.))
    add('probe_wrong_output',lambda r:r['arms']['RESIDUAL']['probes']['day2']['W']['rows'][0].update(emitted=99))
    add('summary_arithmetic',lambda r:r['arms']['RESIDUAL']['probes']['day2']['W']['groups']['held'].update(exact=.1))
    add('probability_nan',lambda r:r['arms']['RESIDUAL']['probes']['day2']['W']['rows'][0]['probabilities'].__setitem__(0,float('nan')))
    add('science_count',lambda r:r.update(science_worlds=1))
    return dict(verdict='PASS',rejected=len(cases),cases=cases,executed_native_records=0)

def summarize(receipts):
    final=receipts[-2:];results={}
    for arm in settings.ARMS:
        worlds=[]
        for r in final:
            out=r['arms'][arm];p=out['probes']['day2'];w=p['W']['groups'];cm=countermodels(r['fixture'])
            shortcuts=max(v for k,v in cm.items() if k!='learned_linear_solvability_witness')
            worlds.append(dict(world=r['world'],held_exact=w['held']['exact'],held_mae=w['held']['mae'],old_exact=w['old']['exact'],new_exact=w['new']['exact'],
                no_old_held=p['N_OLD']['groups']['held']['exact'],shuffled_held=p['SHUFFLED']['groups']['held']['exact'],
                gain_no_old=w['held']['exact']-p['N_OLD']['groups']['held']['exact'],gain_shuffled=w['held']['exact']-p['SHUFFLED']['groups']['held']['exact'],
                shortcut_gap=w['held']['exact']-shortcuts,countermodels=cm,mutable_bytes=out['states']['day2']['W']['mutable_bytes'],
                training_cpu=sum(v['cpu_seconds'] for k,v in out['costs'].items() if k.endswith('/W')),
                probe_cpu=sum(out['probes'][phase]['W']['cpu_seconds'] for phase in settings.PHASES)))
        means={k:sum(x[k] for x in worlds)/2 for k in ('held_exact','held_mae','old_exact','new_exact','no_old_held','shuffled_held','gain_no_old','gain_shuffled','shortcut_gap','mutable_bytes','training_cpu','probe_cpu')}
        guards=dict(held=means['held_exact']>=.75,gain_no_old=means['gain_no_old']>=.2,gain_shuffled=means['gain_shuffled']>=.2,
            old=means['old_exact']>=.75,new=means['new_exact']>=.75,shortcut=means['shortcut_gap']>=.1)
        results[arm]=dict(worlds=worlds,means=means,guards=guards,behavioral_feasibility=all(guards.values()))
    # For short cycle1, countermodels only see feedback actually delivered there.
    quick=receipts[0];old_only=copy.deepcopy(quick['fixture']);old_only['records']={'old':old_only['records']['old']}
    return dict(schema='ADDITION_THREE_CYCLE_SUMMARY',cycles_completed=3,dev_worlds=len(receipts),training_histories=sum(r['training_histories'] for r in receipts),
        training_bytes=sum(r['training_bytes'] for r in receipts),audited_rows=sum(r['independent_audit']['audited_rows'] for r in receipts),
        final_dev_worlds=[r['world'] for r in final],final_dev_audited_rows=sum(r['independent_audit']['audited_rows'] for r in final),
        results=results,verdict='DEV_FEASIBLE_NOT_CONFIRMED' if any(x['behavioral_feasibility'] for x in results.values()) else 'HOLD_RETENTION_GUARD',
        science_worlds=0,new_V2_adopted=False,cycle1_delivered_feedback_countermodels=countermodels(old_only),formal_launch_authorized=False,
        content_parameter_search=False,additional_cycles_authorized=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');args=p.parse_args();root=Path(args.root)
    paths=[root/'cycles'/('cycle1' if w==831001 else 'cycle2' if w==831101 else 'cycle3')/f'WORLD_{w}.json.gz' for w in settings.DEV_WORLDS]
    receipts=[load(p) for p in paths]
    accepted=[verify(r) for r in receipts[-2:]]
    (root/'cycles/cycle3/DELIVERY_AUDIT.json').write_text(json.dumps(dict(verdict='PASS',worlds=accepted,tamper=tamper_suite(receipts[-1])),indent=2)+'\n')
    summary=summarize(receipts);(root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(verdict=summary['verdict'],counts={k:summary[k] for k in ('dev_worlds','training_histories','training_bytes','audited_rows')},means={a:o['means'] for a,o in summary['results'].items()})),flush=True)
