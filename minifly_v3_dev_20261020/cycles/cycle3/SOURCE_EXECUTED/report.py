"""Post-qualification reduction of stored results; never trains a learner."""
import json
from pathlib import Path
import statistics
from collections import Counter
import numpy as np
from artifacts import HERE,atomic_json,sha
from audit import verify_scores,cpu_integral,sample_integral


def countermodels(receipt):
    tr=receipt['trajectories']['W'];final=tr['points'][-1]
    old=[p for p in tr['points'] if p['phase']=='OLD'][-1]
    taught=old['scores']['OLD']['rows'];held=final['scores']['HELD']['rows']
    freq=Counter(r['want'] for r in taught);default=sorted(freq,key=lambda x:(-freq[x],x))[0]
    first={};second={}
    for row in taught:
        p=bytes.fromhex(row['prompt']);first.setdefault(p[0],Counter())[row['want']]+=1;second.setdefault(p[2],Counter())[row['want']]+=1
    pick=lambda c:sorted(c,key=lambda x:(-c[x],x))[0]
    outputs=dict(exact_key_default=[],canonical_key_default=[],frequency=[],first_symbol=[],second_symbol=[])
    for row in held:
        p=bytes.fromhex(row['prompt'])
        for name in ('exact_key_default','canonical_key_default','frequency'):outputs[name].append(default)
        outputs['first_symbol'].append(pick(first[p[0]]));outputs['second_symbol'].append(pick(second[p[2]]))
    scores={k:sum(g==r['want'] for g,r in zip(v,held))/len(held) for k,v in outputs.items()}
    # Explicitly engineered evaluator witnesses; never loaded into the learner.
    symbols=sorted(set(bytes.fromhex(r['prompt'])[0] for r in taught));index={b:i for i,b in enumerate(symbols)}
    def feature(row):
        p=bytes.fromhex(row['prompt']);x=np.zeros(20);x[index[p[0]]]=1;x[10+index[p[2]]]=1;return x
    X=np.stack([feature(r) for r in taught]);HX=np.stack([feature(r) for r in held])
    labels=sorted(set(r['want'] for r in taught));Y=np.array([[float(r['want']==a) for a in labels] for r in taught])
    beta=np.linalg.lstsq(X,Y,rcond=None)[0];got=[labels[i] for i in (HX@beta).argmax(1)]
    scores['position_separable_categorical']=sum(a==r['want'] for a,r in zip(got,held))/len(held)
    y=np.array([int(bytes.fromhex(r['want'])) for r in taught]);value=np.linalg.lstsq(X,y,rcond=None)[0]
    scores['engineered_additive_witness']=float(np.mean(np.rint(HX@value)==[int(bytes.fromhex(r['want'])) for r in held]))
    return dict(training_scope='OLD-only unique examples; task-solvability witness is engineered and not a candidate',scores=scores)


def main():
    assert json.loads((HERE/'cycles/cycle3/AUDIT.json').read_text())['accepted']
    receipts=[json.loads(p.read_text()) for p in sorted((HERE/'cycles/cycle3/jobs').glob('*/RECEIPT.json'))]
    pc=[r['trajectories']['W']['old_end_production_cpu'] for r in receipts if r['arm']=='P-C']
    cstar=statistics.median(pc)
    atomic_json(HERE/'CPU_BUDGET_LOCK.json',dict(Cstar=cstar,engineering_PC_worlds=[r['world'] for r in receipts if r['arm']=='P-C'],
        engineering_PC_cpu=pc,rule='median complete OLD production CPU of P-C over two engineering worlds',
        scientific_HELD_used=False,technical_only=True))
    metrics={};worlds=sorted({r['world'] for r in receipts})
    for r in receipts:
        cells={}
        for h,tr in r['trajectories'].items():
            for p in tr['points']:verify_scores(p)
            final=tr['points'][-1];old=[p for p in tr['points'] if p['phase']=='OLD']
            cells[h]=dict(final_held=final['scores']['HELD']['accuracy'],final_old=final['scores']['OLD']['accuracy'],
                final_new=final['scores']['NEW']['accuracy'],sample_area=sample_integral(old),
                compute_area=cpu_integral(old,cstar),old_end=old[-1]['scores']['OLD']['accuracy'],
                held_old_end=old[-1]['scores']['HELD']['accuracy'],
                old_after_first_rest=next(p for p in tr['points'] if p['phase']=='OLD_REST')['scores']['OLD']['accuracy'],
                parameter_steps=tr['final_diagnostics']['parameter_steps'],
                local_update_l1=tr['final_diagnostics']['stats']['local_update_l1'],
                final_checkpoint=tr['final_checkpoint'])
        metrics[f"{r['world']}_{r['arm']}"]=cells
    arms={}
    for arm in ('F-C','F-G','P-C','P-G'):
        rows=[metrics[f'{w}_{arm}'] for w in worlds]
        arms[arm]={h:{k:statistics.mean(row[h][k] for row in rows) for k in
            ('final_held','final_old','final_new','sample_area','compute_area','old_end','held_old_end','old_after_first_rest')}
                   for h in ('W','N_OLD','SH_OLD')}
    comparisons={}
    for control in ('P-C','F-G'):
        comparisons[f'P-G_minus_{control}']={k:statistics.mean(metrics[f'{w}_P-G']['W'][k]-metrics[f'{w}_{control}']['W'][k]
            for w in worlds) for k in ('final_held','sample_area','compute_area')}
    for history in ('N_OLD','SH_OLD'):
        comparisons[f'P-G_W_minus_{history}']=statistics.mean(metrics[f'{w}_P-G']['W']['final_held']-metrics[f'{w}_P-G'][history]['final_held'] for w in worlds)
    positive=all(x>0 for key,value in comparisons.items() for x in (value.values() if isinstance(value,dict) else [value]))
    verdict='ENGINEERING_DIRECTIONAL_SIGNAL_PILOT_NOT_AUTHORIZED' if positive else 'NO_ADVANCE_ENGINEERING_NO_DEMONSTRATED_E3_GAIN'
    c1=json.loads((HERE/'cycles/cycle1/AUDIT.json').read_text());c2=json.loads((HERE/'cycles/cycle2/AUDIT.json').read_text())
    parent=json.loads((HERE/'cycles/cycle3/PARENT_POSITIVE_CONTROL.json').read_text())
    total=199680+parent['record_applications']+c1['record_applications']+c1['preliminary_record_applications']+c2['record_applications']
    result=dict(schema='MINIFLY_V3_THREE_CYCLE_SUMMARY',verdict=verdict,technical_only=True,science_records=0,
        engineering_worlds=worlds,complete_engineering_jobs=8,continuous_histories=24,
        full_life_record_applications=199680,total_record_applications_so_far=total,
        Cstar=cstar,arms=arms,comparisons=comparisons,per_world=metrics,
        countermodels={str(r['world']):countermodels(r) for r in receipts if r['arm']=='P-G'},
        parent_positive_control=parent,
        claim='E1 acquisition and retention; no confirmed E3 improvement; no E4',pilot_started=False,
        automatic_adoption=False,statistical_inference='Two engineering worlds are descriptive only.')
    atomic_json(HERE/'results/SUMMARY.json',result)
    rows=[]
    for arm,cells in arms.items():
        w=cells['W'];rows.append(f"| {arm} | {w['old_end']:.1%} | {w['old_after_first_rest']:.1%} | {w['final_old']:.1%} | {w['final_new']:.1%} | {w['final_held']:.1%} | {w['sample_area']:.5f} | {w['compute_area']:.5f} |")
    text='# MiniFly v3 three-cycle engineering report\n\n'+f"Verdict: **{verdict}**. Engineering only; no formal pilot or confirmation ran.\n\n"
    text+='The generic locally plastic sequence circuit feeds the retained native memory. There is no arithmetic parser, supplied numeric representation, independent answer head, attention, Transformer, ELM or BPTT learning path. Gradual warmup and cumulative-dose-matched steady controls were retained.\n\n'
    text+='## W history means over two engineering worlds\n\n| Arm | OLD end taught | After first day | OLD final | NEW final | HELD final | Sample area | Compute area |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'+'\n'.join(rows)+'\n\n'
    text+=f"Eight full world/arm jobs contain 24 correlated continuous histories and 199,680 external record applications. Including preliminary, first-cycle and second-cycle training, the recorded total is {total:,}; separately listed partial fixtures and read-only queries are not training records. Cstar is {cstar:.3f} production CPU seconds, from the preregistered engineering P-C median.\n\n"
    text+='## What passed and what it means\n\nCycle 1 verified the original archived native assets, scalar/batched native state and readout parity, local direct derivatives, payload coverage, frame timing, read-only generation and explicit EOS/truncation. An initial readout projection error was corrected before comparisons; its failed fixture records are retained.\n\nCycle 2 selected the first qualifying finite local-rate candidate using generic taught fixtures, without HELD feedback. It verified bounded local updates, frozen/plastic forward equivalence at birth and exact whole-record checkpoint/resume. Gradual teaching had lower early taught recall; that cost is retained in the curves.\n\nCycle 3 exercised the entire OLD/rest/NEW/rest chronology, all four arms and W/N_OLD/SH_OLD histories on two registered engineering worlds. The separate reducer checks immutable trace/receipt hashes, exact generated outputs, native residuals, gradual/steady multipliers, clocks, shuffled-label multisets and checkpoint hashes.\n\n'
    text+='Technical qualification is not evidence that the new representation learns arithmetic. All E3 comparisons and costs are reported in SUMMARY.json; taught acquisition or a stable checkpoint does not replace retained unseen-addition performance. No extra training, altered task or approximate-answer success criterion was introduced to rescue the outcome.\n'
    (HERE/'REPORT.md').write_text(text)
    print(json.dumps(dict(verdict=verdict,total_record_applications=total,arms=arms,comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
