"""Descriptive developmental curves only. E1 acquisition is kept apart from E3."""
import math
import statistics
from pathlib import Path
from artifacts import read,write,now
import bootstrap
import config
import tasks
from collections import Counter


def acc(rows):return sum(r['correct'] for r in rows)/len(rows)


def area(points):
    xs=[math.log2(n+1) for n,_ in points]
    return sum((b-a)*(y+z)/2 for a,b,(_,y),(_,z) in zip(xs,xs[1:],points,points[1:]))/(xs[-1]-xs[0])


def shortcuts(world):
    source=tasks.addition_streams(world);counts=Counter(y for _,y in source['old'])
    mode=counts.most_common(1)[0][0]
    scores={}
    table=dict(source['old'])
    for name,pairs in source['groups'].items():
        examples=[tasks.render(p) for p in pairs]
        scores[name]=dict(frequency_only=sum(y==mode for _,y in examples)/len(examples),
            exact_key_with_frequency_fallback=sum(table.get(p,mode)==y for p,y in examples)/len(examples),
            taught_keys=sum(p in table for p,_ in examples))
    return dict(modal_observed_answer=mode.hex(),scores=scores,
        scope='Evaluator-only correct-OLD exposure countermodels; not operative model outputs.')


def report():
    root=bootstrap.HERE;folder=root/'results';profiles=[]
    for p in sorted((folder/'track_b/receipts').glob('*.json.gz')):
        result=read(p);row=dict(world=result['world'],version=result['version'],histories={})
        for name,h in result['histories'].items():
            curves=[dict(phase=q['phase'],dose=q['dose'],production_cpu=q['production_cpu'],
                         **{k:acc(v) for k,v in q['scores'].items()}) for q in h['curve']]
            new=[(q['dose'],q['new']) for q in curves if q['phase']=='NEW']
            row['histories'][name]=dict(curve=curves,NEW_acquisition_AUC=area(new),
                 OLD_presentations=16384,NEW_presentations=1024,total_presentations=17408,
                 idle_cpu=h['rest_cpu'],query_and_guard_cpu=h['query_and_guard_cpu'],
                 forbidden_write_guard_cpu=h['forbidden_write_guard_cpu'],
                 production_training_cpu=h['resource']['production_cpu'])
        profiles.append(row)
    a=read(folder/'track_a/SUMMARY.json') if (folder/'track_a/SUMMARY.json').exists() else None
    contrasts=[]
    for row in profiles:
        W=row['histories']['W']
        for control in ('SH_OLD','N_OLD'):
            C=row['histories'][control]
            wzero=next(q for q in W['curve'] if q['phase']=='NEW' and q['dose']==0)
            czero=next(q for q in C['curve'] if q['phase']=='NEW' and q['dose']==0)
            contrasts.append(dict(world=row['world'],version=row['version'],control=control,
                NEW_AUC_difference=W['NEW_acquisition_AUC']-C['NEW_acquisition_AUC'],
                NEW_zero_shot_difference=wzero['new']-czero['new'],
                final_HELD_difference=W['curve'][-1]['held']-C['curve'][-1]['held'],
                OLD_loss_during_NEW_W=wzero['old']-W['curve'][-1]['old']))
    summary=dict(utc=now(),science=False,language_task=False,track_a=a,track_b_profiles=profiles,
        paired_descriptive_contrasts=contrasts,countermodels={str(w):shortcuts(w) for w in config.TRACK_B_WORLDS},
        goal='Unseen addition generalization per all external presentations and complete compute, with relevant OLD retention.',
        acquisition_curve_is_not_zero_shot_generalization=True,
        independently_paired_worlds=len({r['world'] for r in profiles}),
        track_b_jobs=len(profiles),track_b_correlated_histories=3*len(profiles),
        track_b_record_applications=52224*len(profiles),automatic_adoption=False,formal_statistics=False)
    write(folder/'SUMMARY.json',summary)
    lines=['# Native interface qualification and addition acquisition transfer','',
       'Developmental evidence only. No arithmetic rules, language task, attention, Transformer or ELM was added. Native and recurrent paths remain separate.', '',
       '## Track A — one bounded native-interface attempt','']
    if a:
        lines+=['| Stage | Behavioral result |','|---|---|']
        for d in a['decisions']:
            state='NOT RUN: predecessor failed' if d.get('not_run') else 'QUALIFIED on finite E1 fixture' if d['qualified'] else 'NOT QUALIFIED; fixed budget exhausted'
            lines.append(f"| {d['stage']} | {state} |")
        lines+=['','A1/A2 use externally scheduled single outputs; they are not autonomous sequence-generation results. A3 alone requires its own answer bytes and EOS. Target margins, empty/premature EOS and suffix challenges are in the stage receipts. No failed stage is repeated or enlarged.','']
        lines+=['| Stage | History | Clean acquired, immediate / rest | Suffix acquired, immediate / rest |',
                '|---|---|---:|---:|']
        for decision in a['decisions']:
            if decision.get('not_run'):continue
            receipt=read(folder/'track_a'/(decision['stage']+'.json.gz'))
            for name,h in receipt['histories'].items():
                entries=[]
                for stratum in ('clean','suffix_collision'):
                    entries.append(' / '.join(f"{sum(r['correct'] for r in p['scores']['rows'] if r['stratum']==stratum)}/{8 if stratum=='clean' else 4}" for p in h['probes'][-2:]))
                lines.append(f"| {decision['stage']} | {name} | {entries[0]} | {entries[1]} |")
        lines+=['']
    lines+=['## Track B — addition curves and retained OLD competence','',
       '| World | Reference | OLD history | NEW AUC | NEW final | HELD final | OLD final |',
       '|---|---|---|---:|---:|---:|---:|']
    for row in profiles:
        for history,h in row['histories'].items():
            final=h['curve'][-1]
            lines.append(f"| {row['world']} | {row['version']} | {history} | {h['NEW_acquisition_AUC']:.4f} | {final['new']:.1%} | {final['held']:.1%} | {final['old']:.1%} |")
    lines+=['','The NEW curve measures learning of subsequently taught items. NEW at dose 0 and always-untaught HELD are separately reported as transfer probes. W and N_OLD share the correct bytes; SH_OLD preserves OLD prompt and blockwise answer distributions but breaks their pairing. All histories share NEW order. The 16,384 prior presentations are charged in every history. N_OLD has birth weights and optimizer at NEW onset, so its later learning cannot be called previously trained arithmetic.','',
      'KEEP has 28,440 actor parameters; G62 has 28,800. Their recurrent weights have no autonomous decay, so the rest interval is a parameter no-op, not active consolidation. All output queries use own bytes and own EOS; exact retention after NEW is displayed.','',
      '### Paired descriptive contrasts','',
      '| World | Reference | W minus control | NEW AUC difference | NEW at zero difference | Final HELD difference |',
      '|---|---|---|---:|---:|---:|']
    for c in contrasts:
        lines.append(f"| {c['world']} | {c['version']} | {c['control']} | {c['NEW_AUC_difference']:+.4f} | {c['NEW_zero_shot_difference']:+.1%} | {c['final_HELD_difference']:+.1%} |")
    lines+=['',
      '## Costs and evidence limits','',
      'Track A includes actual prefeedback output readout and target updates. Parent scheduled-API guard costs remain in its API timing. Native forbidden-write checks, idle advancement, recurrent guards, read-only queries and independent audit costs are separately recorded. No cross-family parameter-matched ranking is claimed.','',
      'This implements the two requested steps under the addition goal. Correct OLD helping NEW learning is not itself proof of answering untaught additions. No model is automatically promoted and no formal-world campaign is launched. Language/construction and learned-native-hybrid proposals are archived, not implemented in this task.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
