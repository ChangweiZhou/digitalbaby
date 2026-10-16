"""Fixed HELD-vs-total-presentations reducer; technical fixtures cannot promote a core."""
import math
import statistics
from pathlib import Path
import config
from artifacts import read,write,now


def accuracy(rows):return sum(r['correct'] for r in rows)/len(rows)


def area(points):
    if len(points)<2 or len({x for x,y in points})!=len(points):raise ValueError('distinct fixed doses required')
    if points!=sorted(points):raise ValueError('ordered fixed doses required')
    x=[math.log2(t+1) for t,y in points]
    return sum((b-a)*(p[1]+q[1])/2 for a,b,p,q in zip(x,x[1:],points,points[1:]))/(x[-1]-x[0])


def profile(receipt):
    if receipt['technical'] or receipt['manifest']['world'] not in config.SCIENCE_WORLDS:
        raise ValueError('technical fixtures cannot enter primary science')
    result={}
    for history,h in receipt['histories'].items():
        qs=sorted((q for q in h['curve'] if q['phase']=='OBSERVATION' and 'GLOBAL' in q['tags']),key=lambda q:q['T'])
        if tuple(q['T'] for q in qs)!=config.GLOBAL_GRID:raise ValueError('primary checkpoint grid mismatch')
        final=next(q for q in h['curve'] if q['T']==17408 and q['phase']=='AFTER_REST')
        result[history]=dict(HELD_AUC=area([(q['T'],accuracy(q['scores']['held'])) for q in qs]),
          curve=[dict(T=q['T'],HELD=accuracy(q['scores']['held']),OLD=accuracy(q['scores']['old']),NEW=accuracy(q['scores']['new']),
                   production_cpu=q['production_cpu'],old_count=q['old_count'],new_count=q['new_count'],shortcut_envelope=q['shortcuts']['envelope']) for q in qs],
          final_HELD=accuracy(final['scores']['held']),final_OLD=accuracy(final['scores']['old']),final_NEW=accuracy(final['scores']['new']),
          final_shortcut_envelope=final['shortcuts']['envelope'])
    return result


def paired(profiles):
    if set(profiles)!={(w,s) for w in config.SCIENCE_WORLDS for s in config.SCHEDULES}:raise ValueError('complete fixed paired roster required')
    differences=[profiles[w,'INTERLEAVED']['W']['HELD_AUC']-profiles[w,'BLOCKED']['W']['HELD_AUC'] for w in config.SCIENCE_WORLDS]
    mean=statistics.mean(differences)
    from scipy.stats import t
    half=float(t.ppf(.975,7))*statistics.stdev(differences)/math.sqrt(8)
    return dict(primary_contrast='INTERLEAVED/W minus BLOCKED/W HELD AUC',world_differences=differences,
                mean=mean,paired_t_interval=[mean-half,mean+half],n_worlds=8,automatic_adoption=False)


def science_report(addition_folders,native_folders,output):
    output=Path(output);profiles={};raw={};native=[]
    for folder in addition_folders:
        r=read(Path(folder)/'RECEIPT.json.gz');assert read(Path(folder)/'AUDIT.json')['accepted']
        key=(r['manifest']['world'],r['manifest']['schedule'])
        if key in profiles:raise ValueError('duplicate world/schedule')
        profiles[key]=profile(r);raw[key]=r
    primary=paired(profiles)
    for w in config.SCIENCE_WORLDS:
        new=lambda s:{tag:q['operative_digest'] for q in raw[w,s]['histories']['N_OLD']['curve'] if q['phase']=='OBSERVATION' for tag in q['tags'] if tag.startswith('NEW:')}
        if new('BLOCKED')!=new('INTERLEAVED'):raise AssertionError('N_OLD equal-NEW state invariant failed')
    for folder in native_folders:
        r=read(Path(folder)/'RECEIPT.json.gz')
        if r['technical']:raise ValueError('technical native fixture in science')
        assert read(Path(folder)/'AUDIT.json')['accepted']
        W=r['histories']['W']['curve'][-2:]
        complete=all(all(row['correct'] for row in q['rows']) for q in W)
        native.append(dict(world=r['manifest']['world'],variant=r['manifest']['variant'],finite_fixture_qualified=complete,
                          scores=[dict(phase=q['phase'],correct=sum(x['correct'] for x in q['rows']),total=len(q['rows'])) for q in W]))
    if {(x['world'],x['variant']) for x in native}!={(w,v) for w in config.NATIVE_MAPS for v in config.NATIVE_VARIANTS}:raise ValueError('fixed native roster incomplete')
    summary=dict(utc=now(),primary=primary,profiles=[dict(world=w,schedule=s,histories=p) for (w,s),p in sorted(profiles.items())],native=native,
        record_applications=841728,correlated_histories=56,independent_addition_worlds=8,automatic_adoption=False,
        interpretation='Positive primary difference alone is not an addition-rule result; inspect causal controls, shortcuts, final HELD, OLD/NEW and compute.')
    write(output/'SUMMARY.json',summary)
    lines=['# ADDITION-ORDER / NATIVE-CONTEXT V1','',f"Primary paired HELD AUC difference: {primary['mean']:+.6f}; 95% paired t interval {primary['paired_t_interval']}.",'',
      '| World | Schedule | History | HELD AUC | Final HELD | Final OLD | Final NEW | Shortcut |','|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary['profiles']:
        for h,p in row['histories'].items():lines.append(f"| {row['world']} | {row['schedule']} | {h} | {p['HELD_AUC']:.4f} | {p['final_HELD']:.1%} | {p['final_OLD']:.1%} | {p['final_NEW']:.1%} | {p['final_shortcut_envelope']:.1%} |")
    lines+=['','Native functional outcomes are separate finite E1 tests; no automatic core adoption. Full world-level curves and causal controls are in SUMMARY.json.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
