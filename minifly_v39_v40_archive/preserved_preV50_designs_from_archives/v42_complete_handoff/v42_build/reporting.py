"""V42 fixed-history audit reports; consumes completed evaluator records only.

No fitting, training, policy selection, or modification of scientific records.
"""
from __future__ import annotations
import csv
import itertools
import json
import math
from pathlib import Path
import statistics

POLICIES = ('READ-HYBRID', 'SLOT-ONLY', 'EXACT-ONLY', 'ANCHOR-ONLY', 'SUPER')
POLICY_MASKS = {
    'READ-HYBRID': {'exact_slot', 'exact_trace', 'supported_slot'},
    'SLOT-ONLY': {'exact_slot', 'supported_slot'},
    'EXACT-ONLY': {'exact_slot', 'exact_trace'},
    'ANCHOR-ONLY': {'exact_slot'},
    'SUPER': set(),
}
CONTRASTS = (
    ('SLOT-ONLY', 'READ-HYBRID', 'primary'),
    ('SLOT-ONLY', 'SUPER', 'primary'),
    ('EXACT-ONLY', 'READ-HYBRID', 'diagnostic'),
    ('ANCHOR-ONLY', 'SLOT-ONLY', 'diagnostic'),
    ('READ-HYBRID', 'SUPER', 'reference'),
)

def _native(x):
    if hasattr(x, 'item'):
        return _native(x.item())
    if isinstance(x, dict):
        return {str(k): _native(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_native(y) for y in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _json(path, data):
    Path(path).write_text(json.dumps(_native(data), indent=2, sort_keys=True, allow_nan=False) + '\n')


def _csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_native(v), sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _summary(values):
    values = [float(x) for x in values if _finite(x)]
    n = len(values)
    if not n:
        return {'n_seeds': 0, 'mean': None, 'sd': None, 'ci95_low': None, 'ci95_high': None, 'df': None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    lo = hi = None
    if n > 1:
        from scipy.stats import t
        margin = float(t.ppf(.975, n - 1)) * sd / math.sqrt(n)
        lo, hi = mean - margin, mean + margin
    return {'n_seeds': n, 'mean': mean, 'sd': sd, 'ci95_low': lo, 'ci95_high': hi, 'df': n-1 if n > 1 else None}


def _paired(seed_values):
    ordered = sorted((int(seed), float(value)) for seed, value in seed_values.items())
    values = [v for _, v in ordered]
    result = _summary(values)
    result['seed_differences'] = {str(seed): value for seed, value in ordered}
    result['student_t'] = result['student_p_two_sided'] = result['signflip_p_two_sided'] = None
    result['zero_sample_variance'] = False
    n = len(values)
    if n:
        observed = abs(statistics.mean(values))
        # A finite reference distribution over signs, not a claim of randomized assignment.
        null = [abs(sum(sign * value for sign, value in zip(signs, values)) / n)
                for signs in itertools.product((-1, 1), repeat=n)]
        tolerance = max(1e-14, observed * 1e-12)
        result['signflip_p_two_sided'] = sum(x >= observed - tolerance for x in null) / len(null)
    if n > 1:
        from scipy.stats import t
        sd = result['sd']
        if sd == 0:
            result['zero_sample_variance'] = True
            # Avoid a misleading numerical p=0 when the observed equality is deterministic.
            result['student_p_two_sided'] = 1.0 if result['mean'] == 0 else None
        else:
            tv = result['mean'] / (sd / math.sqrt(n))
            result['student_t'] = tv
            result['student_p_two_sided'] = float(2 * t.sf(abs(tv), n - 1))
    return result


def _unwrap(records):
    values = list(records.values()) if isinstance(records, dict) else list(records)
    result = []
    for rec in values:
        if 'data' in rec and isinstance(rec['data'], dict) and 'seed' not in rec:
            rec = rec['data']
        if 'seed' in rec and ('metrics' in rec or 'online' in rec):
            result.append(rec)
    return sorted(result, key=lambda x: int(x['seed']))


def _expected_seeds(plan):
    if isinstance(plan.get('seeds'), list):
        return sorted(int(x) for x in plan['seeds'])
    jobs = plan.get('jobs', [])
    seeds = [int(x['seed']) for x in jobs if isinstance(x, dict) and 'seed' in x]
    if seeds:
        return sorted(set(seeds))
    config = plan.get('configuration', plan.get('config', {}))
    if isinstance(config.get('seeds'), list):
        return sorted(int(x) for x in config['seeds'])
    return []


def _metric_rows(records):
    rows = []
    for rec in records:
        seed = int(rec['seed'])
        metrics = rec.get('metrics', [])
        if isinstance(metrics, dict):
            metrics = [dict(arm=a, metric=m, value=v) for a, ms in metrics.items() for m, v in ms.items()]
        for metric in metrics:
            value = metric.get('value')
            if _finite(value):
                rows.append({'seed': seed, 'arm': metric['arm'], 'metric': metric['metric'],
                             'group': metric.get('group', ''), 'value': float(value)})
    return rows


def _lookup(rows):
    out = {}
    for row in rows:
        key = (row['arm'], row['metric'], row['group'])
        values = out.setdefault(key, {})
        if row['seed'] in values:
            raise ValueError('Duplicate scientific metric: ' + str((row['seed'],) + key))
        values[row['seed']] = row['value']
    return out


def _value(index, arm, candidates):
    for metric in candidates:
        entry = index.get((arm, metric, ''))
        if entry is not None:
            return entry
    return None


def _fmt(x, digits=4, percent=False):
    if x is None or not _finite(x):
        return '—'
    return f'{100*x:.2f}%' if percent else f'{x:.{digits}f}'


def _contrast_text(row):
    if row is None:
        return 'unavailable'
    mean = row['mean']
    ci = '' if row['ci95_low'] is None else f" [{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}]"
    return f'{mean:+.5f}{ci} (n={row["n_seeds"]})'


def _strata(records):
    rows = []
    for rec in records:
        seed = int(rec['seed'])
        whole_n = rec.get('online', {}).get('whole', {}).get('READ-HYBRID', {}).get('n')
        for item in rec.get('stratum_effects', []):
            n = item.get('n', 0)
            delta = item.get('read_minus_super_loss_sum')
            if delta is None:
                delta = item.get('read_loss_sum', 0) - item.get('super_loss_sum', 0)
            stratum = item['stratum']
            shared = {'seed': seed, 'scope': item.get('scope', 'whole'), 'stratum': stratum, 'n_events': n,
                      'whole_byte_denominator': whole_n}
            contribution = item.get('read_minus_super_whole_bpb')
            if contribution is None and whole_n:
                contribution = delta / whole_n
            rows.append(dict(shared, contrast='READ-HYBRID minus SUPER', loss_sum_difference=delta,
                             bits_per_category_event=delta/n if n else None, bits_per_whole_byte=contribution))
            # Removing exact-trace authority changes only this stratum. Non-authority fallback is shared.
            change = -delta if stratum == 'exact_trace' else 0.0
            rows.append(dict(shared, contrast='SLOT-ONLY minus READ-HYBRID', loss_sum_difference=change,
                             bits_per_category_event=change/n if n else None,
                             bits_per_whole_byte=(-contribution if stratum == 'exact_trace' else 0.0) if contribution is not None else None))
    return rows



def _pooled_attribution(records, strata):
    """Use every completed history in the denominator, including zero-event seeds."""
    denominator = sum(rec.get('online', {}).get('whole', {}).get('READ-HYBRID', {}).get('n', 0)
                      for rec in records)
    groups = {}
    for row in strata:
        groups.setdefault((row['contrast'], row['scope'], row['stratum']), []).append(row)
    pooled = []
    for (contrast, scope, stratum), items in sorted(groups.items()):
        n = sum(x['n_events'] for x in items)
        loss = math.fsum(x['loss_sum_difference'] for x in items)
        pooled.append({'contrast': contrast, 'scope': scope, 'stratum': stratum,
                       'n_events': n, 'n_completed_seeds': len(records),
                       'n_seeds_with_events': sum(x['n_events'] > 0 for x in items),
                       'whole_byte_denominator': denominator,
                       'loss_sum_difference': loss,
                       'bits_per_category_event': loss / n if n else None,
                       'bits_per_whole_byte': loss / denominator if denominator else None})
    return pooled


def _reconcile_attribution(records, pooled):
    """Check category sums against the independently supplied whole-policy totals."""
    result = {}
    for label, arm, reference in (
            ('READ-HYBRID minus SUPER', 'READ-HYBRID', 'SUPER'),
            ('SLOT-ONLY minus READ-HYBRID', 'SLOT-ONLY', 'READ-HYBRID')):
        available = bool(records) and all(
            'loss_sum' in rec.get('online', {}).get('whole', {}).get(policy, {})
            for rec in records for policy in (arm, reference))
        if not available:
            result[label] = {'status': 'unavailable_without_complete_whole_loss_totals'}
            continue
        denominator = sum(rec['online']['whole']['READ-HYBRID']['n'] for rec in records)
        whole_difference = math.fsum(
            rec['online']['whole'][arm]['loss_sum'] - rec['online']['whole'][reference]['loss_sum']
            for rec in records) / denominator
        selected = [row for row in pooled if row['contrast'] == label and row['scope'] in ('target', 'nontarget')]
        contribution_sum = math.fsum(row['bits_per_whole_byte'] for row in selected)
        category_events = sum(row['n_events'] for row in selected)
        if category_events != denominator:
            raise ValueError('Attribution categories do not cover every whole-stream byte: ' + label)
        if not math.isclose(contribution_sum, whole_difference, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Pooled category contributions do not reconcile with whole-policy loss: ' + label)
        result[label] = {'status': 'passed', 'whole_byte_denominator': denominator,
                         'category_events': category_events,
                         'whole_policy_bpb_difference': whole_difference,
                         'category_contribution_sum': contribution_sum,
                         'residual_bpb': contribution_sum - whole_difference}
    return result


def _endpoint_summary(records):
    result = {}
    for rec in records:
        for arm, endpoints in rec.get('endpoints', {}).items():
            for name, row in endpoints.items():
                if isinstance(row, dict):
                    result.setdefault((arm, name), {})[int(rec['seed'])] = row
    return result


def _report(root, plan, records, per_seed, summaries, contrasts, strata, pooled_strata, completeness):
    index = {(x['arm'], x['metric'], x['group']): x['mean'] for x in summaries}
    endpoints = _endpoint_summary(records)
    def mean_metric(arm, candidates):
        return _value(index, arm, candidates)
    def endpoint(arm, name):
        values = [x.get('accuracy') for x in endpoints.get((arm,name), {}).values()]
        values = [x for x in values if _finite(x)]
        return statistics.mean(values) if values else mean_metric(arm, [name, name+'_accuracy'])
    def contrast(a,b,candidates):
        return next((x for x in contrasts if x['arm']==a and x['reference']==b and x['metric'] in candidates and not x['group']), None)
    status = ('All scheduled seed audits are complete.' if completeness['complete'] else
              'This is an interim report. Remaining seed audits have not yet been included; no final policy ranking is issued.')
    lines = ['# V42: fixed-history read-authority audit', '', status, '',
        f"Completed histories: **{len(records)}/{len(completeness['expected_seeds']) if completeness['expected_seeds'] else 'unspecified'}**. "
        f"Recorded seeds: {', '.join(str(x['seed']) for x in records) or 'none'}. "
        '**New training presentations: zero.**', '',
        'V42 evaluates five read policies on the same SLOTS writing histories saved by V41. The four existing V41 seeds are reused; this is a post hoc intervention on retrieval and is not four new independent replications. The policy masks are fixed before the new audit is run. No threshold is selected or gate relaxed from these outcomes.', '',
        'The original histories contain 15,344 byte presentations per seed, or 61,376 in total. These are reused observations, not newly presented training data. Reference preparation scans raw contexts and makes read-only terminal queries; it never calls the learner update. The standalone runner transforms embedded sufficient statistics and scalar probe outcomes, so it also performs zero learner updates and does not replay V41 training.', '',
        'The recoverable chat recommendation was to separate probationary exact-trace authority from corroborated slot authority, retaining the firewall between fallback reading and writing. The operational policy masks, paired contrasts, context checks and stopping rules are documented as refinements in the accompanying protocol; they are not presented as verbatim recovery of an unseen numerical specification.', '',
        '| Policy | Exact consolidated anchor | Exact probationary trace | Supported slot similarity |',
        '|---|---:|---:|---:|']
    for arm in POLICIES:
        masks=POLICY_MASKS[arm]
        lines.append(f"| {arm} | {'Yes' if 'exact_slot' in masks else 'No'} | {'Yes' if 'exact_trace' in masks else 'No'} | {'Yes' if 'supported_slot' in masks else 'No'} |")
    lines += ['', 'When a removed authority would have answered, the policy uses the saved SUPER prediction. The first four policies retain the saved READ-HYBRID nonauthoritative fallback, including its base fallback if the superposed output norm is effectively zero; SUPER always uses its own saved prediction. No read mask changes allocation, confirmation, demotion, output training or subsequent writing.', '',
        '## Prediction loss and retention', '',
        'The table shows equal-weight means over completed seeds. Loss is bits per observed byte; lower is better. Recall values are endpoint accuracies with their original finite denominators.', '',
        '| Policy | Whole bpb ↓ | Target bpb ↓ | Non-target bpb ↓ | E1 held-out | Old held-out | Old seen | Current seen |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in POLICIES:
        cells=[arm, _fmt(mean_metric(arm,['whole_bpb'])),_fmt(mean_metric(arm,['target_bpb'])),_fmt(mean_metric(arm,['nontarget_bpb']))]
        cells += [_fmt(endpoint(arm,x),percent=True) for x in ('E1_all','A_after_D_heldout','A_after_D_seen','after_D_current_seen')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'Primary paired differences use the same seed on both sides. Negative loss differences favor the first policy. The brackets are pointwise Student 95% intervals, conditional on the retained seed histories.', '',
        f"- SLOT-ONLY minus READ-HYBRID, whole loss: **{_contrast_text(contrast('SLOT-ONLY','READ-HYBRID',['whole_bpb']))}** bits/byte.",
        f"- SLOT-ONLY minus SUPER, whole loss: **{_contrast_text(contrast('SLOT-ONLY','SUPER',['whole_bpb']))}** bits/byte.", '',
        'PER_SEED.csv preserves all policy/metric seed values. PAIRED_CONTRASTS.csv preserves every matched seed difference, intervals and both inferential reference calculations. A lower overall loss does not imply a uniform improvement: old seen, current seen, old held-out and revision endpoints must be inspected separately.', '',
        '## Where the loss difference arises', '',
        'Attribution is evaluated on frozen predictions before the actual byte is observed. Category membership comes from the inherited retrieval path; target versus non-target labels are used only for reporting. A policy never receives evaluator labels. The table pools event totals across completed seeds. Its final column divides by all scored stream bytes, including histories with zero events in that category. Category contributions within a scope therefore add to that scope’s contribution to whole-stream loss.', '',
        '| Contrast | Scope | Authority category | Events | Δ bits/category event | Δ bits/whole byte |',
        '|---|---|---|---:|---:|---:|']
    for con in ('READ-HYBRID minus SUPER','SLOT-ONLY minus READ-HYBRID'):
        for row in pooled_strata:
            if row['contrast'] == con and row['scope'] in ('target', 'nontarget'):
                lines.append(f"| {con} | {row['scope']} | {row['stratum']} | {row['n_events']:,} | {_fmt(row['bits_per_category_event'],5)} | {_fmt(row['bits_per_whole_byte'],6)} |")
    if not strata:
        lines.append('| Pending | — | — | 0 | — | — |')
    lines += ['', 'The exact-trace and supported-similarity categories are disjoint for every scored query. Consequently READ-HYBRID − SLOT-ONLY − EXACT-ONLY + ANCHOR-ONLY is identically zero for loss and correctness at each event, up to arithmetic roundoff. This is an algebraic consequence of the masks. It supplies a bookkeeping check, not evidence that the mechanisms are biologically or statistically independent. It also does not describe how they would interact if their writing histories were allowed to change.', '',
        '## Context evidence and scope of the conclusion', '',
        'Historical held-out probes use rendering 2. The evaluator checks that their complete contexts were never among the preceding training contexts. Once that check passes, exact-trace removal cannot affect them: SLOT-ONLY and READ-HYBRID necessarily retain identical E1 and old held-out results. EXACT-ONLY and ANCHOR-ONLY likewise coincide there because both disable the remaining similarity route. Preservation of these endpoints is a structural implication of this fixture, not additional evidence of durable learning.', '',
        'Seen-context endpoints are evaluated only where an appropriate recorded historical prediction or read-only final-state probe is available. A terminal-state check does not validate an unrecorded earlier seen-context endpoint. No such historical inference is substituted. The result records retain endpoint counts, context-proof outcomes and final read-only audit provenance; their authority should be assessed separately.', '',
        'Every finalized seed record retains the supplied evaluator audits and provenance without rewriting them. ANALYSIS.json collects those records for inspection. Intermediate reports reflect completed records only; checkpoints can contain additional progress which has not yet qualified as a completed audit.', '',
        '## Interpretation and next decision', '']
    sr=contrast('SLOT-ONLY','READ-HYBRID',['whole_bpb'])
    ss=contrast('SLOT-ONLY','SUPER',['whole_bpb'])
    if not completeness['complete']:
        lines += ['Wait for the fixed roster to finish before drawing the aggregate conclusion. The current tables document available outcomes, not a stopping criterion.']
    elif sr and ss:
        verb='reduces' if sr['mean']<0 else 'increases' if sr['mean']>0 else 'leaves unchanged'
        lines += [f"On these fixed histories, removing exact-trace read authority {verb} average whole-stream loss by {abs(sr['mean']):.5f} bits/byte. "
                  f"SLOT-ONLY’s residual difference from SUPER is {ss['mean']:+.5f} bits/byte. "
                  'The category decomposition identifies which inherited reads account for that change. The seen-context results determine whether the improvement sacrifices exact recall that the held-out fixture cannot detect.']
        if ss['mean']>0:
            lines += ['', 'SLOT-ONLY still has a positive mean whole-stream loss gap to SUPER. Any case for retaining it therefore rests on the measured retention or recall tradeoff, rather than claiming best average prediction loss.']
        lines += ['', 'The appropriate carry-forward claim is conditional: this mask is a candidate read policy for the unchanged memory history. A future end-to-end learner would still need a prospective test on new streams with all learning enabled. V42 itself creates no new evidence about learning efficiency, capacity under further training, arbitrary natural byte streams or neuron reduction.']
    else:
        lines += ['The primary paired metrics are unavailable in the supplied completed records; no substantive ranking is inferred.']
    lines += ['', '## Statistical interpretation', '',
        'Seeds are the replication unit. Bytes, identities, repeated presentations and probe checkpoints are dependent within a seed and are not counted as independent observations. With all four existing seeds, Student intervals have three degrees of freedom. Their Gaussian paired-difference assumption cannot be diagnosed reliably from four points. The exact two-sided sign-reversal reference calculation enumerates all sixteen sign patterns and has minimum attainable p=0.125 for a nonzero four-seed contrast. A parametric Student p-value can be smaller under its assumptions; this does not evade the small independent sample size. Sign reversal additionally requires the relevant sign symmetry/exchangeability interpretation, rather than following automatically from paired data.', '',
        'All comparisons are exploratory and pointwise; no family-wide confidence claim or post hoc winner qualification is made. Deterministic policy equalities are documented as equalities. For zero sample variance and a nonzero paired mean, a Student p-value is left blank rather than reporting a misleading numerical zero. Category estimates are exact decompositions of these recorded losses, not independently replicated causal mediators.', '',
        '## Reproducibility files', '',
        'The single-file runner embeds the validated reference data; it requires no historical model files and does not regenerate V41 learning histories. It regenerates these reports from checksummed completed records and resumes interrupted evaluation. PER_SEED.csv, SUMMARY.csv, PAIRED_CONTRASTS.csv, ATTRIBUTION.csv and POOLED_ATTRIBUTION.csv contain numeric tables. ANALYSIS.json contains completion status, preserved seed audits and provenance. The accompanying protocol specifies the fixed roster and operational refinements; execution logs and verification records describe source matching, historical checks and read-only behavior.', '']
    Path(root,'ANALYSIS.md').write_text('\n'.join(lines))


def make_figure(root, summaries, contrasts):
    """Small static scientific figure; no result files or calculations are changed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index={(r['arm'],r['metric'],r['group']):r for r in summaries}
    if not all((a,'whole_bpb','') in index for a in POLICIES):
        return None
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    labels=[a.replace('-','-\n',1) for a in POLICIES]
    colors=['#677085','#087e8b','#bd6b32','#8a679b','#334155']
    means=[index[(a,'whole_bpb','')]['mean'] for a in POLICIES]
    axes[0].bar(range(len(POLICIES)),means,color=colors)
    axes[0].set_xticks(range(len(POLICIES)),labels,fontsize=8)
    axes[0].set_ylabel('Whole-stream bits per byte (lower is better)')
    axes[0].set_title('Same writing histories; different reads')
    for i,a in enumerate(POLICIES):
        row=index[(a,'whole_bpb','')]
        if row['ci95_low'] is not None:
            axes[0].errorbar(i,row['mean'],yerr=[[row['mean']-row['ci95_low']],[row['ci95_high']-row['mean']]],fmt='none',ecolor='#222222',capsize=3)
    chosen=[r for r in contrasts if r['metric']=='whole_bpb' and not r['group'] and r['importance']=='primary']
    for i,row in enumerate(chosen):
        vals=list(row['seed_differences'].values())
        offsets=[(j-(len(vals)-1)/2)*.045 for j in range(len(vals))]
        axes[1].scatter(vals,[i+x for x in offsets],color='#667085',s=28,zorder=3,label='Seed differences' if i==0 else None)
        axes[1].plot(row['mean'],i,'D',color='#087e8b',markersize=6)
        if row['ci95_low'] is not None:
            axes[1].plot([row['ci95_low'],row['ci95_high']],[i,i],color='#087e8b',linewidth=2)
    axes[1].axvline(0,color='#8a8a8a',linewidth=1)
    axes[1].set_yticks(range(len(chosen)),[f"{r['arm']}\nminus {r['reference']}" for r in chosen],fontsize=8)
    axes[1].set_xlabel('Paired difference in bits per byte')
    axes[1].set_title('Seed differences and Student 95% intervals')
    axes[1].legend(loc='best',fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('V42: post hoc retrieval audit; zero new training',fontsize=13)
    path=Path(root,'V42_results.png')
    fig.savefig(path,dpi=180)
    plt.close(fig)
    return str(path)


def generate(root, plan, records):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    records=_unwrap(records)
    seeds=[int(x['seed']) for x in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated completed seed in report input')
    expected=_expected_seeds(plan)
    if expected and set(seeds)-set(expected):
        raise ValueError('Report contains seeds outside the locked plan')
    completeness={'expected_seeds':expected,'completed_seeds':seeds,'missing_seeds':sorted(set(expected)-set(seeds)),
                  'complete':bool(expected) and set(seeds)==set(expected),'new_training_presentations':0}
    rows=_metric_rows(records)
    lookup=_lookup(rows)
    summaries=[]
    for (arm,metric,group),values in sorted(lookup.items()):
        summaries.append(dict(arm=arm,metric=metric,group=group,seed_values={str(k):v for k,v in sorted(values.items())},**_summary(list(values.values()))))
    contrasts=[]
    for arm,ref,importance in CONTRASTS:
        keys=sorted({(m,g) for a,m,g in lookup if a==arm} & {(m,g) for a,m,g in lookup if a==ref})
        for metric,group in keys:
            av,bv=lookup[(arm,metric,group)],lookup[(ref,metric,group)]
            values={seed:av[seed]-bv[seed] for seed in sorted(set(av)&set(bv))}
            contrasts.append(dict(arm=arm,reference=ref,importance=importance,metric=metric,group=group,**_paired(values)))
    strata=_strata(records)
    pooled_strata=_pooled_attribution(records,strata)
    attribution_reconciliation=_reconcile_attribution(records,pooled_strata)
    _csv(root/'PER_SEED.csv',rows,['seed','arm','metric','group','value'])
    _csv(root/'SUMMARY.csv',summaries,['arm','metric','group','n_seeds','mean','sd','ci95_low','ci95_high','df','seed_values'])
    _csv(root/'PAIRED_CONTRASTS.csv',contrasts,['arm','reference','importance','metric','group','n_seeds','mean','sd','ci95_low','ci95_high','df','student_t','student_p_two_sided','signflip_p_two_sided','zero_sample_variance','seed_differences'])
    _csv(root/'ATTRIBUTION.csv',strata,['seed','contrast','scope','stratum','n_events','whole_byte_denominator','loss_sum_difference','bits_per_category_event','bits_per_whole_byte'])
    _csv(root/'POOLED_ATTRIBUTION.csv',pooled_strata,['contrast','scope','stratum','n_events','n_completed_seeds','n_seeds_with_events','whole_byte_denominator','loss_sum_difference','bits_per_category_event','bits_per_whole_byte'])
    report={'schema_version':1,'experiment':'V42 fixed-history read-authority audit','completion':completeness,
            'policies':{a:sorted(v) for a,v in POLICY_MASKS.items()},'summary':summaries,'paired_contrasts':contrasts,
            'stratum_attribution':strata,'pooled_stratum_attribution':pooled_strata,
            'attribution_reconciliation':attribution_reconciliation,'seed_audits':{str(r['seed']):r.get('audits',{}) for r in records},
            'seed_provenance':{str(r['seed']):r.get('provenance',{}) for r in records},
            'seed_limitations':{str(r['seed']):r.get('limitations',[]) for r in records},
            'statistical_limits':['Four V41 seeds reused; no new independent replication.',
                                  'Pointwise exploratory paired Student intervals, df=3 when four seeds complete.',
                                  'Four-seed exact two-sided sign-reversal reference p cannot be below 0.125.',
                                  'The mask interaction is zero by algebra and does not demonstrate mechanism independence.']}
    _json(root/'ANALYSIS.json',report)
    _report(root,plan,records,rows,summaries,contrasts,strata,pooled_strata,completeness)
    if completeness['complete']:
        make_figure(root,summaries,contrasts)
    return report


def self_test():
    """Deterministic reporting tests on synthetic data, never scientific outcomes."""
    import tempfile
    tests={}
    p=_paired({1:1.,2:2.,3:3.,4:4.})
    assert p['df']==3 and p['signflip_p_two_sided']==.125
    assert p['ci95_low']<p['mean']<p['ci95_high']
    tests['four_seed_df_and_exact_reference']=True
    q=_paired({1:0.,2:0.,3:0.,4:0.})
    assert q['mean']==0 and q['signflip_p_two_sided']==1 and q['student_p_two_sided']==1
    tests['deterministic_equality_is_not_significance']=True
    with tempfile.TemporaryDirectory() as tmp:
        empty=generate(tmp,{'seeds':[1,2,3,4]},[])
        assert not empty['completion']['complete'] and 'interim' in Path(tmp,'ANALYSIS.md').read_text()
        sample={'seed':1,'metrics':[{'arm':a,'metric':'whole_bpb','value':float(i+1)} for i,a in enumerate(POLICIES)],
                'online':{'whole':{'READ-HYBRID':{'n':10}}},
                'stratum_effects':[{'scope':'nontarget','stratum':'exact_trace','n':2,'read_minus_super_loss_sum':4,'read_minus_super_whole_bpb':.4}]}
        generated=generate(tmp,{'seeds':[1,2,3,4]},[sample])
        assert generated['stratum_attribution'][1]['bits_per_whole_byte']==-.4
        assert generated['paired_contrasts'][0]['n_seeds']==1
        assert generated['paired_contrasts'][0]['ci95_low'] is None
        saved=json.loads(Path(tmp,'ANALYSIS.json').read_text())
        assert saved['completion']['missing_seeds']==[2,3,4]
    # One category is absent from one seed. Its numerator remains zero there,
    # but that seed's stream bytes must remain in every pooled denominator.
    missing = []
    for seed, category, delta in ((1, 'exact_trace', 4.), (2, 'supported_slot', 6.)):
        read_loss = 20. + delta
        slot_loss = 20. if category == 'exact_trace' else read_loss
        missing.append({'seed': seed, 'metrics': [], 'online': {'whole': {
            'READ-HYBRID': {'n': 10, 'loss_sum': read_loss},
            'SLOT-ONLY': {'n': 10, 'loss_sum': slot_loss},
            'SUPER': {'n': 10, 'loss_sum': 20.}}},
            'stratum_effects': [
                {'scope': 'target', 'stratum': category, 'n': 2,
                 'read_minus_super_loss_sum': delta, 'read_minus_super_whole_bpb': delta / 10},
                {'scope': 'nontarget', 'stratum': 'no_candidate', 'n': 8,
                 'read_minus_super_loss_sum': 0., 'read_minus_super_whole_bpb': 0.}]})
    pooled = _pooled_attribution(missing, _strata(missing))
    supported = next(row for row in pooled if row['stratum'] == 'supported_slot'
                     and row['contrast'] == 'READ-HYBRID minus SUPER')
    assert supported['n_seeds_with_events'] == 1 and supported['n_completed_seeds'] == 2
    assert supported['whole_byte_denominator'] == 20
    assert supported['bits_per_whole_byte'] == .3
    assert supported['bits_per_category_event'] == 3.
    reconciliation = _reconcile_attribution(missing, pooled)
    assert reconciliation['READ-HYBRID minus SUPER']['category_contribution_sum'] == .5
    assert reconciliation['SLOT-ONLY minus READ-HYBRID']['category_contribution_sum'] == -.2
    assert all(row['status'] == 'passed' for row in reconciliation.values())
    with tempfile.TemporaryDirectory() as tmp:
        generated = generate(tmp, {'seeds': [1,2,3,4]}, missing)
        assert generated['attribution_reconciliation'] == reconciliation
        table = Path(tmp, 'ANALYSIS.md').read_text()
        assert '| supported_slot | 2 | 3.00000 | 0.300000 |' in table
    tests['missing_category_seed_remains_in_pooled_denominator'] = True
    tests['pooled_categories_include_zero_fallback_and_reconcile_whole_loss'] = True
    tests['empty_and_partial_reports']=True
    tests['stratum_sign_and_denominator']=True
    return tests
