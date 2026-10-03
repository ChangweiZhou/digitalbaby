"""Fixed exact primary family; all-cue/causal summaries are labelled descriptive."""
import json
import math
import statistics
from pathlib import Path
from scipy.stats import beta, t
from compact_bridge import ROOT, ARMS
from compact_fixture import SCIENCE_WORLDS, make_world
from compact_audit import audit_receipt, read, paired_private_equal
from compact_integrity import verify
from compact_storage import atomic_json


def cp_upper(k, n, alpha):
    return 1. if k == n else float(beta.ppf(1. - alpha, k + 1, n - k))


def cp_lower(k, n, alpha):
    return 0. if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def descriptive_ci(values):
    mean = statistics.mean(values)
    if len(values) < 2 or statistics.variance(values) == 0.:
        return {'mean': mean, 'ci': None, 'note': 'zero variance; no inferential interval from this descriptive reducer'}
    half = float(t.ppf(.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    return {'mean': mean, 'ci': [mean - half, mean + half], 'note': 'descriptive paired Student-t, not multiplicity adjusted'}


def score(bdoc, world, stage):
    rows = bdoc['probes'][-1]['rows']
    wanted = world['intact_old_items'] if stage == 'old' else list(range(32 if stage == 'new' else 8))
    return statistics.mean(r['correct'] for r in rows if r['stage'] == stage and r['item'] in wanted)


def primary(bdoc, world, stage):
    return next(r['correct'] for r in bdoc['probes'][-1]['rows']
                if r['stage'] == stage and r['item'] == world['primary_items'][stage])


def main():
    identity = verify(); stages = ('old', 'new', 'revision')
    samples = {a: {s: [] for s in stages} for a in ARMS}
    pri = {a: {s: [] for s in stages} for a in ARMS}
    causal = {a: {s: [] for s in stages} for a in ARMS}
    resources = {a: [] for a in ARMS}; roster = []
    for world_id in SCIENCE_WORLDS:
        world = make_world(world_id); docs = []
        for arm in ARMS:
            path = ROOT / 'results/science/receipts' / f'{world_id}_{arm}.json.gz'
            doc = read(path); audit_receipt(doc, identity); docs.append(doc); roster.append(path.name)
            resources[arm].append({'worker_s': doc['worker_s'], 'peak_rss_bytes': doc['peak_rss_bytes']})
            for stage in stages:
                w = doc['branches']['W']; n = doc['branches']['N_' + stage]
                samples[arm][stage].append(score(w, world, stage))
                pri[arm][stage].append(primary(w, world, stage))
                causal[arm][stage].append(score(w, world, stage) - score(n, world, stage))
        paired_private_equal(*docs)
    actual = sorted(p.name for p in (ROOT / 'results/science/receipts').glob('*.json.gz'))
    if actual != sorted(roster): raise ValueError('unexpected science receipt roster')
    n = len(SCIENCE_WORLDS); alpha = .05 / 6
    gates = {}
    for stage in stages:
        baseline, candidate = pri[ARMS[0]][stage], pri[ARMS[1]][stage]
        losses = sum(b == 1 and c == 0 for b, c in zip(baseline, candidate))
        hits = sum(candidate)
        upper = cp_upper(losses, n, alpha); lower = cp_lower(hits, n, alpha)
        gates[stage] = {'paired_loss_worlds': losses, 'selected_cue_correct': hits, 'n': n,
                        'loss_probability_upper': upper, 'candidate_accuracy_lower': lower,
                        'noninferiority_pass': upper < .05, 'absolute_accuracy_pass': lower > .90,
                        'mean_accuracy_difference_lower_bound': -upper}
    summary = {'identity': identity, 'evidence': 'E1 taught association acquisition/retention/revision',
               'worlds': n, 'world_arm_receipts': len(roster), 'logical_lives': len(roster) * 4,
               'primary_method': 'six one-sided Clopper-Pearson bounds, Bonferroni family 95%; one frozen random cue per world/stratum',
               'gates': gates, 'adopt_compact': all(g['noninferiority_pass'] and g['absolute_accuracy_pass'] for g in gates.values()),
               'all_cue_accuracy_descriptive': {a: {s: descriptive_ci(samples[a][s]) for s in stages} for a in ARMS},
               'causal_effect_descriptive': {a: {s: descriptive_ci(causal[a][s]) for s in stages} for a in ARMS},
               'resources': {a: {'worker_s_total': sum(x['worker_s'] for x in resources[a]),
                                  'max_peak_rss_bytes': max(x['peak_rss_bytes'] for x in resources[a])} for a in ARMS},
               'private_bank_parity_all_worlds': True}
    atomic_json(ROOT / 'results/SUMMARY.json', summary)
    lines = ['# R_center physical shared-bank deletion', '',
             '**Decision: ' + ('ADOPT CONTENT_4 within this frozen E1 scope.' if summary['adopt_compact'] else 'DO NOT ADOPT: one or more prespecified reliability gates failed.') + '**', '',
             f'Complete: {n} paired worlds, {len(roster)} world-arm receipts, {len(roster)*4} lives. Three design/audit/trial/revision cycles preceded science.', '',
             '| Final taught stratum | Eight-store accuracy | Four-store accuracy | Four-store minus no-stage-write |',
             '|---|---:|---:|---:|']
    for s in stages:
        lines.append(f"| {s} | {100*statistics.mean(samples[ARMS[0]][s]):.2f}% | {100*statistics.mean(samples[ARMS[1]][s]):.2f}% | {100*statistics.mean(causal[ARMS[1]][s]):+.2f} pp |")
    lines += ['', 'The table uses all scored cues and is descriptive. Primary decisions use the six exact bounds in SUMMARY.json; all three noninferiority and all three absolute-accuracy gates must pass.', '',
              'CONTENT_4 physically has four independently born stores and no shared bank. The eight-store parent was not modified. All saved private states and values match between arms on every paired branch/checkpoint.', '',
              'Claims are E1 only. Every scored association was taught; an exact-key table can solve it. No E3, autonomous output, teacher-free learning, or long-context claim is made. State count falls from eight to four; measured runtime/RSS do not necessarily fall by half.', '',
              '| Gate | Loss upper bound | Accuracy lower bound | NI / absolute pass |', '|---|---:|---:|---|']
    for s,g in gates.items(): lines.append(f"| {s} | {100*g['loss_probability_upper']:.2f}% | {100*g['candidate_accuracy_lower']:.2f}% | {g['noninferiority_pass']} / {g['absolute_accuracy_pass']} |")
    (ROOT / 'results/REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'complete': True, 'adopt_compact': summary['adopt_compact']}))


if __name__ == '__main__': main()
