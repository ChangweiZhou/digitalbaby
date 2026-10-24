"""Named timeline-level mutation operators used by the fixture table (fixtures_c.py)."""
import copy
import json

import base_files as BF
import refcalc as RC
import scores_doc as SD
from mrk_util import dump_json, sha256_bytes

P = BF.D + '/'


def truncate(tl, label):
    """Keep commits up to and including `label` (a lock-stage scenario needs nothing later)."""
    i = tl.index(label)
    tl.commits = tl.commits[:i + 1]
    return tl


def merge_into(tl, keep, absorb):
    """Fold commit `absorb` into commit `keep` (one commit then carries both change sets)."""
    k, a = tl.get(keep), tl.get(absorb)
    k['changes'].update(a['changes'])
    tl.commits.remove(a)
    return tl


def move_after(tl, label, after, utc):
    c = tl.get(label)
    tl.commits.remove(c)
    tl.commits.insert(tl.index(after) + 1, c)
    c['utc'] = utc
    return tl


def make_untracked(tl, path):
    """Drop `path` from every commit; keep its final content in the worktree as untracked."""
    final = tl.final_tree().get(path)
    for c in tl.commits:
        c['changes'].pop(path, None)
    tl.worktree[path] = final
    tl.dirty.append(path)
    return tl


def insert_after(tl, after, label, utc, who, changes):
    meta = BF.CODEX if who == 'codex' else BF.CLAUDE
    c = {'label': label, 'utc': utc, 'author': meta['author'], 'trailers': copy.deepcopy(meta['trailers']),
         'branches': [BF.CODEX_BRANCH], 'changes': dict(changes)}
    tl.commits.insert(tl.index(after) + 1, c)
    return tl


def fix_branches(tl):
    idx = [i for i, c in enumerate(tl.commits)
           if c['trailers'].get('Acting-Agent', [''])[0].startswith('claude')]
    last = max(idx) if idx else -1
    for i, c in enumerate(tl.commits):
        c['branches'] = [BF.CODEX_BRANCH] + ([BF.CLAUDE_BRANCH] if i <= last else [])
    return tl


def claim_doc(b, rows=None, **over):
    """A CLAIM.json as `MFC claim` would write it for the build `b` (fields per SCHEMAS 15), using the
    reference numbers; fixtures then edit fields to reproduce an incident."""
    rows = rows if rows is not None else getattr(b, 'official_rows', None)
    v = SD.verdict(rows)
    pw = v['per_world']
    n = len(pw)
    best = sum(x['best'] for x in pw) / n
    names = sorted(set(nm for x in pw for nm in x['best_names']))
    doc = {
        'schema': 'minifly.claim.v1', 'delivery': BF.D, 'protocol_id': b.lock['protocol_id'],
        'level_attempted': 'E3', 'level_earned': 'E3' if v['verdict'] == 'PASS' else 'NONE',
        'reuse_specificity': 'not_tested', 'primary_gate': v['verdict'], 'dev_shadow_verdict': None,
        'invalid_reason': None, 'stop_reason': None, 'gate_type': 'objective', 'labels': [],
        'worlds_role': 'MAIN', 'n_worlds': n,
        'objective_metric': {
            'held_exact_own_eos': round(v['mean_cand'], 6),
            'held_by_stratum': {}, 'determined_fraction': 1.0,
            'old_end': round(sum(x['old_end'] for x in pw) / n, 6),
            'old_final': round(sum(x['old_final'] for x in pw) / n, 6),
            'old_drop': round(sum(x['old_drop'] for x in pw) / n, 6), 'new_final': 1.0,
            'best_shortcut': {'name': names[0], 'value': round(best, 6)},
            'best_other_arm': {'name': 'NPT', 'value': round(sum(x['npt'] for x in pw) / n, 6)},
            'headroom': round(v['mean_cand'] - best, 6), 'G': round(v['mean_g'], 6),
            'G_per_sample': round(v['mean_g'] / 0.15, 6), 'G_per_cpu_h': round(v['mean_g'] / 0.36, 6)},
        'designs_tried_k': getattr(b, 'designs_k', 1), 'amendments': [], 'coprimary_results': [],
        'posthoc_files': [], 'lineage_parent': None, 'family_id': 'FAM-' + sha256_bytes(BF.D.encode())[:8],
        'native_path_only': True,
        'compute': {'records_total_incl_failed': 9600, 'cpu_h_total_incl_failed': 0.36,
                    'family_cpu_h': 0.36},
        'provenance': {'interrupted': False, 'resume_attempts': [], 'resumed_units': [], 'replay_units': []},
        'verdict_source': {'decide_version': '1.0.0', 'verdict_sha256': None,
                           'endpoint_lock_sha256': b.lock_sha, 'source_lock_sha256': b.sl['sha']},
        'predictions_scored': [
            {'id': 'P1', 'observed': round(v['mean_cand'], 6), 'hit': True, 'interval_score': 0.35,
             'brier': 0.09},
            {'id': 'P2', 'observed': round(sum(x['old_drop'] for x in pw) / n, 6), 'hit': True,
             'interval_score': 0.08, 'brier': 0.0225},
            {'id': 'P3', 'observed': 0.5, 'hit': True, 'interval_score': 0.2, 'brier': 0.01}],
        'findings_proposed': [], 'findings_none_reason': 'Corpus fixture: nothing durable to record.',
        'review': {'path': 'review/REVIEW.json', 'decision': 'ADMIT'}, 'review_debt_id': None,
        'authorization': {'path': 'AUTHORIZATION.json',
                          'recommendation_shown': 'A, because it opens held E3 at low cost.'},
        'confirm_attempt': None, 'notes': '', 'checker_sha256': None,
        'headline': getattr(b, 'headline', None) or SD.headline(rows),
    }
    for k, val in over.items():
        doc[k] = val
    return doc


def edit_json(tl, label, path, fn):
    """Apply fn(obj) to the JSON file `path` as written by commit `label` (in place)."""
    c = tl.get(label)
    obj = json.loads(c['changes'][path].decode('utf-8'))
    res = fn(obj)
    c['changes'][path] = dump_json(obj if res is None else res)
    return tl


__all__ = ['RC']
