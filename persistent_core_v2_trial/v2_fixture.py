"""Reuse the frozen lifetime and wrapped relation tasks with fresh world seeds."""
import hashlib
import json
import bootstrap
import compact_fixture as lifetime
from centered_core import bootstrap as native_bootstrap
import importlib.util

path = bootstrap.PARENT / 'vendor/package/REFERENCE_SOURCE/MINIFLY_THREE_MECHANISM_ROUND_20260928/fixture.py'
spec = importlib.util.spec_from_file_location('_frozen_relation_fixture', path)
relations = importlib.util.module_from_spec(spec); spec.loader.exec_module(relations)
DT, RECORD_SECONDS, DAY = lifetime.DT, lifetime.RECORD_SECONDS, lifetime.DAY
SCIENCE_WORLDS = tuple(range(501001, 501097))
DEVELOPMENT_WORLDS = (500101, 500102, 500103)
ASSAYS = ('lifetime', 'reuse')


def digest(doc):
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_world(world, assay):
    if assay == 'lifetime':
        doc = lifetime.make_world(world)
        doc.pop('sha256'); doc['assay'] = assay
        doc['branches'] = ['W', 'N_old', 'N_new', 'N_revision']
        doc['boundaries'] = {'384': ['old_end', 'old_day'], '768': ['new_end', 'new_day'],
                             '864': ['revision_end', 'final']}
    elif assay == 'reuse':
        base = relations.make_world(world)
        old, new = base['old_relation'], base['new_relation']
        sets = {k: [{'cue_hex': r['cue_hex'], 'outcome': 48 + r['label']} for r in rs]
                for k, rs in [('old', old['taught']), ('heldout', old['heldout']), ('new', new['taught'])]}
        events = []
        for stage in ('old', 'new'):
            order = relations._schedule(world, stage + '-relation', len(sets[stage]))
            for item in order:
                i = len(events)
                events.append(dict(sets[stage][item], stage=stage, item=item, index=i,
                                   at=i * RECORD_SECONDS + (DAY if stage == 'new' else 0.)))
        doc = {'world': world, 'assay': assay, 'sets': sets, 'events': events, 'branches': ['W', 'N_old'],
               'boundaries': {'144': ['old_end', 'old_day'], '216': ['new_end', 'final']},
               'clocks': {'old_end': 144 * RECORD_SECONDS, 'old_day': 144 * RECORD_SECONDS + DAY,
                          'new_end': 216 * RECORD_SECONDS + DAY, 'final': 216 * RECORD_SECONDS + 2 * DAY},
               'source_fixture_digest': base['digest']}
        taught = {r['cue_hex'] for r in sets['old'] + sets['new']}
        assert not taught & {r['cue_hex'] for r in sets['heldout']}
        assert len(sets['old']) == 12 and len(sets['heldout']) == 6 and len(sets['new']) == 6
    else: raise ValueError(assay)
    for group in doc['sets'].values():
        for row in group:
            cue = bytes.fromhex(row['cue_hex'])
            assert len(cue) == 12 and len(cue.replace(b' ', b'')) >= 4
    doc['sha256'] = digest(doc)
    return doc


def permitted(branch, stage):
    table = {'W': {'old': True, 'new': True, 'revision': True},
             'N_old': {'old': False, 'new': True, 'revision': True},
             'N_new': {'old': True, 'new': False, 'revision': True},
             'N_revision': {'old': True, 'new': True, 'revision': False}}
    return table[branch][stage]
