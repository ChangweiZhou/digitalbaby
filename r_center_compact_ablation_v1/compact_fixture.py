"""Evaluator-owned fixture. No world, stage, key map or label enters a core."""
import hashlib
import json
import random

DT = 30. / 14
RECORD_SECONDS = 165.
DAY = 86400.
BRANCHES = ('W', 'N_old', 'N_new', 'N_revision')
STAGES = ('old', 'new', 'revision')
SCIENCE_WORLDS = tuple(range(491001, 491097))
DEVELOPMENT_WORLDS = (490101, 490102, 490103)


def rng(world, key):
    return random.Random(int.from_bytes(hashlib.sha256(f'COMPACT-v1|{world}|{key}'.encode()).digest()[:16], 'big'))


def permitted(branch, stage):
    if branch not in BRANCHES or stage not in STAGES: raise ValueError('unknown branch/stage')
    return branch == 'W' or branch != {'old': 'N_old', 'new': 'N_new', 'revision': 'N_revision'}[stage]


def make_world(world):
    cues = [b' ' * 8 + bytes((a, 43, b, 61)) for a in b'01234567' for b in b'01234567']
    rng(world, 'split').shuffle(cues)
    sets = {}
    for stage, part in [('old', cues[:32]), ('new', cues[32:])]:
        labels = list(b'0123') * 8
        rng(world, stage + 'labels').shuffle(labels)
        sets[stage] = [{'cue_hex': c.hex(), 'outcome': y} for c, y in zip(part, labels)]
    revised = []
    for y in b'0123':
        ids = [i for i, c in enumerate(sets['old']) if c['outcome'] == y]
        rng(world, f'revision-{y}').shuffle(ids)
        revised.extend(ids[:2])
    revised.sort()
    sets['revision'] = [dict(sets['old'][i], outcome=48 + (sets['old'][i]['outcome'] - 48 + 1) % 4,
                             old_item=i) for i in revised]
    intact = [i for i in range(32) if i not in revised]
    primary = {'old': rng(world, 'primary-old').choice(intact),
               'new': rng(world, 'primary-new').randrange(32),
               'revision': rng(world, 'primary-revision').randrange(8)}
    events = []
    for stage in STAGES:
        for bout in range(12):
            order = list(range(len(sets[stage])))
            rng(world, f'{stage}-bout-{bout}').shuffle(order)
            for item in order:
                index = len(events)
                at = index * RECORD_SECONDS + STAGES.index(stage) * DAY
                events.append(dict(sets[stage][item], index=index, stage=stage, item=item, at=at))
    clocks = {'old_end': 384 * RECORD_SECONDS, 'old_day': 384 * RECORD_SECONDS + DAY,
              'new_end': 768 * RECORD_SECONDS + DAY, 'new_day': 768 * RECORD_SECONDS + 2 * DAY,
              'revision_end': 864 * RECORD_SECONDS + 2 * DAY, 'final': 864 * RECORD_SECONDS + 3 * DAY}
    doc = {'world': world, 'sets': sets, 'events': events, 'revised_old_items': revised,
           'intact_old_items': intact, 'primary_items': primary, 'clocks': clocks}
    keys = [bytes.fromhex(c['cue_hex']).replace(b' ', b'')[-4:] for c in sets['old'] + sets['new']]
    assert len(keys) == len(set(keys)) == 64 and all(len(k) == 4 for k in keys)
    assert all(len(bytes.fromhex(c['cue_hex'])) == 12 for c in sets['old'] + sets['new'])
    assert len(events) == 864 and len(intact) == 24 and len(revised) == 8
    doc['sha256'] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc
