#!/usr/bin/env python3
"""Deterministically (re)generate the MRK-1 regression corpus and EXPECTED.json.

    python3 -I -B tools/minifly_check/corpus/_build/build_corpus.py          # write
    python3 -I -B tools/minifly_check/corpus/_build/build_corpus.py --check  # rebuild in a temp dir and
                                                                           # compare byte for byte

Writes corpus/_build/base/ (the hand-designed clean base; EXPECTED_VALUES.md there is hand-written and
kept), one directory per SPEC 12.2 row under corpus/ (FIXTURE.json, timeline.json, repo/, history/) and
tools/minifly_check/EXPECTED.json. Standard library only. It never imports checker modules and never
runs a learner; selftest never runs this builder.
"""
import argparse
import filecmp
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import base_artifacts as BA  # noqa: E402
import base_files as BF  # noqa: E402
import fixtures_g as G  # noqa: E402
import table  # noqa: E402
from base_timeline import build_base  # noqa: E402
from mrk_util import dir_size, dump_json, write_fixture  # noqa: E402

CORPUS = os.path.dirname(HERE)
KIT = os.path.dirname(CORPUS)
SIZE_WARN = 400 * 1024


def bundles_for(cmd, tier, sealed):
    if cmd[0] == 'gate':
        stage = cmd[cmd.index('--stage') + 1]
        return BA.bundle_ids(stage, sealed=sealed, tier=tier)
    if cmd[0] in ('canary', 'governance'):
        return BA.bundle_ids(cmd[0], sealed=sealed, tier=tier)
    return []


def expected_entry(e, tl):
    cmds = e['commands'](tl) if callable(e['commands']) else e['commands']
    mnf = list(e['must_not_fire'])
    if e['kind'] == 'clean':
        for c in cmds:
            for i in bundles_for(c, e['tier'], e['sealed']):
                if i not in mnf:
                    mnf.append(i)
    return {'commands': [list(c) for c in cmds], 'must_fire': list(e['must_fire']),
            'must_not_fire': mnf, 'must_warn': list(e['must_warn']),
            'expected_exit': list(e['expected_exit']), 'expect': dict(e['expect']),
            'source_incident': e['source_incident'], 'fm_ids': list(e['fm_ids'])}


def fixture_meta(e, delivery):
    return {'id': e['id'], 'title': e['title'], 'fm_ids': list(e['fm_ids']),
            'source_incident': e['source_incident'], 'kind': e['kind'], 'built_from': e['built_from'],
            'mutation': e['mutation'], 'expectation': e['expectation'], 'delivery': delivery,
            'generator': 'tools/minifly_check/corpus/_build/build_corpus.py', 'format': 'TimelineGit'}


def build_all(corpus_dir, expected_path, verbose=True):
    tl, _b = build_base()
    write_fixture(os.path.join(corpus_dir, '_build', 'base'), tl, keep=('EXPECTED_VALUES.md',))
    expected = {}
    sizes = {}
    for e in table.T:
        tl, _b = e['build']()
        delivery = {'C00f': G.GD, 'G01': G.GD, 'G02': G.GD, 'G03': G.GD, 'G05': G.GD, 'S23': G.GD,
                    'S24': G.GD, 'S25': G.GD, 'K01': None}.get(e['id'], BF.D)
        dest = os.path.join(corpus_dir, e['id'])
        write_fixture(dest, tl, fixture_meta(e, delivery))
        expected[e['id']] = expected_entry(e, tl)
        sizes[e['id']] = dir_size(dest)
    with open(expected_path, 'wb') as fh:
        fh.write(dump_json(expected))
    if verbose:
        for k in sorted(sizes):
            flag = '  (over %d KiB)' % (SIZE_WARN // 1024) if sizes[k] > SIZE_WARN else ''
            print('%-5s %7.1f KiB%s' % (k, sizes[k] / 1024.0, flag))
        print('fixtures: %d, total %.1f KiB' % (len(sizes), sum(sizes.values()) / 1024.0))
    return expected


def compare_trees(a, b):
    diffs = []
    for root, dirs, files in os.walk(a):
        rel = os.path.relpath(root, a)
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.pyc'):
                continue
            pa, pb = os.path.join(root, f), os.path.join(b, rel, f)
            if not os.path.exists(pb) or not filecmp.cmp(pa, pb, shallow=False):
                diffs.append(os.path.join(rel, f))
    return diffs


def validate(corpus_dir, expected_path):
    """Every JSON/JSONL(.gz) file parses; every timeline is linear with history for superseded content."""
    import gzip
    import json
    problems = []
    fixture_dirs = [os.path.join(corpus_dir, e['id']) for e in table.T]
    fixture_dirs.append(os.path.join(corpus_dir, '_build', 'base'))
    for fd in fixture_dirs:
        for root, _dirs, files in os.walk(fd):
            for f in files:
                p = os.path.join(root, f)
                try:
                    if f.endswith('.json'):
                        with open(p, 'rb') as fh:
                            json.loads(fh.read().decode('utf-8'))
                    elif f.endswith('.jsonl') or f.endswith('.jsonl.gz'):
                        opener = gzip.open if f.endswith('.gz') else open
                        with opener(p, 'rb') as fh:
                            for line in fh.read().decode('utf-8').splitlines():
                                if line.strip():
                                    json.loads(line)
                except ValueError as exc:
                    problems.append('%s: %s' % (os.path.relpath(p, corpus_dir), exc))
        with open(os.path.join(fd, 'timeline.json'), 'rb') as fh:
            tl = json.loads(fh.read().decode('utf-8'))
        shas = [c['sha'] for c in tl['commits']]
        if len(set(shas)) != len(shas) or tl['head'] != shas[-1]:
            problems.append('%s: bad commit list' % fd)
        last = {}
        for c in tl['commits']:
            for p in c['paths']:
                last[p] = c['sha']
        for c in tl['commits']:
            for p in c['paths']:
                in_hist = os.path.exists(os.path.join(fd, 'history', c['sha'], p))
                in_repo = os.path.exists(os.path.join(fd, 'repo', p))
                if last[p] != c['sha'] and not in_hist and in_repo:
                    # superseded content must be in history unless the commit deleted the path
                    problems.append('%s: %s@%s has no history copy' % (os.path.basename(fd), p, c['sha'][:8]))
    with open(expected_path, 'rb') as fh:
        exp = json.loads(fh.read().decode('utf-8'))
    fm = set()
    for k, v in exp.items():
        fm.update(v['fm_ids'])
        if len(v['commands']) != len(v['expected_exit']):
            problems.append('EXPECTED %s: commands/expected_exit length mismatch' % k)
    missing = sorted(set('FM-%02d' % i for i in range(1, 31)) - fm)
    if missing:
        problems.append('FM classes without a fixture: %s' % ', '.join(missing))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='rebuild into a temp dir and compare')
    args = ap.parse_args(argv)
    if not args.check:
        build_all(CORPUS, os.path.join(KIT, 'EXPECTED.json'))
        return 0
    tmp = tempfile.mkdtemp(prefix='mrk1_corpus_check_')
    try:
        tcorpus = os.path.join(tmp, 'corpus')
        os.makedirs(os.path.join(tcorpus, '_build', 'base'))
        shutil.copy(os.path.join(CORPUS, '_build', 'base', 'EXPECTED_VALUES.md'),
                    os.path.join(tcorpus, '_build', 'base', 'EXPECTED_VALUES.md'))
        build_all(tcorpus, os.path.join(tmp, 'EXPECTED.json'), verbose=False)
        diffs = []
        for e in table.T + [{'id': os.path.join('_build', 'base')}]:
            d = compare_trees(os.path.join(tcorpus, e['id']), os.path.join(CORPUS, e['id']))
            d += ['(missing) ' + x for x in compare_trees(os.path.join(CORPUS, e['id']),
                                                          os.path.join(tcorpus, e['id']))]
            diffs += [os.path.join(e['id'], x) for x in d]
        if not filecmp.cmp(os.path.join(tmp, 'EXPECTED.json'), os.path.join(KIT, 'EXPECTED.json'),
                           shallow=False):
            diffs.append('EXPECTED.json')
        if diffs:
            print('DIFFERS (%d files):' % len(diffs))
            for d in diffs[:50]:
                print('  ' + d)
            return 1
        print('byte-identical: %d fixtures, base and EXPECTED.json' % len(table.T))
        problems = validate(CORPUS, os.path.join(KIT, 'EXPECTED.json'))
        for p in problems[:50]:
            print('INVALID ' + p)
        if problems:
            return 1
        print('valid: every JSON/JSONL(.gz) file parses; timelines consistent; FM-01..FM-30 covered')
        return 0
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    sys.exit(main())
