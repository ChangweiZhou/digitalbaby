"""Deterministic helpers for the MRK-1 regression-corpus builder (stdlib only, Python 3.9+).

Nothing here imports or runs checker code. The builder writes fixtures in the TimelineGit format of
SCHEMAS.md Section 19.5: <fixture>/timeline.json, <fixture>/repo/ (final tree) and
<fixture>/history/<sha>/<path> (content at a commit whose content differs from repo/).
"""
import copy
import hashlib
import json
import os
import shutil
import struct
import zlib

EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj):
    """sort_keys, separators (',', ':'), UTF-8, no NaN (SPEC 7.3)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def sha256_json(obj):
    return sha256_bytes(canonical_json(obj))


def dump_json(obj):
    """Pretty, stable JSON file bytes (indent 2, sorted keys, trailing newline)."""
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + '\n').encode('utf-8')


def dump_compact(obj):
    """Compact JSON file bytes (sorted keys, trailing newline) for large machine-written files."""
    return canonical_json(obj) + b'\n'


def dump_jsonl(rows):
    out = []
    for r in rows:
        out.append(json.dumps(r, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                              allow_nan=False))
    return ('\n'.join(out) + ('\n' if out else '')).encode('utf-8')


def gzip_bytes(data):
    """RFC 1952 gzip with a fixed header (mtime 0, no name, OS 255) so output does not depend on
    the Python version; the deflate stream comes from zlib level 9, raw mode."""
    header = b'\x1f\x8b\x08\x00' + struct.pack('<I', 0) + b'\x02\xff'
    comp = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    body = comp.compress(data) + comp.flush()
    trailer = struct.pack('<II', zlib.crc32(data) & 0xffffffff, len(data) & 0xffffffff)
    return header + body + trailer


def hexs(s):
    return s.encode('ascii').hex()


def commit_sha(label):
    """Commit shas are derived from the commit label only, so the base and every fixture copied from
    it share the same shas for the same commits."""
    return hashlib.sha1(('mrk1-corpus|' + label).encode('utf-8')).hexdigest()


class Timeline(object):
    """A linear commit history plus the files each commit writes.

    commits: list of dicts {label, utc, author, trailers {k: [v]}, branches [str],
             changes {repo path: bytes | None (deleted)}}.
    dirty:   repo paths reported as modified/untracked in the worktree (timeline.json 'dirty').
    worktree: {repo path: bytes | None} applied on top of the final tree without a commit.
    """

    def __init__(self):
        self.commits = []
        self.dirty = []
        self.worktree = {}

    def clone(self):
        t = Timeline()
        t.commits = copy.deepcopy(self.commits)
        t.dirty = list(self.dirty)
        t.worktree = dict(self.worktree)
        return t

    def add(self, label, utc, author, trailers, branches, changes):
        if any(c['label'] == label for c in self.commits):
            raise ValueError('duplicate commit label %s' % label)
        self.commits.append({'label': label, 'utc': utc, 'author': author,
                             'trailers': copy.deepcopy(trailers), 'branches': list(branches),
                             'changes': dict(changes)})
        return commit_sha(label)

    def index(self, label):
        for i, c in enumerate(self.commits):
            if c['label'] == label:
                return i
        raise KeyError(label)

    def get(self, label):
        return self.commits[self.index(label)]

    def sha(self, label):
        self.index(label)
        return commit_sha(label)

    def head_sha(self):
        return commit_sha(self.commits[-1]['label'])

    def tree_at(self, label=None):
        """Files as of the commit `label` (inclusive); None means the final committed tree."""
        tree = {}
        for c in self.commits:
            for p, b in c['changes'].items():
                if b is None:
                    tree.pop(p, None)
                else:
                    tree[p] = b
            if label is not None and c['label'] == label:
                break
        return tree

    def final_tree(self):
        tree = self.tree_at(None)
        for p, b in self.worktree.items():
            if b is None:
                tree.pop(p, None)
            else:
                tree[p] = b
        return tree

    def file(self, path, label=None):
        return self.tree_at(label).get(path)

    def last_toucher(self, path):
        last = None
        for c in self.commits:
            if path in c['changes']:
                last = c['label']
        return last


def write_fixture(dest, timeline, fixture_meta=None, keep=()):
    """Write a fixture directory: FIXTURE.json (optional), timeline.json, repo/, history/. Entries
    named in `keep` (hand-written files such as EXPECTED_VALUES.md) survive the rewrite."""
    if os.path.isdir(dest):
        for name in os.listdir(dest):
            if name in keep:
                continue
            full = os.path.join(dest, name)
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
    else:
        os.makedirs(dest)
    final = timeline.final_tree()
    for p in sorted(final):
        _write(os.path.join(dest, 'repo', p), final[p])
    last = {}
    for c in timeline.commits:
        for p in c['changes']:
            last[p] = c['label']
    for c in timeline.commits:
        sha = commit_sha(c['label'])
        for p in sorted(c['changes']):
            b = c['changes'][p]
            if b is None:
                continue
            if last[p] != c['label'] or final.get(p) != b:
                _write(os.path.join(dest, 'history', sha, p), b)
    tl = {'head': timeline.head_sha(), 'dirty': sorted(timeline.dirty), 'shallow': False,
          'commits': []}
    for c in timeline.commits:
        tl['commits'].append({'sha': commit_sha(c['label']), 'utc': c['utc'], 'author': c['author'],
                              'trailers': c['trailers'], 'paths': sorted(c['changes']),
                              'branches': sorted(c['branches']), 'label': c['label']})
    _write(os.path.join(dest, 'timeline.json'), dump_json(tl))
    if fixture_meta is not None:
        _write(os.path.join(dest, 'FIXTURE.json'), dump_json(fixture_meta))
    # files beside repo/ (outside the fixture's repository), e.g. an owner key file for `heldout reveal`
    for rel, data in sorted(getattr(timeline, 'root_files', {}).items()):
        _write(os.path.join(dest, rel), data)


def fixture_files(timeline, fixture_meta=None, prefix=''):
    """The files write_fixture would write, as {prefix + relative path: bytes} (used to embed a runnable
    nested fixture inside another fixture's kit tree, e.g. C00f's tools/minifly_check/corpus/X01/)."""
    out = {}
    final = timeline.final_tree()
    for p in sorted(final):
        out[prefix + 'repo/' + p] = final[p]
    last = {}
    for c in timeline.commits:
        for p in c['changes']:
            last[p] = c['label']
    for c in timeline.commits:
        sha = commit_sha(c['label'])
        for p in sorted(c['changes']):
            b = c['changes'][p]
            if b is None:
                continue
            if last[p] != c['label'] or final.get(p) != b:
                out[prefix + 'history/%s/%s' % (sha, p)] = b
    tl = {'head': timeline.head_sha(), 'dirty': sorted(timeline.dirty), 'shallow': False, 'commits': []}
    for c in timeline.commits:
        tl['commits'].append({'sha': commit_sha(c['label']), 'utc': c['utc'], 'author': c['author'],
                              'trailers': c['trailers'], 'paths': sorted(c['changes']),
                              'branches': sorted(c['branches']), 'label': c['label']})
    out[prefix + 'timeline.json'] = dump_json(tl)
    if fixture_meta is not None:
        out[prefix + 'FIXTURE.json'] = dump_json(fixture_meta)
    return out


def _write(path, data):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, 'wb') as fh:
        fh.write(data)


def dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total
