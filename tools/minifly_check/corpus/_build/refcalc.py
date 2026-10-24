"""Independent reference arithmetic for the corpus base (documentation only).

Written from SPEC Sections 9.1-9.6 without reading the checker. It produces the hand-checked numbers
in base/EXPECTED_VALUES.md and the executor-side copies of results/dev/SCORES.json and
results/admission/TESTBED.json. selftest never runs it; the checker's own numbers are compared
against EXPECTED.json, which states these values.
Tie rule used here (SPEC 9.2): ties go to the lexicographically smallest answer (byte order).
"""
import math
from collections import Counter

# Student-t one-sided 0.95 quantiles (df: value), from standard tables.
T95 = {1: 6.313752, 2: 2.919986, 3: 2.353363, 4: 2.131847, 5: 2.015048, 6: 1.943180,
       7: 1.894579, 8: 1.859548, 9: 1.833113, 10: 1.812461}

EXPOSURE_INDEPENDENT = ['CONST', 'LENFREQ', 'SUFFIX1', 'SUFFIX2', 'SUFFIX3', 'COPY_A', 'COPY_B',
                        'ID_ZERO', 'DIGIT_NO_CARRY'] + ['FIXED_%d' % k for k in range(9)] + ['MODE_REPEAT']
FULL = EXPOSURE_INDEPENDENT + ['EXACT', 'CANON', 'NEAREST']


def dec(h):
    return bytes.fromhex(h).decode('ascii')


def parse(p):
    a, rest = p.split('+')
    return int(a), int(rest.rstrip('='))


def emitted_from_stream(stream, eos='\n'):
    i = stream.find(eos)
    return (stream, False) if i < 0 else (stream[:i], True)


def is_exact(row):
    em, own = emitted_from_stream(dec(row['stream_hex']))
    a, b = parse(dec(row['prompt_hex']))
    return own and em == str(a + b) and row['answer_path'] == 'native'


def _most_frequent(answers):
    if not answers:
        return None
    c = Counter(answers)
    top = max(c.values())
    return sorted([a for a, n in c.items() if n == top], key=lambda s: s.encode('ascii'))[0]


def ladder(taught, cand_rows, pool):
    """taught: [(prompt, answer)] of the candidate; cand_rows: rows of one (world, pool, checkpoint)
    cell of the candidate. Returns {member: exact rate over the rows given}."""
    const = _most_frequent([ans for _, ans in taught])
    tmap = dict(taught)
    canon = {}
    for p, ans in taught:
        a, b = parse(p)
        canon.setdefault(tuple(sorted((a, b))), ans)
    streams = [emitted_from_stream(dec(r['stream_hex']))[0] for r in cand_rows]
    lens = Counter(len(s) for s in streams)
    bytes_c = Counter(ch for s in streams for ch in s)
    mode_len = sorted([n for n, k in lens.items() if k == max(lens.values())])[0] if lens else 0
    mode_byte = sorted([c for c, k in bytes_c.items() if k == max(bytes_c.values())])[0] if bytes_c else ''
    if pool == 'HELD':
        members = FULL
    elif pool == 'PRESENTATION':
        members = [m for m in FULL if m != 'CANON']
    else:
        members = EXPOSURE_INDEPENDENT
    out = {}
    for m in members:
        hit = 0
        for r in cand_rows:
            p = dec(r['prompt_hex'])
            a, b = parse(p)
            t = str(a + b)
            stream = dec(r['stream_hex'])
            pred = const
            if m == 'LENFREQ':
                pred = _most_frequent([ans for q, ans in taught if len(q) == len(p)]) or const
            elif m.startswith('SUFFIX'):
                k = int(m[-1])
                pred = _most_frequent([ans for q, ans in taught if q[-k:] == p[-k:]]) or const
            elif m == 'EXACT':
                pred = tmap.get(p, const)
            elif m == 'CANON':
                pred = canon.get(tuple(sorted((a, b))), const)
            elif m == 'NEAREST':
                best = max(sum(1 for x, y in zip(q, p) if x == y) for q, _ in taught)
                cands = [ans for q, ans in taught if sum(1 for x, y in zip(q, p) if x == y) == best]
                pred = sorted(cands, key=lambda s: s.encode('ascii'))[0]
            elif m == 'COPY_A':
                pred = str(a)
            elif m == 'COPY_B':
                pred = str(b)
            elif m == 'ID_ZERO':
                pred = str(b) if a == 0 else (str(a) if b == 0 else const)
            elif m == 'DIGIT_NO_CARRY':
                pred = str((a + b) % 10)
            elif m.startswith('FIXED_'):
                pred = stream[:int(m.split('_')[1])]
            elif m == 'MODE_REPEAT':
                pred = mode_byte * mode_len
            if pred == t:
                hit += 1
        out[m] = hit / float(len(cand_rows)) if cand_rows else 0.0
    return out


def best(lad):
    top = max(lad.values())
    names = sorted(k for k, v in lad.items() if v == top)
    return names, top


def paired_lcb(d, alpha=0.05):
    n = len(d)
    mean = sum(d) / n
    if n < 2:
        return mean, None, mean - 2 * math.sqrt(math.log(1 / alpha) / (2 * n))
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    if sd == 0:
        return mean, 0.0, mean - 2 * math.sqrt(math.log(1 / alpha) / (2 * n))
    return mean, sd, mean - T95[n - 1] * sd / math.sqrt(n)


def paired_ucb(d, alpha=0.05):
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    if sd == 0:
        return mean, sd, mean + 2 * math.sqrt(math.log(1 / alpha) / (2 * n))
    return mean, sd, mean + T95[n - 1] * sd / math.sqrt(n)


def pct(x):
    return '%.1f' % (100.0 * x)
