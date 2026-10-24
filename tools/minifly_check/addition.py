"""MRK-1 addition task semantics (WP3): truth, canonical orbits, strata S0-S4, determinacy witness.

SPEC Sections 9.1 (truth), 9.3 (strata) and 9.4 (witness). Python 3.9+, standard library only.
This module never trains or runs a learner; it only parses prompts and reasons about taught sets.

Prompt format (lock.task.format, kind 'addition'): prompt = a + sep + b + terminator, with a and b
non-empty ASCII decimal digit strings. The expected answer is the normalized decimal sum (no leading
zeros, '0' for zero) in ASCII, written as lowercase hex.

Conventions used throughout:
- an operand "value" is its normalized digit string (leading zeros stripped);
- a taught pair is the (a, b) of a taught prompt, in the orientation it was taught;
- the witness (pair_rectangle / column_closure) uses only feedback-bearing taught rows
  (Taught.from_rows); presentation-only rows still count for S1 and S2/S4 membership.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import common

STRATA = ('S0', 'S1', 'S2', 'S3', 'S4')
WITNESSES = ('pair_rectangle', 'column_closure')
DEFAULT_FMT = {'kind': 'addition', 'sep_hex': '2b', 'terminator_hex': '3d', 'eos_hex': '0a',
               'normalize_leading_zeros': True}
_MAX_DIGITS = 4000  # Python's int/str conversion limit is 4300 digits
_DIGITS = frozenset(b'0123456789')

Pair = Tuple[str, str]


# --------------------------------------------------------------------------------------
# Truth semantics
# --------------------------------------------------------------------------------------

def is_addition(fmt: Any) -> bool:
    return isinstance(fmt, dict) and fmt.get('kind') == 'addition'


def _fmt_bytes(fmt: dict, key: str) -> bytes:
    v = fmt.get(key)
    if not isinstance(v, str):
        raise common.InvalidInput('task.format.%s must be a hex string, got %r' % (key, v))
    try:
        b = bytes.fromhex(v)
    except ValueError:
        raise common.InvalidInput('task.format.%s is not hex: %r' % (key, v))
    if not b:
        raise common.InvalidInput('task.format.%s must not be empty' % key)
    return b


_FMT_CACHE: Dict[Tuple[str, str], Tuple[bytes, bytes]] = {}


def _sep_term(fmt: dict) -> Tuple[bytes, bytes]:
    key = (str(fmt.get('sep_hex')), str(fmt.get('terminator_hex')))
    st = _FMT_CACHE.get(key)
    if st is None:
        st = (_fmt_bytes(fmt, 'sep_hex'), _fmt_bytes(fmt, 'terminator_hex'))
        if set(st[0]) & _DIGITS or set(st[1]) & _DIGITS:
            raise common.InvalidInput('task.format sep/terminator must not contain decimal digits')
        _FMT_CACHE[key] = st
    return st


def norm(s: str) -> str:
    """Normalized decimal digit string: leading zeros stripped, '0' for zero."""
    t = str(s).lstrip('0')
    return t if t else '0'


def parse(prompt_hex: Any, fmt: Any) -> Optional[Pair]:
    """(a, b) as the decimal digit strings written in the prompt, or None when the prompt is not
    exactly digits + sep + digits + terminator under `fmt` (or fmt is not an addition format)."""
    if not is_addition(fmt) or not isinstance(prompt_hex, str):
        return None
    try:
        data = bytes.fromhex(prompt_hex)
    except ValueError:
        return None
    sep, term = _sep_term(fmt)
    if not data.endswith(term):
        return None
    body = data[:len(data) - len(term)]
    parts = body.split(sep)
    if len(parts) != 2:
        return None
    a, b = parts
    if not a or not b or len(a) > _MAX_DIGITS or len(b) > _MAX_DIGITS:
        return None
    if not (set(a) <= _DIGITS and set(b) <= _DIGITS):
        return None
    return a.decode('ascii'), b.decode('ascii')


def _require_normalized(fmt: dict) -> None:
    if fmt.get('normalize_leading_zeros') is not True:
        # SPEC 8 only defines normalized decimal answers; any other convention is not recomputable.
        raise common.InvalidInput('task.format.normalize_leading_zeros must be true: the kit only defines '
                                  'normalized decimal answers (got %r)' % (fmt.get('normalize_leading_zeros'),))


def sum_str(a: str, b: str) -> str:
    return str(int(a) + int(b))


def expected_bytes(prompt_hex: Any, fmt: Any) -> Optional[bytes]:
    p = parse(prompt_hex, fmt)
    if p is None:
        return None
    _require_normalized(fmt)
    return sum_str(p[0], p[1]).encode('ascii')


def expected(prompt_hex: Any, fmt: Any) -> Optional[str]:
    """Lowercase hex of the normalized decimal sum, or None when the prompt does not parse."""
    e = expected_bytes(prompt_hex, fmt)
    return None if e is None else e.hex()


def canonical(a: Any, b: Any) -> Pair:
    """Canonical orbit of a pair: normalized operands sorted by numeric value."""
    na, nb = norm(a), norm(b)
    return (na, nb) if (len(na), na) <= (len(nb), nb) else (nb, na)


def has_carry(a: str, b: str) -> bool:
    """True when any column of a + b produces a carry."""
    a, b = norm(a), norm(b)
    carry = 0
    for i in range(1, max(len(a), len(b)) + 1):
        da = ord(a[-i]) - 48 if i <= len(a) else 0
        db = ord(b[-i]) - 48 if i <= len(b) else 0
        s = da + db + carry
        if s >= 10:
            return True
        carry = 0
    return False


def columns(a: str, b: str) -> Tuple[Tuple[Optional[int], Optional[int], int], ...]:
    """Column triples (a_i, b_i, carry_in_i) from the least significant column, on normalized operands.

    A position where an operand has no digit is None (a missing digit is a different input from '0').
    A final carry-out adds the column (None, None, 1), which emits the leading '1'."""
    a, b = norm(a), norm(b)
    out = []
    carry = 0
    for i in range(1, max(len(a), len(b)) + 1):
        da = ord(a[-i]) - 48 if i <= len(a) else None
        db = ord(b[-i]) - 48 if i <= len(b) else None
        out.append((da, db, carry))
        carry = 1 if (da or 0) + (db or 0) + carry >= 10 else 0
    if carry:
        out.append((None, None, 1))
    return tuple(out)


# --------------------------------------------------------------------------------------
# Taught sets
# --------------------------------------------------------------------------------------

class Taught:
    """Index of a taught set for strata and determinacy.

    pairs:  all taught pairs (used for S1 orbits, S2/S4 operand positions and answer length);
    wpairs: pairs that may serve as witness units (feedback-bearing rows; defaults to pairs).
    Pairs keep their taught orientation and are stored with normalized operands."""

    def __init__(self, pairs: Iterable[Pair], wpairs: Optional[Iterable[Pair]] = None, fmt: Optional[dict] = None):
        self.fmt = dict(fmt) if isinstance(fmt, dict) else dict(DEFAULT_FMT)
        self.pairs: Set[Pair] = set((norm(a), norm(b)) for a, b in pairs)
        self.wpairs: Set[Pair] = (set((norm(a), norm(b)) for a, b in wpairs) if wpairs is not None
                                  else set(self.pairs))
        self.orbits = set(canonical(a, b) for a, b in self.pairs)
        self.first_vals = set(a for a, _ in self.pairs)
        self.second_vals = set(b for _, b in self.pairs)
        self.max_answer_len = max((len(sum_str(a, b)) for a, b in self.pairs), default=0)
        self._right: Optional[Dict[str, Set[str]]] = None
        self._left: Optional[Dict[str, Set[str]]] = None
        self._nbrs: Optional[Dict[str, Set[str]]] = None
        self._cols: Optional[Dict[tuple, List[Pair]]] = None

    @classmethod
    def from_rows(cls, rows: Iterable[dict], fmt: dict, label: str = 'TAUGHT') -> 'Taught':
        """Build from TAUGHT rows {prompt_hex, answer_hex, feedback_bearing, presentations}. Witness pairs are
        rows with feedback_bearing true and a positive number of presentations (a row without them never serves
        as a unit). An unparseable prompt raises InvalidInput."""
        pairs: List[Pair] = []
        wpairs: List[Pair] = []
        for i, r in enumerate(rows):
            p = parse(r.get('prompt_hex'), fmt)
            if p is None:
                raise common.InvalidInput('%s row %d: prompt_hex %r does not parse under the addition format'
                                          % (label, i + 1, r.get('prompt_hex')))
            pairs.append(p)
            pres = r.get('presentations')
            if (r.get('feedback_bearing') is True and isinstance(pres, (int, float)) and not isinstance(pres, bool)
                    and pres > 0):
                wpairs.append(p)
        return cls(pairs, wpairs, fmt)

    # witness adjacency ------------------------------------------------------------
    def _adjacency(self) -> None:
        if self._right is None:
            right: Dict[str, Set[str]] = {}
            left: Dict[str, Set[str]] = {}
            nbrs: Dict[str, Set[str]] = {}
            for a, b in self.wpairs:
                right.setdefault(a, set()).add(b)
                left.setdefault(b, set()).add(a)
                nbrs.setdefault(a, set()).add(b)
                nbrs.setdefault(b, set()).add(a)
            self._right, self._left, self._nbrs = right, left, nbrs

    def has(self, x: str, y: str, commutative: bool) -> bool:
        return (x, y) in self.wpairs or (commutative and (y, x) in self.wpairs)

    def orient(self, x: str, y: str) -> Pair:
        """The taught orientation of {x, y} (as-is preferred)."""
        return (x, y) if (x, y) in self.wpairs else (y, x)

    def column_index(self) -> Dict[tuple, List[Pair]]:
        if self._cols is None:
            idx: Dict[tuple, List[Pair]] = {}
            for p in sorted(self.wpairs, key=_pair_key):
                for c in set(columns(p[0], p[1])):
                    idx.setdefault(c, []).append(p)
            self._cols = idx
        return self._cols


def _pair_key(p: Pair) -> tuple:
    return (len(p[0]), p[0], len(p[1]), p[1])


def _num_key(s: str) -> tuple:
    return (len(s), s)


def as_taught(taught: Any, fmt: Optional[dict] = None) -> Taught:
    """Coerce a Taught, an iterable of (a, b) pairs, or an iterable of TAUGHT row dicts into a Taught."""
    if isinstance(taught, Taught):
        return taught
    items = list(taught or [])
    if items and isinstance(items[0], dict):
        return Taught.from_rows(items, fmt or DEFAULT_FMT)
    return Taught([(str(a), str(b)) for a, b in items], fmt=fmt)


def as_pair(item: Any, fmt: Optional[dict] = None) -> Optional[Pair]:
    """An item as normalized (a, b): accepts a pair, {a, b}, or {prompt_hex} parsed under fmt."""
    if item is None:
        return None
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return norm(item[0]), norm(item[1])
    if isinstance(item, dict):
        if 'a' in item and 'b' in item:
            return norm(item['a']), norm(item['b'])
        p = parse(item.get('prompt_hex'), item.get('fmt') or fmt or DEFAULT_FMT)
        return None if p is None else (norm(p[0]), norm(p[1]))
    raise common.InvalidInput('unsupported item %r' % (item,))


# --------------------------------------------------------------------------------------
# Strata (SPEC 9.3)
# --------------------------------------------------------------------------------------

def stratum(item: Any, taught: Any) -> Optional[str]:
    """S0 trivial (an operand is 0, or the answer equals an operand); S1 presentation twin (canonical
    orbit taught); S4 extrapolation (an operand value unseen in its position, or the answer longer than
    every taught answer); S3 carry / S2 interpolation otherwise. None when the item does not parse."""
    t = as_taught(taught)
    p = as_pair(item, t.fmt)
    if p is None:
        return None
    a, b = p
    s = sum_str(a, b)
    if a == '0' or b == '0' or s == a or s == b:
        return 'S0'
    if canonical(a, b) in t.orbits:
        return 'S1'
    if a not in t.first_vals or b not in t.second_vals or len(s) > t.max_answer_len:
        return 'S4'
    return 'S3' if has_carry(a, b) else 'S2'


# --------------------------------------------------------------------------------------
# Determinacy witness (SPEC 9.4)
# --------------------------------------------------------------------------------------

def _rectangle(a: str, b: str, t: Taught, commutative: bool) -> Optional[Tuple[str, str]]:
    """The smallest (a', b') with a' != a, b' != b and (a, b'), (a', b), (a', b') taught, else None."""
    t._adjacency()
    if commutative:
        bprimes = sorted(t._nbrs.get(a, set()) - {b}, key=_num_key)
        aprimes = sorted(t._nbrs.get(b, set()) - {a}, key=_num_key)
        if not bprimes or not aprimes:
            return None
        for x in aprimes:
            nx = t._nbrs.get(x, set())
            for y in bprimes:
                if y in nx:
                    return x, y
        return None
    bprimes = sorted(t._right.get(a, set()) - {b}, key=_num_key)
    aprimes = sorted(t._left.get(b, set()) - {a}, key=_num_key)
    if not bprimes or not aprimes:
        return None
    for x in aprimes:
        rx = t._right.get(x, set())
        for y in bprimes:
            if y in rx:
                return x, y
    return None


def determined(item: Any, taught: Any, witness: str, commutative: bool = False) -> Optional[bool]:
    """Whether a held item is determined by the taught units under `witness`.

    pair_rectangle: there exist a' != a and b' != b with (a, b'), (a', b) and (a', b') all taught
    (either orientation when commutative). column_closure: every column triple (a_i, b_i, carry_in_i)
    the item requires occurs in some taught item (orientation as taught). None when the item does not parse."""
    t = as_taught(taught)
    p = as_pair(item, t.fmt)
    if p is None:
        return None
    a, b = p
    if witness == 'pair_rectangle':
        return _rectangle(a, b, t, bool(commutative)) is not None
    if witness == 'column_closure':
        idx = t.column_index()
        return all(c in idx for c in columns(a, b))
    raise common.InvalidInput('units.witness must be one of %s, got %r' % ('|'.join(WITNESSES), witness))


def support(item: Any, taught: Any, witness: str, commutative: bool = False) -> Optional[FrozenSet[Pair]]:
    """Taught pairs (taught orientation, normalized) of one minimal witness for the item, or None when
    it is not determined. pair_rectangle: the three corners of the smallest rectangle. column_closure:
    a greedy cover of the required column triples, pruned until no member is redundant."""
    t = as_taught(taught)
    p = as_pair(item, t.fmt)
    if p is None:
        return None
    a, b = p
    if witness == 'pair_rectangle':
        r = _rectangle(a, b, t, bool(commutative))
        if r is None:
            return None
        x, y = r
        if commutative:
            return frozenset((t.orient(a, y), t.orient(x, b), t.orient(x, y)))
        return frozenset(((a, y), (x, b), (x, y)))
    if witness == 'column_closure':
        idx = t.column_index()
        need = set(columns(a, b))
        if not all(c in idx for c in need):
            return None
        cand: Dict[Pair, Set[tuple]] = {}
        for c in need:
            for q in idx[c]:
                cand.setdefault(q, set()).add(c)
        chosen: List[Pair] = []
        uncovered = set(need)
        while uncovered:
            best = min(cand, key=lambda q: (-len(cand[q] & uncovered), _pair_key(q)))
            chosen.append(best)
            uncovered -= cand[best]
        for q in list(chosen):  # prune redundant members (keeps the witness minimal by inclusion)
            rest = [r for r in chosen if r != q]
            if set().union(*[cand[r] for r in rest]) >= need if rest else not need:
                chosen = rest
        return frozenset(chosen)
    raise common.InvalidInput('units.witness must be one of %s, got %r' % ('|'.join(WITNESSES), witness))
