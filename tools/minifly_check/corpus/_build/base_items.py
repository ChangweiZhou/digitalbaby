"""The hand-designed item universe and arm behaviours of the corpus base delivery.

Every number derived from this file is hand-computed in base/EXPECTED_VALUES.md. Design notes:
- Task: single-digit addition, prompt 'a+b=', answer decimal, EOS 0x0a, gen_cap 4 bytes.
- Taught units: the cross through 5, i.e. (a,5) for a=1..8 and (5,b) for b in 1..8, b != 5 (15 items).
  Under pair_rectangle with a'=5, b'=5 every held (a,b) with a,b != 5 is determined:
  (a,5), (5,b) and (5,5) are taught.
- HELD: 8 S2 items (no carry, 1-digit answers) and 8 S3 items (carry, 2-digit answers, none equal
  to 11 so the CONST answer '11' scores 0). No zero operand, so no S0; no canonical orbit is taught
  (every taught orbit contains 5), so no S1; every operand value occurs in its position in a taught
  pair and no answer is longer than the longest taught answer, so no S4.
- On HELD the best shortcut is DIGIT_NO_CARRY at exactly 8/16 = 0.5 in every world: the candidate W
  is wrong on at least one S2 and one S3 item per world, so FIXED_1 and FIXED_2 of its own stream stay
  below 0.5; CONST/SUFFIX/EXACT/CANON/NEAREST/COPY/ID_ZERO/MODE_REPEAT score 0.
"""

GEN_CAP = 4
EOS = '\n'


def prompt(a, b):
    return '%d+%d=' % (a, b)


def truth(a, b):
    return str(a + b)


def dnc(a, b):
    """DIGIT_NO_CARRY for single-digit operands."""
    return str((a + b) % 10)


TAUGHT_PAIRS = [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5),
                (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8)]
OLD_PAIRS = [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5)]
NEW_PAIRS = [(5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8)]
HELD_S2 = [(1, 2), (2, 1), (2, 2), (1, 4), (3, 2), (2, 4), (4, 2), (3, 4)]
HELD_S3 = [(2, 8), (3, 7), (6, 4), (4, 8), (8, 4), (6, 7), (7, 7), (8, 8)]
HELD_PAIRS = HELD_S2 + HELD_S3

# SHUF: orbit-consistent permutation of taught answers: {2,5}<->{4,5}, {6,5}<->{8,5};
# {1,5}, {3,5}, {5,5}, {7,5} keep their answers (agreement 7/15 items, 4/8 orbits).
SHUF_SWAP = {frozenset((2, 5)): '9', frozenset((4, 5)): '7',
             frozenset((6, 5)): '13', frozenset((8, 5)): '11'}


def shuf_label(a, b):
    key = frozenset((a, b))
    return SHUF_SWAP.get(key, truth(a, b))


def pool_items():
    """{pool: [(item_id, a, b)]}."""
    return {
        'TAUGHT': [('t%02d' % (i + 1), a, b) for i, (a, b) in enumerate(TAUGHT_PAIRS)],
        'OLD': [('o%02d' % (i + 1), a, b) for i, (a, b) in enumerate(OLD_PAIRS)],
        'NEW': [('n%02d' % (i + 1), a, b) for i, (a, b) in enumerate(NEW_PAIRS)],
        'HELD': [('h%02d' % (i + 1), a, b) for i, (a, b) in enumerate(HELD_PAIRS)],
    }


# Cells written per (world, native arm): (pool, checkpoint).
NATIVE_CELLS = [('TAUGHT', 'prefix'), ('OLD', 'old_end'), ('OLD', 'final'), ('NEW', 'final'),
                ('HELD', 'final')]
# Cells written per (technical world, testbed arm): the primary pool and both OLD guard checkpoints.
TESTBED_CELLS = [('HELD', 'final'), ('OLD', 'old_end'), ('OLD', 'final')]

TECH_WORLDS = ['w_tech_%d' % i for i in range(1, 9)]
DEV_WORLDS = ['w_dev_1', 'w_dev_2']
REH_WORLDS = ['w_reh_1']
OFF_WORLDS = ['w_off_%d' % i for i in range(1, 9)]
NATIVE_TECH_WORLD = 'w_tech_1'

ARMS = ['W', 'NPT', 'SHUF', 'FROZEN', 'ORACLE']
CEILINGS = ['EXACT_BP', 'ORACLE_FEATURES']
FLOORS = ['FRF_W256_LR0.001', 'FRF_W256_LR0.005', 'FRF_W256_LR0.02',
          'FRF_W1024_LR0.001', 'FRF_W1024_LR0.005', 'FRF_W1024_LR0.02',
          'FRF_W4096_LR0.001', 'FRF_W4096_LR0.005', 'FRF_W4096_LR0.02']

# W's wrong HELD items per world: (S2 indexes, S3 indexes) into HELD_S2 / HELD_S3.
W_HELD_WRONG = {
    'w_off_1': ([0], [0]), 'w_off_2': ([1], [1]), 'w_off_3': ([2], [2, 3]), 'w_off_4': ([3], [4]),
    'w_off_5': ([4, 5], [5]), 'w_off_6': ([6], [6]), 'w_off_7': ([7], [7, 0]), 'w_off_8': ([0], [2]),
    'w_dev_1': ([1], [3]), 'w_dev_2': ([5], [6, 7]), 'w_reh_1': ([2], [1]), 'w_tech_1': ([3], [5]),
}
# W's OLD item lost between old_end and final (index into OLD_PAIRS), per world.
W_OLD_DROP = {'w_off_3': 5, 'w_off_6': 1}
# NPT gets this one S3 item right in addition to the DIGIT_NO_CARRY answers.
NPT_LUCKY = (2, 8)
# ORACLE (plumbing) is right on the first 4 HELD items only; elsewhere on HELD it answers '11'.
ORACLE_HELD_RIGHT = 4


def wrong(ans):
    """A wrong answer of the same length class: 1-digit v -> v+1, 2-digit v -> v+2."""
    v = int(ans)
    return str(v + 1) if v <= 8 else str(v + 2)


def default_behaviour():
    """Mutable behaviour description; mutations edit a copy of it before rows are generated."""
    return {
        'w_held_wrong': {k: (list(v[0]), list(v[1])) for k, v in W_HELD_WRONG.items()},
        'w_old_drop': dict(W_OLD_DROP),
        'w_mode': None,           # None | 'fixed2' (C06a) ; HELD behaviour of W
        'w_old_mode': None,       # None | 'dnc' (C06c): W answers OLD at shortcut level
        'npt_mode': 'dnc',        # 'dnc' | 'copy_w' (NPT identical to W) | 'zero' (NPT answers '0')
        'npt_s0_wrong': False,    # C06b: NPT is wrong on zero-operand items
        'ceiling_mode': 'pass',   # 'pass' or 'fail'
        'floor_pass': [],         # floor ids that behave like a ceiling
    }


def _held_index(item_id):
    """(stratum class, index within class) for HELD item ids: h01..h08 S2, then S3, then extras."""
    idx = int(item_id[1:]) - 1
    n2, n3 = len(HELD_S2), len(HELD_S3)
    if idx < n2:
        return 'S2', idx
    if idx < n2 + n3:
        return 'S3', idx - n2
    return 'X', idx - n2 - n3


def emitted_for(arm, world, pool, checkpoint, item_id, a, b, bh):
    """The emitted answer (str) of a native or testbed arm for one item. A value of GEN_CAP or more
    characters fills the stream without an EOS."""
    t = truth(a, b)
    if arm == 'W' or (arm == 'NPT' and bh['npt_mode'] == 'copy_w'):
        if pool == 'HELD':
            cls, idx = _held_index(item_id)
            if bh['w_mode'] == 'fixed2':
                if cls == 'S2':
                    return wrong(t)
                if cls == 'S3':
                    n_eos = 6 if _windex(world) % 2 == 1 else 5
                    return t if idx < n_eos else t + '00'
                return t
            s2w, s3w = bh['w_held_wrong'].get(world, ([0], [0]))
            if (cls == 'S2' and idx in s2w) or (cls == 'S3' and idx in s3w):
                return wrong(t)
            return t
        if pool == 'OLD' and bh['w_old_mode'] == 'dnc':
            return dnc(a, b)
        if pool == 'OLD' and checkpoint == 'final' and world in bh['w_old_drop']:
            if int(item_id[1:]) - 1 == bh['w_old_drop'][world]:
                return wrong(t)
        if pool == 'NEW' and bh.get('w_new_const') and world in (bh.get('w_new_const_worlds') or ()):
            return bh['w_new_const']  # S37: one constant answer on the NEW pool (modal fraction 1.0)
        return t
    if arm == 'NPT':
        if bh['npt_mode'] == 'zero':
            return '0'
        if bh['npt_s0_wrong'] and (a == 0 or b == 0):
            return wrong(t)
        if pool == 'HELD' and (a, b) == NPT_LUCKY:
            return t
        return dnc(a, b)
    if arm == 'SHUF':
        if pool == 'HELD':
            return str(b)
        return shuf_label(a, b)
    if arm == 'FROZEN':
        if pool == 'HELD' and world in (bh.get('frozen_like_w_held') or ()):
            return emitted_for('W', world, pool, checkpoint, item_id, a, b, bh)  # S59: probe 2c
        return str(a)
    if arm == 'ORACLE':
        if pool == 'HELD':
            return t if int(item_id[1:]) <= ORACLE_HELD_RIGHT else '11'
        return t
    if arm.startswith('CEILING_'):
        return _ceiling(arm[len('CEILING_'):], world, pool, checkpoint, item_id, a, b, bh)
    if arm.startswith('FLOOR_'):
        fid = arm[len('FLOOR_'):]
        if fid in bh['floor_pass']:
            return _ceiling('EXACT_BP', world, pool, checkpoint, item_id, a, b, bh, force=True)
        return _floor(fid, world, pool, item_id, a, b)
    raise KeyError(arm)


def _windex(world):
    return int(world.rsplit('_', 1)[1])


def _ceiling(cid, world, pool, checkpoint, item_id, a, b, bh, force=False):
    t = truth(a, b)
    if bh['ceiling_mode'] == 'fail' and not force:
        # A ceiling that never learned the composition: digitwise sum without carry.
        return dnc(a, b)
    wi = _windex(world)
    idx = int(item_id[1:])
    if cid == 'EXACT_BP':
        if pool == 'HELD' and idx == 16 and wi % 2 == 0:
            return wrong(t)
        if pool == 'OLD' and checkpoint == 'final' and idx == 3 and wi in (3, 7):
            return wrong(t)
        return t
    # ORACLE_FEATURES
    if pool == 'HELD' and idx == 15 and wi % 2 == 1:
        return wrong(t)
    if pool == 'OLD' and checkpoint == 'final' and idx == 5 and wi in (2, 6):
        return wrong(t)
    return t


def _floor(fid, world, pool, item_id, a, b):
    """Floors answer at shortcut level (DIGIT_NO_CARRY) with one config-specific extra error."""
    j = FLOORS.index(fid)
    wi = _windex(world)
    idx = int(item_id[1:]) - 1
    if pool == 'HELD' and idx == (j % len(HELD_S2)) and (wi + j) % 2 == 0:
        return wrong(dnc(a, b))
    return dnc(a, b)


def stream_of(emitted):
    """stream_hex bytes: the emitted answer, then EOS bytes up to gen_cap (generation continues past
    EOS; this learner keeps emitting EOS)."""
    s = (emitted + EOS * GEN_CAP)[:GEN_CAP]
    return s
