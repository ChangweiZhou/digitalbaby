"""LOCK-14 (governance corpus fixture C00f only): task.format.gen_cap_bytes is an integer in [1, 64].

A real, runnable check added by the C00f governance range, so that the base and head checkers can be run
on the fixture's nested corpus (SPEC 13.2: code changes are classified by running both versions)."""
import common
from common import FAIL, PASS, Result


@common.check('LOCK-14')
def check_lock_14(ctx):
    fmt = (ctx.lock.get('task') or {}).get('format') or {}
    v = fmt.get('gen_cap_bytes')
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 64:
        return Result('LOCK-14', FAIL, 'task.format.gen_cap_bytes %r is not an integer in [1, 64]' % (v,))
    return Result('LOCK-14', PASS, 'task.format.gen_cap_bytes %d' % v)
