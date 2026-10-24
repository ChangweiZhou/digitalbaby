#!/usr/bin/env python3
"""MiniFly research-loop kit checker (MRK-1) dispatcher.

    python3 -I tools/minifly_check/minifly_check.py [--repo-root PATH] [--git timeline:<dir>] [--json] <command> ...

Exit codes: 0 PASS; 1 FAIL; 2 MISSING_OR_INVALID_INPUT (MISSING and no FAIL);
3 CHECKER_ERROR (dispatcher exception, SELF-01 failure, unknown subcommand, usage error,
missing module or handler). Every nonzero exit blocks; there is no --force and no --skip.

Handlers are imported lazily, so a missing or broken module fails only its own command.
"""
import os
import sys

_KIT_DIR = os.path.dirname(os.path.abspath(__file__))
if not sys.path or sys.path[0] != _KIT_DIR:
    sys.path.insert(0, _KIT_DIR)

import argparse  # noqa: E402

EXIT_PASS, EXIT_FAIL, EXIT_MISSING, EXIT_ERROR = 0, 1, 2, 3

# command key -> (module, function). Keys of two-word commands are 'lock build', 'cycle close'.
HANDLERS = {
    'init': ('lockcheck', 'cmd_init'),
    'lock': ('lockcheck', 'cmd_lock_build'),
    'gate': ('common', 'cmd_gate'),
    'launch': ('launch', 'cmd_launch'),
    'cycle': ('launch', 'cmd_cycle'),
    'score': ('scorer', 'cmd_score'),
    'oc': ('decide', 'cmd_oc'),
    'seal': ('seal', 'cmd_seal'),
    'canary': ('canary', 'cmd_canary'),
    'review': ('review', 'cmd_review'),
    'claim': ('claimcheck', 'cmd_claim'),
    'lint': ('claimcheck', 'cmd_lint'),
    'ledger': ('kb', 'cmd_ledger'),
    'kb': ('kb', 'cmd_kb'),
    'selftest': ('governance', 'cmd_selftest'),
    'governance': ('governance', 'cmd_governance'),
    'heldout': ('governance', 'cmd_heldout'),
    'retro': ('retro', 'cmd_retro'),
}
# Subcommands whose spec usage ends in "[...]": unknown extra arguments are passed as args.extra.
ALLOW_EXTRA = ('seal', 'kb', 'heldout')

LAUNCH_STAGES = ('testbed', 'technical', 'dev', 'rehearsal', 'official')
GATE_STAGES = ('lock', 'roster', 'official', 'publish', 'pointer')
CYCLE_KINDS = ('testbed', 'technical', 'dev', 'rehearsal')
DECISIONS = ('continue', 'revise', 'stop', 'escalate')
LEDGERS = ('FINDINGS', 'INCIDENTS', 'USER_DECISIONS', 'REVIEW_DEBT', 'CANARY_COMMITMENTS')
TASK_FAMILIES = ('addition_multi_digit', 'addition_single_digit', 'addition_component', 'other_arithmetic',
                 'non_arithmetic')


class UsageError(Exception):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write('%s: error: %s\n' % (self.prog, message))
        raise UsageError(message)


def _global_options(p, suppress):
    kw = {'default': argparse.SUPPRESS} if suppress else {}
    p.add_argument('--repo-root', dest='repo_root', metavar='PATH',
                   help='repository root (default: git toplevel of the current directory)',
                   **({'default': argparse.SUPPRESS} if suppress else {'default': None}))
    p.add_argument('--git', dest='git', metavar='timeline:<fixture_dir>',
                   help='history source for selftest only: timeline:<fixture_dir>',
                   **({'default': argparse.SUPPRESS} if suppress else {'default': None}))
    p.add_argument('--json', dest='json', action='store_true', help='print the report as JSON to stdout', **kw)


def build_parser():
    common_opts = Parser(add_help=False)
    _global_options(common_opts, suppress=True)
    p = Parser(prog='minifly_check.py',
               description='MiniFly research-loop kit checker (MRK-1). Exit 0 PASS, 1 FAIL, '
                           '2 MISSING_OR_INVALID_INPUT, 3 CHECKER_ERROR. Every nonzero exit blocks.')
    _global_options(p, suppress=False)
    sp = p.add_subparsers(dest='command', metavar='<command>', parser_class=Parser)

    def add(name, help_):
        return sp.add_parser(name, help=help_, parents=[common_opts], description=help_)

    a = add('init', 'scaffold <dir>/ENDPOINT_LOCK.json from the template')
    a.add_argument('dir')
    a.add_argument('--parent', metavar='DIR')
    a.add_argument('--claim-level', dest='claim_level', choices=('E1', 'E2', 'E3'))
    a.add_argument('--task-family', dest='task_family', choices=TASK_FAMILIES)
    a.add_argument('--agent', choices=('codex', 'claude'), help='author_agent (default: $MINIFLY_AGENT)')

    a = add('lock', 'lock build <dir>: write SOURCE_LOCK.json')
    a.add_argument('action', choices=('build',))
    a.add_argument('dir')

    a = add('gate', 'run a stage gate bundle and write checks/GATE_<stage>_<utc>.json')
    a.add_argument('--stage', required=True, choices=GATE_STAGES)
    a.add_argument('--dry', action='store_true', help='write checks/DRY_<stage>_<utc>.json (counts only for CYC-02)')
    a.add_argument('--pointer', metavar='PATH', help='pointer file (pointer stage)')
    a.add_argument('dir')

    a = add('launch', 'launch --stage S [--resume] <dir> -- <cmd ...>: gate, clean-closure check, run, record ATTEMPTS')
    a.add_argument('--stage', required=True, choices=LAUNCH_STAGES)
    a.add_argument('--resume', action='store_true',
                   help='official only: continue an interrupted official attempt under the identical lock (missing '
                        'units only, plus a determinism replay of one completed unit; RUN-02, RUN-04)')
    a.add_argument('dir')
    a.add_argument('cmd', nargs=argparse.REMAINDER, help='-- <cmd ...>')

    a = add('cycle', 'cycle close <dir> --kind K --decision D --reason TEXT')
    a.add_argument('action', choices=('close',))
    a.add_argument('dir')
    a.add_argument('--kind', required=True, choices=CYCLE_KINDS)
    a.add_argument('--decision', required=True, choices=DECISIONS)
    a.add_argument('--reason', required=True, metavar='TEXT')

    a = add('score', 'score <dir> --role R: canonical scorer (writes results/<role>/SCORES.json)')
    a.add_argument('dir')
    a.add_argument('--role', required=True, choices=('admission', 'dev', 'rehearsal', 'official'))

    a = add('oc', 'oc <dir>: operating characteristic of the locked rule')
    a.add_argument('dir')

    a = add('seal', 'seal {commit-nonce|reveal|derive|verify} <dir> [...]')
    a.add_argument('action', choices=('commit-nonce', 'reveal', 'derive', 'verify'))
    a.add_argument('dir')
    a.add_argument('--nonce-file', dest='nonce_file', metavar='PATH', help='private nonce path outside the repo')

    a = add('canary', 'canary <dir> --seed HEX [--n 20] [--out DIR]')
    a.add_argument('dir')
    a.add_argument('--seed', required=True)
    a.add_argument('--n', type=int, default=20)
    a.add_argument('--out', metavar='DIR')

    a = add('review', 'review replay-plan <dir> --seed HEX [--n 4] | review check <dir>')
    a.add_argument('action', choices=('replay-plan', 'check'))
    a.add_argument('dir')
    a.add_argument('--seed')
    a.add_argument('--n', type=int, default=4)

    a = add('claim', 'claim <dir>: generate CLAIM.json and the headline')
    a.add_argument('dir')

    a = add('lint', 'lint <files...>: T0 text and pointer lint')
    a.add_argument('files', nargs='+')

    a = add('ledger', 'ledger append --ledger L --row ROW.json')
    a.add_argument('action', choices=('append',))
    a.add_argument('--ledger', required=True, choices=LEDGERS)
    a.add_argument('--row', required=True, metavar='ROW.json')

    a = add('kb', 'kb {status|score|backtest-snapshot|backtest-score} [...]')
    a.epilog = ('backtest-score also takes --contract FILE --kb FILE [--outcomes FILE] [--seed N] [--boot N] '
                '(parsed by kb.py); --out is a DIR for backtest-snapshot and a FILE otherwise')
    a.add_argument('action', choices=('status', 'score', 'backtest-snapshot', 'backtest-score'))
    a.add_argument('--as-of', dest='as_of')
    a.add_argument('--out', metavar='DIR')

    a = add('selftest', 'selftest [--mutate] [--only ID]: run the corpus against EXPECTED.json')
    a.add_argument('--mutate', action='store_true')
    a.add_argument('--only', metavar='ID')

    a = add('governance', 'governance --base REV --head REV: classify protected-path changes')
    a.add_argument('--base', required=True)
    a.add_argument('--head', required=True)

    a = add('heldout', 'heldout {seal|reveal|score} [...]')
    a.add_argument('action', choices=('seal', 'reveal', 'score'))
    a.add_argument('--batch')
    a.add_argument('--in', dest='in_dir', metavar='DIR')
    a.add_argument('--key-file', dest='key_file')
    a.add_argument('--checker-sha', dest='checker_sha')
    a.add_argument('--out', metavar='PATH')

    a = add('retro', 'retro (--all | <dir>) --out DIR: read-only legacy annotations')
    a.add_argument('dir', nargs='?')
    a.add_argument('--all', action='store_true')
    a.add_argument('--out', required=True, metavar='DIR')
    return p


def _split_launch(argv):
    """For the launch subcommand, everything after the first '--' is the launched command."""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ('--repo-root', '--git'):
            i += 2
            continue
        if tok.startswith(('--repo-root=', '--git=')) or tok == '--json':
            i += 1
            continue
        break
    if i < len(argv) and argv[i] == 'launch':
        if '--' in argv[i:]:
            j = argv.index('--', i)
            return argv[:j], argv[j + 1:]
        return argv, []
    return argv, None


def resolve_handler(command):
    import importlib
    mod_name, fn_name = HANDLERS[command]
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:  # noqa: BLE001 - fail closed
        return None, 'module %s unavailable for "%s": %s: %s' % (mod_name, command, type(e).__name__, e)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        return None, 'handler %s.%s missing for "%s"' % (mod_name, fn_name, command)
    return fn, ''


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, launch_cmd = _split_launch(argv)
    parser = build_parser()
    try:
        args, extra = parser.parse_known_args(argv)
    except UsageError:
        return EXIT_ERROR
    except SystemExit as e:  # --help
        return e.code if isinstance(e.code, int) else EXIT_PASS
    if not getattr(args, 'command', None):
        parser.print_help(sys.stderr)
        return EXIT_ERROR
    if extra:
        if args.command not in ALLOW_EXTRA:
            sys.stderr.write('minifly_check.py: error: unrecognized arguments: %s\n' % ' '.join(extra))
            return EXIT_ERROR
    args.extra = list(extra)
    if args.command == 'launch':
        args.cmd = list(launch_cmd or []) + [c for c in (args.cmd or []) if c != '--']
    if args.command == 'retro' and bool(args.all) == bool(args.dir):
        sys.stderr.write('retro: give exactly one of --all or <dir>\n')
        return EXIT_ERROR
    if args.command not in HANDLERS:
        sys.stderr.write('CHECKER_ERROR: unknown subcommand %r\n' % args.command)
        return EXIT_ERROR
    fn, err = resolve_handler(args.command)
    if fn is None:
        sys.stderr.write('CHECKER_ERROR: %s\n' % err)
        return EXIT_ERROR
    try:
        rc = fn(args)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (EXIT_PASS if e.code is None else EXIT_ERROR)
    except KeyboardInterrupt:
        sys.stderr.write('interrupted\n')
        return EXIT_ERROR
    except Exception as e:  # noqa: BLE001 - map the common exceptions, fail closed otherwise
        name = type(e).__name__
        if name == 'MissingInput':
            sys.stderr.write('MISSING: %s\n' % e)
            return EXIT_MISSING
        if name == 'InvalidInput':
            sys.stderr.write('FAIL (invalid input): %s\n' % e)
            return EXIT_FAIL
        import traceback
        sys.stderr.write('CHECKER_ERROR: %s: %s\n' % (name, e))
        traceback.print_exc(limit=4, file=sys.stderr)
        return EXIT_ERROR
    if isinstance(rc, bool) or not isinstance(rc, int) or rc not in (0, 1, 2, 3):
        sys.stderr.write('CHECKER_ERROR: handler for "%s" returned %r, not an exit code\n' % (args.command, rc))
        return EXIT_ERROR
    return rc


if __name__ == '__main__':
    sys.exit(main())
