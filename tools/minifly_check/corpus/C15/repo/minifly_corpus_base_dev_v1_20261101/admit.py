"""Corpus stub admit.py for the MRK-1 regression corpus (endpoint-locked; never executed).

Edited after the testbed gate failed; no amendment recorded."""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--testbed', action='store_true')
    mode.add_argument('--technical', action='store_true')
    ap.add_argument('--only-world')
    ap.add_argument('--only-arm')
    args = ap.parse_args(argv)
    if not os.environ.get('MINIFLY_ATTEMPT_ID'):
        sys.stderr.write('admit.py: run through MFC launch\n')
        return 2
    sys.stdout.write('testbed\n' if args.testbed else 'technical\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
