"""Corpus stub runner for the MRK-1 regression corpus (never executed)."""
import json
import os
import sys

import learner


def main(argv):
    attempt = os.environ.get('MINIFLY_ATTEMPT_ID')
    if not attempt:
        sys.stderr.write('run.py: run through MFC launch\n')
        return 2
    with open('params.json', 'rb') as fh:
        params = json.loads(fh.read().decode('utf-8'))
    model = learner.Learner(params)
    sys.stdout.write(json.dumps({'attempt': attempt, 'arms': sorted(model.log)}))
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
