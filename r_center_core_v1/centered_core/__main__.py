"""JSONL byte-event adapter. No task generator or evaluator in the CLI."""
import argparse
import dataclasses
import json
import sys

from .core import CenteredCore


def dispatch(core, event):
    if not isinstance(event, dict) or 'op' not in event:
        raise ValueError('event must contain op')
    op = event['op']
    fields = {'feed': {'op', 'byte', 't'}, 'predict': {'op', 't'},
              'observe': {'op', 'byte', 't', 'learn'}, 'flush': {'op', 't'},
              'rest': {'op', 'seconds'}}
    if op not in fields or set(event) - fields[op]: raise ValueError('unknown operation or field')
    if op == 'feed': core.feed(event['byte'], event['t']); result = None
    elif op == 'predict': result = dataclasses.asdict(core.predict(event['t']))
    elif op == 'observe': result = core.observe_outcome(event['byte'], event['t'], learn=event.get('learn', True))
    elif op == 'flush': result = {'clock_error': core.flush(event['t'])}
    else: result = {'clock_error': core.rest(event['seconds'])}
    return {'op': op, 'result': result}


def main():
    parser = argparse.ArgumentParser(description='R_center engineering core v1, fixed 12-byte cues / ASCII 0..3')
    parser.add_argument('--load', metavar='CHECKPOINT')
    parser.add_argument('--save', metavar='CHECKPOINT')
    args = parser.parse_args()
    core = CenteredCore.load(args.load) if args.load else CenteredCore()
    for line_no, line in enumerate(sys.stdin, 1):
        if not line.strip(): continue
        try:
            event = json.loads(line)
            result = dispatch(core, event)
        except (ValueError, TypeError, KeyError) as exc:
            print(json.dumps({'line': line_no, 'error': str(exc)}), file=sys.stderr, flush=True)
            return 1
        print(json.dumps(result, allow_nan=False), flush=True)
    if args.save: core.save(args.save)
    return 0


if __name__ == '__main__': sys.exit(main())
