"""Fixed observable stream, actual-call clamps, disposable pre-feedback probes."""
import dataclasses
import math
import types
import numpy as np
from compact_fixture import DT, RECORD_SECONDS, permitted


def instrument(core):
    trace = []
    for index, m in enumerate(core.models):
        for name in ('teach_logged', 'teach_signed'):
            original = getattr(type(m), name)
            def wrapped(self, *args, _orig=original, _name=name, _index=index, **kwargs):
                write = kwargs['write']
                expected = None
                if not write:
                    t = args[-1]
                    expected = self.fly.clone()
                    expected.event(self._teach_prologue(t), self.pending_x, 0., False)
                result = _orig(self, *args, **kwargs)
                if not write:
                    for attr in ('fast', 'slow', 'adapt'):
                        if not np.array_equal(getattr(self.fly.m, attr), getattr(expected.m, attr)):
                            raise AssertionError('clamped write altered native state')
                    if self.fly.m.elapsed != expected.m.elapsed or result != 0.:
                        raise AssertionError('clamped clock/write mismatch')
                trace.append({'store': _index, 'api': _name, 'write': write,
                              'coefficients': [float(x) for x in args[:-1]], 'applied_l1': float(result),
                              'no_write_reference_equal': expected is not None})
                return result
            setattr(m, name, types.MethodType(wrapped, m))
    return trace


def record(core, event, branch, trace):
    trace.clear()
    at = event['at']
    for i, byte in enumerate(bytes.fromhex(event['cue_hex'])):
        core.feed(byte, at + i * DT)
    prediction = core.predict(at + 12 * DT)
    learn = permitted(branch, event['stage'])
    receipt = core.observe_outcome(event['outcome'], at + 12 * DT, learn=learn)
    core.feed(10, at + 13 * DT)
    error = core.flush(at + RECORD_SECONDS)
    if error >= 1e-8 or len(trace) != len(core.models): raise AssertionError('record interface mismatch')
    return {'index': event['index'], 'stage': event['stage'], 'item': event['item'], 'learn': learn,
            'predicted_at': at + 12 * DT, 'observed_at': at + 12 * DT, 'prediction_precedes_outcome': True,
            'prediction': dataclasses.asdict(prediction), 'write': receipt,
            'actual_calls': list(trace), 'flush_error': float(error)}


def probe(core, world, at, name):
    before = core.state_digest()
    rows = []
    for stage in ('old', 'new', 'revision'):
        if name.startswith('old_') and stage != 'old': continue
        # Old and revised readings are separately labelled; revised keys use their current target.
        for item, cue in enumerate(world['sets'][stage]):
            c = core.clone()
            for i, byte in enumerate(bytes.fromhex(cue['cue_hex'])): c.feed(byte, at + i * DT)
            p = c.predict(at + 12 * DT)
            rows.append({'stage': stage, 'item': item, 'prediction': dataclasses.asdict(p),
                         'correct': int(p.emitted == cue['outcome'])})
    if core.state_digest() != before: raise AssertionError('probe mutated continuing life')
    return {'name': name, 'at': at, 'rows': rows,
            'private_digest': [m.state_digest() for m in core.private],
            'continuing_state_digest': before}


def checkpoints_after(index):
    return {384: ('old_end', 'old_day'), 768: ('new_end', 'new_day'),
            864: ('revision_end', 'final')}.get(index, ())


def primary_scores(branch_doc, world):
    rows = {(r['stage'], r['item']): r['correct'] for r in branch_doc['probes'][-1]['rows']}
    return {stage: rows[(stage, item)] for stage, item in world['primary_items'].items()}
