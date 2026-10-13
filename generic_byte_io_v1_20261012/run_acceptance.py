"""Prospective bounded IO engineering acceptance, not a promotion experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

from generic_byte_io import ByteCore

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'ACCEPTANCE_SPEC.json').read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixtures(seed):
    rng = random.Random(seed)
    used = set()
    cohorts = {}
    for cohort in SPEC['cohorts']:
        rows = []
        for i, (plen, alen) in enumerate(zip(SPEC['prompt_payload_lengths'], SPEC['answer_payload_lengths'])):
            if cohort == 'NEW' and plen == 0:
                plen = SPEC['new_prompt_zero_length_override']
            while True:
                prompt = bytes(rng.randrange(256) for _ in range(plen))
                if 4 <= i <= 9:
                    prompt = prompt[:-4] + b'\x00=\n\xff'
                if prompt not in used:
                    used.add(prompt)
                    break
            if i == 10:
                answer = bytes(range(256)) if cohort == 'OLD' else bytes(reversed(range(256)))
            else:
                answer = bytes(rng.randrange(256) for _ in range(alen))
            rows.append((prompt, answer))
        cohorts[cohort] = rows
    return cohorts


def train(core, rows, learn):
    target_events = 0
    payload_bytes = 0
    for _ in range(SPEC['epochs_per_cohort']):
        for prompt, answer in rows:
            core.begin_sequence()
            core.observe_bytes(prompt, learn=learn)
            core.start_reply()
            core.observe_bytes(answer, learn=learn)
            core.observe_end(learn=learn)
            payload_bytes += len(prompt) + len(answer)
            target_events += len(prompt) + len(answer) + 1
    return {'payload_bytes': payload_bytes, 'observed_target_events': target_events}


def probe(core, rows):
    before = core.learned_digest()
    results = []
    for prompt, answer in rows:
        core.begin_sequence()
        core.observe_bytes(prompt, learn=False)
        core.start_reply()
        output = core.generate(max_bytes=512)
        results.append({'prompt_hex': prompt.hex(), 'expected_hex': answer.hex(),
                        'emitted_hex': output.payload.hex(), 'status': output.status,
                        'selected_events': output.selected_events,
                        'correct': output.status == 'ENDED' and output.payload == answer})
    assert core.learned_digest() == before, 'inference changed persistent learned state'
    return results


def accuracy(rows):
    return sum(row['correct'] for row in rows) / len(rows)


def main():
    output = ROOT / 'technical' / 'acceptance'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'SUMMARY.json').exists():
        raise RuntimeError('accepted results already exist; do not silently rerun')
    source_lock = {'scope': 'ENGINEERING_ACCEPTANCE_NOT_SCIENTIFIC_CONFIRMATION',
                   'files': {str(p.relative_to(ROOT)): sha256(p) for p in
                             [ROOT / 'ACCEPTANCE_SPEC.json', ROOT / 'run_acceptance.py',
                              ROOT / 'generic_byte_io' / 'core.py', ROOT / 'generic_byte_io' / '__init__.py']}}
    (output / 'SOURCE_LOCK.json').write_text(json.dumps(source_lock, indent=2) + '\n')
    started = time.perf_counter()
    worlds = []
    training = {'payload_bytes': 0, 'observed_target_events': 0}
    for seed in SPEC['world_seeds']:
        w, n = ByteCore(), ByteCore()
        data = fixtures(seed)
        disabled_before = n.learned_digest()
        for core, learn in [(w, True), (n, False)]:
            count = train(core, data['OLD'], learn)
            for key in training:
                training[key] += count[key]
        initial = {'W': probe(w, data['OLD']), 'N': probe(n, data['OLD'])}
        w.rest(86400)
        n.rest(86400)
        for core, learn in [(w, True), (n, False)]:
            count = train(core, data['NEW'], learn)
            for key in training:
                training[key] += count[key]
            core.rest(86400)
        assert n.learned_digest() == disabled_before, 'no-write history mutated learned state'
        final = {label: {cohort: probe(core, data[cohort]) for cohort in SPEC['cohorts']}
                 for label, core in [('W', w), ('N', n)]}
        # Explicit state lesion; no encoder or replacement predictor is supplied.
        w._weights.fill(0)
        final['ERASE'] = {cohort: probe(w, data[cohort]) for cohort in SPEC['cohorts']}
        metrics = {'old_immediate_W': accuracy(initial['W']),
                   'old_final_W': accuracy(final['W']['OLD']),
                   'new_final_W': accuracy(final['W']['NEW']),
                   'old_final_N': accuracy(final['N']['OLD']),
                   'old_final_ERASE': accuracy(final['ERASE']['OLD'])}
        receipt = {'seed': seed, 'classification': source_lock['scope'],
                   'initial': initial, 'final': final, 'metrics': metrics,
                   'storage': w.storage_accounting(),
                   'no_write_frozen': True, 'inference_learning_frozen': True}
        (output / f'FIXTURE_{seed}.json').write_text(json.dumps(receipt, indent=2) + '\n')
        worlds.append(metrics)
        print(json.dumps({'seed': seed, **metrics}), flush=True)
    means = {key: sum(world[key] for world in worlds) / len(worlds) for key in worlds[0]}
    rules = SPEC['pass']
    gates = {'sequence_acquisition': means['old_immediate_W'] >= rules['exact_sequence_and_model_EOS_mean_min'],
             'sequence_retention': means['old_final_W'] >= rules['exact_sequence_and_model_EOS_mean_min'],
             'new_sequence_learning': means['new_final_W'] >= rules['exact_sequence_and_model_EOS_mean_min'],
             'every_fixture_retention_and_new': all(min(row['old_final_W'], row['new_final_W']) >= rules['each_world_old_and_new_retained_min'] for row in worlds),
             'gain_over_no_write': means['old_final_W'] - means['old_final_N'] >= rules['old_gain_over_no_write_and_erase_min'],
             'gain_over_erasure': means['old_final_W'] - means['old_final_ERASE'] >= rules['old_gain_over_no_write_and_erase_min'],
             'source_unchanged': all(sha256(ROOT / key) == value for key, value in source_lock['files'].items())}
    summary = {'classification': source_lock['scope'], 'evidence': SPEC['evidence'],
               'fixture_count': len(worlds), 'distinct_taught_sequences': len(worlds) * 32,
               'scored_sequence_episodes': len(worlds) * 128,
               'training': training, 'means': means, 'gates': gates,
               'verdict': 'PASS_BEHAVIORAL_ENGINEERING_CHECKS' if all(gates.values()) else 'FAIL_ENGINEERING_CHECKS',
               'wall_seconds': time.perf_counter() - started,
               'separate_interface_audit_required': True}
    (output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
