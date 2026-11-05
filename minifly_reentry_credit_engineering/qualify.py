"""Three engineering cycles. Only development seeds and reduced training histories."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
import tempfile
import time

import numpy as np
from learner import EOS, Learner
from worlds import exposure_digest, make_world, small_world, tokens

ROOT = Path(__file__).resolve().parent
SEEDS = (66101, 66102, 66103)


def save_result(cycle, result):
    (ROOT / f'CYCLE{cycle}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Cycle {cycle} complete', flush=True)


def cycle1():
    model = Learner(66001, dtype=np.float64)
    model.theta[:, :200] = 0
    model.h[0, 0] = 0.8
    model.memory[0, 1, 0] = 2
    model.memory[0, 2, 1] = 2
    x = np.zeros((8, 16))
    _, first_read, *_ = model._read(False)
    reentry, output, empty = (copy.deepcopy(model) for _ in range(3))
    output.route = empty.route = False
    empty.memory.fill(0)
    lp_on, _, _, read_on, *_ = reentry.predict(x)
    lp_off, _, _, read_off, *_ = output.predict(x)
    empty.predict(x)
    assert first_read[0, 2] == 0 and read_on[0, 2] > 0 and read_off[0, 2] == 0
    assert np.array_equal(output.h, empty.h)
    assert np.array_equal(reentry.h[1:], output.h[1:])
    assert np.linalg.norm(read_off) > 0 and not np.array_equal(lp_on, lp_off)
    assert abs(np.exp(lp_on).sum() - 1) < 1e-13
    save_result(1, {'status': 'PASS', 'first_read': first_read[0].tolist(),
                    'reentry_final_read': read_on[0].tolist(), 'output_only_final_read': read_off[0].tolist(),
                    'output_workspace_invariant_to_memory': True, 'untouched_circuits_bit_identical': True,
                    'sensor_sha256': model.sensor.digest, 'fixture': 'Hand-initialized routing witness, no training'})


def native_loss(model, examples):
    value = 0.0
    for x, y in examples:
        logp, _, q, r, *_ = model.predict(x)
        value -= logp[y]
        model.write(y, q, r)
    return float(value)


def cycle2():
    from reference import gradient
    rng = np.random.default_rng(66002)
    results = []
    for route in (True, False):
        for horizon in (1, 8):
            for cap in (False, True):
                model = Learner(66002, route, horizon, dtype=np.float64)
                model.h[:] = rng.uniform(-0.4, 0.4, model.h.shape)
                model.memory[:] = rng.normal(0, 0.2, model.memory.shape)
                if cap:
                    model.memory[0] *= 4.2 / np.linalg.norm(model.memory[0])
                model.theta[:, 200:216] = rng.normal(0, 0.1, (8, 16))
                ex = [(rng.integers(0, 2, (8, 16)).astype(np.float64), y) for y in b'31+22=53\n42+13=5']
                loss_ref, gt, go, h, memory = gradient(model, ex, horizon)
                local = copy.deepcopy(model)
                loss = sum(local.step(y, update=False, x=x) for x, y in ex)
                error = max(float(np.max(np.abs(local.gtheta - gt))), float(np.max(np.abs(local.ghead - go))))
                state_error = max(float(np.max(np.abs(local.h - h))), float(np.max(np.abs(local.memory - memory))))
                assert error < 1e-11 and state_error < 1e-12 and abs(loss - loss_ref) < 1e-10
                assert (local.caps > 0) == cap
                gate_gradient = float(np.linalg.norm(local.gtheta[:, 200:]))
                assert (gate_gradient > 0) == (horizon == 8)
                direction = rng.normal(size=model.theta.shape)
                direction /= np.linalg.norm(direction)
                expected = float(np.sum(gt * direction))
                # Condition each finite-difference window on the actual starting state.
                anchors, anchor = [], copy.deepcopy(model)
                for index in range(0, len(ex), horizon):
                    window = ex[index:index + horizon]
                    anchors.append((copy.deepcopy(anchor), window))
                    native_loss(anchor, window)
                def perturbed(eps):
                    total = 0.0
                    for initial, window in anchors:
                        trial = copy.deepcopy(initial)
                        trial.theta += eps * direction
                        total += native_loss(trial, window)
                    return total
                finite = (perturbed(1e-5) - perturbed(-1e-5)) / 2e-5
                assert abs(finite - expected) < 1e-7
                independent = copy.deepcopy(model)
                independent.theta[0, 0] += 0.05
                native_loss(independent, ex)
                assert np.array_equal(independent.h[1:], local.h[1:])
                assert np.array_equal(independent.memory[1:], local.memory[1:])
                results.append({'route': route, 'window': horizon, 'active_cap_fixture': cap,
                                'max_gradient_error': error, 'max_state_error': state_error,
                                'directional_error': abs(finite - expected), 'gate_gradient_norm': gate_gradient,
                                'cap_events': local.caps})
    save_result(2, {'status': 'PASS', 'fixtures': results,
                    'exact_zero_cross_circuit_effect': True, 'reverse_mode': 'Torch, matched detach windows'})


def score(model, panel):
    rows = []
    for prompt, answer in panel:
        before = model.digest()
        response = model.probe(prompt)
        assert model.digest() == before
        expected = [*answer, EOS]
        rows.append({'prompt': prompt.decode(), 'expected': expected, 'emitted': response,
                     'exact': response == expected, 'own_eos': bool(response and response[-1] == EOS)})
    return rows


def local_train(model, stream):
    started = time.process_time()
    loss = sum(model.step(y) for y in stream)
    return loss, time.process_time() - started


def cycle3():
    from reference import train_block
    arms = [('R8', True, 8), ('O8', False, 8), ('R1', True, 1), ('O1', False, 1)]
    all_rows, layouts = [], []
    for seed in SEEDS:
        planned = make_world(seed)
        layout = {'seed': seed, 'phase_records': list(map(len, planned['phases'])),
                  'phase_symbols': [len(tokens(p)) for p in planned['phases']],
                  'exposure_sha256': [exposure_digest(p) for p in planned['phases']],
                  'primary': len(planned['primary']), 'extrapolation': len(planned['extrapolation']),
                  'counterfactual_pairs': len(planned['counterfactual']), 'witness_units': planned['witness_units']}
        layouts.append(layout)
        phases, held = small_world(seed)
        data = [tokens(p) for p in phases]
        assert not set(p for p, _ in held) & {p for phase in phases for p, _ in phase}
        for arm, route, window in arms:
            model = Learner(seed, route, window)
            timing = []
            for phase_index, stream in enumerate(data):
                if phase_index == 1:
                    model.step(stream[0])
                    with tempfile.TemporaryDirectory() as directory:
                        checkpoint = Path(directory) / 'checkpoint.npz'
                        model.save(checkpoint)
                        resumed = Learner.load(checkpoint)
                        assert resumed.digest() == model.digest()
                    loss, cpu = local_train(model, stream[1:])
                    local_train(resumed, stream[1:])
                    assert resumed.digest() == model.digest()
                    timing.append((loss, cpu, len(stream) - 1))
                else:
                    loss, cpu = local_train(model, stream)
                    timing.append((loss, cpu, len(stream)))
                    if phase_index == 2:
                        local_train(resumed, stream)
                        assert resumed.digest() == model.digest()
                all_rows.append({'seed': seed, 'arm': arm, 'phase': phase_index, 'sites': model.sites,
                                 'writes': model.writes, 'updates': model.updates,
                                 'state_sha256': model.digest(), 'outputs': score(model, held)})
            assert model.writes > 0 and model.updates > 0 and np.isfinite(model.theta).all()
            all_rows[-1]['training_timing'] = timing
        for route in (True, False):
            reference = Learner(seed, route)
            start = time.process_time()
            loss = sum(train_block(reference, data[0][i:i + 32]) for i in range(0, len(data[0]), 32))
            cpu = time.process_time() - start
            all_rows.append({'seed': seed, 'arm': 'R32-BP' if route else 'O32-BP', 'phase': 0,
                             'sites': reference.sites, 'updates': reference.updates, 'loss_nats': loss,
                             'training_timing': [(loss, cpu, reference.sites)], 'outputs': score(reference, held)})
    # A controlled native head fixture exercises EOS generation independently of immature training.
    eos_fixture = Learner(66003)
    eos_fixture.head.fill(0)
    eos_fixture.head[EOS, -1] = 1
    assert eos_fixture.probe(b'19+22=') == [EOS]
    assert not score(eos_fixture, [(b'19+22=', b'41')])[0]['exact']
    profile = profile_training(arms)
    save_result(3, {'status': 'PASS_REDUCED_REHEARSAL', 'profile': profile, 'development_layouts': layouts, 'rows': all_rows,
                    'checkpoint_resume': 'Bit-identical through A prime after a mid-update checkpoint inside B for all 12 local histories',
                    'native_eos_and_early_stop_rejection': True, 'numpy': np.__version__,
                    'platform': platform.platform(), 'process_peak_rss_bytes_on_mac': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'full_roster_training': 'NOT_RUN'})


def profile_training(arms):
    from reference import train_block
    # Full-layout development data, fixed early prefix and a separate eight-query panel.
    world = make_world(SEEDS[0])
    stream = tokens(world['phases'][0])[:2048]
    from worlds import addition
    panel = [addition(*item[2]) for item in world['primary'][:8]]
    result = []
    for name, route, horizon in arms + [('R32-BP', True, 32), ('O32-BP', False, 32)]:
        model = Learner(SEEDS[0], route, min(horizon, 8))
        start = time.process_time()
        if horizon == 32:
            loss = sum(train_block(model, stream[i:i + 32]) for i in range(0, len(stream), 32))
        else:
            loss = sum(model.step(y) for y in stream)
        cpu = time.process_time() - start
        assert model.sites == len(stream) and model.updates == len(stream) // 32
        assert np.isfinite(loss)
        start = time.process_time()
        outcomes = score(model, panel)
        generation_cpu = time.process_time() - start
        parameter_arrays = {'theta', 'head', 'memory', 'h', 'fe'}
        persistent_bytes = sum(v.nbytes for k, v in model.arrays().items() if k in parameter_arrays)
        result.append({'arm': name, 'sites': len(stream), 'cpu_seconds': cpu,
                       'sites_per_cpu_second': len(stream) / cpu,
                       'mean_loss_nats': loss / len(stream),
                       'probe_cpu_seconds_8_queries': generation_cpu,
                       'probe_exact_8_queries': sum(o['exact'] for o in outcomes),
                       'all_model_numpy_arrays_bytes': sum(v.nbytes for v in model.arrays().values()),
                       'persistent_core_arrays_bytes': persistent_bytes,
                       'sensor_assets_in_ram_bytes': sum(v.nbytes for v in model.sensor.arrays)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cycle', choices=('1', '2', '3', 'all'))
    args = parser.parse_args()
    for number, fn in enumerate((cycle1, cycle2, cycle3), 1):
        if args.cycle in ('all', str(number)):
            fn()


if __name__ == '__main__':
    main()
