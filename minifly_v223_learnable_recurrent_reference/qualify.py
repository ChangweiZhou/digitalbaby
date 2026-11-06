"""Three engineering cycles on known development data."""
import copy
import json
from pathlib import Path
import random
import resource
import tempfile
import time

import numpy as np
import torch

from learner import EOS, Learner
from reference import gradient
from worlds import addition, fact, make_world, tokens, exposure_digest

ROOT = Path(__file__).resolve().parent
SEED = 66101


def save(name, result):
    (ROOT / name).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(name, result['status'], flush=True)


def score(model, panel):
    before = model.digest()
    rows = []
    for prompt, answer in panel:
        predicted = model.probe(prompt)
        rows.append({'prompt': prompt.decode(), 'expected': [*answer, EOS], 'output': predicted,
                     'exact': predicted == [*answer, EOS], 'own_eos': predicted[-1] == EOS})
    assert model.digest() == before
    return rows


def cycle1():
    model = Learner(SEED, 64, 'dense', dtype=np.float64)
    rng = np.random.default_rng(4)
    model.h[:] = rng.normal(0, .3, 64)
    cell = torch.nn.GRUCell(24, 64, dtype=torch.float64)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.from_numpy(model.cell[:, :, :24].reshape(192, 24)))
        cell.weight_hh.copy_(torch.from_numpy(model.cell[:, :, 24:-2].reshape(192, 64)))
        cell.bias_ih.copy_(torch.from_numpy(model.cell[:, :, -2].reshape(192)))
        cell.bias_hh.copy_(torch.from_numpy(model.cell[:, :, -1].reshape(192)))
        expected = cell(torch.from_numpy(model.embedding[ord('4')]), torch.from_numpy(model.h)).numpy()
    model.transition(model.embedding[ord('4')])
    delta = float(np.max(np.abs(expected - model.h)))
    assert delta < 1e-12
    assert len(np.unique(model.embedding, axis=0)) == 257
    inputs = list(b'1+2=3') + [EOS] + list(b'f/a=456') + [EOS]
    _, weights = model.prepare(inputs)
    expected_weights = np.array([1,1,1,1,4,4,1,1,1,1,4,4,4,4])
    assert np.allclose(weights, expected_weights * len(inputs) / expected_weights.sum())
    model.head.fill(0)
    model.head[EOS, -1] = 1
    assert model.probe(b'1+2=') == [EOS]
    assert not score(model, [addition(1, 2)])[0]['exact']
    a = Learner(SEED, 32, 'connected')
    b = Learner(SEED, 32, 'isolated')
    c = Learner(SEED, 32, 'connected', 'kc')
    assert a.cell.shape == b.cell.shape == c.cell.shape
    assert np.array_equal(a.cell, b.cell) and np.array_equal(a.cell, c.cell)
    assert np.array_equal(a.embedding, c.embedding)
    reached = {0}
    for _ in range(32):
        reached.update(i for i, incoming in enumerate(a.incoming) if reached.intersection(incoming))
    assert len(reached) == 32
    assert all(np.all(row // 8 == i // 8) for i, row in enumerate(b.incoming))
    kc_features, _ = c.prepare(inputs)
    assert np.isfinite(kc_features).all()
    save('CYCLE1.json', {'status': 'PASS', 'archived_cell_equation_max_error': delta,
                         'distinct_direct_rows': 257, 'response_weights': weights.tolist(),
                         'connected_reachable': len(reached), 'matched_mask_parameters': int(a.cell.size),
                         'native_eos_and_premature_eos_rejection': True,
                         'kc_sensor_digest': c.sensor.digest})


def cycle2():
    rng = np.random.default_rng(5)
    rows = []
    for topology, route, width in [('dense','direct',64), ('connected','direct',32),
                                   ('isolated','direct',32), ('connected','kc',32),
                                   ('connected','direct',64)]:
        model = Learner(SEED, width, topology, route, np.float64)
        model.h[:] = rng.normal(0, .2, width)
        stream = tokens([addition(48, 65), fact('apple', 870), addition(23, 22), addition(91, 29)])[:32]
        x, weights = model.prepare(stream)
        reference_loss, gc, gh, expected = gradient(model, x, stream, weights)
        fwd = copy.deepcopy(model)
        loss, fc, fh = fwd.forward_gradient(x, stream, weights)
        error = max(np.max(np.abs(gc - fc)), np.max(np.abs(gh - fh)))
        assert error < 1e-11 and np.max(np.abs(expected-fwd.h)) < 1e-12 and abs(loss-reference_loss) < 1e-10
        direction = rng.normal(size=model.cell.shape)
        direction /= np.linalg.norm(direction)
        values = []
        for sign in (-1, 1):
            alt = copy.deepcopy(model)
            alt.cell += sign * 1e-5 * direction
            value = 0.0
            for features, y, w in zip(x, stream, weights):
                logits = alt.logits()
                value += w * (np.logaddexp.reduce(logits) - logits[y])
                alt.transition(features)
            values.append(value)
        finite = abs((values[1] - values[0]) / 2e-5 - np.sum(fc * direction))
        assert finite < 1e-7
        a, b = copy.deepcopy(model), copy.deepcopy(model)
        for _ in range(3):
            a.train(stream, 'forward')
            b.train(stream, 'bp')
        update_error = max(np.max(np.abs(a.cell-b.cell)), np.max(np.abs(a.head-b.head)), np.max(np.abs(a.h-b.h)))
        assert update_error < 1e-11
        rows.append({'topology': topology, 'route': route, 'width': width, 'gradient_max_error': float(error),
                     'directional_error': float(finite), 'three_update_max_error': float(update_error)})
    save('CYCLE2.json', {'status': 'PASS', 'fixtures': rows, 'dtype': 'float64'})


def cycle3():
    world = make_world(SEED)
    panel = [addition(*ab) for ab in world['taught'][:4]]
    panel += [fact(*item) for item in world['assignments'][0][:4]]
    rng = random.Random(6610101)
    records = []
    for _ in range(256):
        batch = panel.copy()
        rng.shuffle(batch)
        records.extend(batch)
    stream = tokens(records)
    rows = []
    for name, width, topology, route in [('dense64',64,'dense','direct'), ('connected32',32,'connected','direct'),
                                         ('connected64',64,'connected','direct'), ('isolated32',32,'isolated','direct'),
                                         ('kc32',32,'connected','kc')]:
        model = Learner(SEED, width, topology, route)
        resumed = None
        clock = time.process_time()
        loss = 0.0
        for dose in (1, 4, 16, 64, 256):
            stop = len(tokens(records[:dose * 8]))
            while model.sites + 32 <= stop:
                block = stream[model.sites:model.sites+32]
                loss += model.train(block, 'bp')
                if resumed is not None:
                    resumed.train(block, 'bp')
            if resumed is not None:
                assert resumed.digest() == model.digest()
                resumed = None
            if dose == 4:
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp)/'state.npz'
                    model.save(path)
                    resumed = Learner.load(path)
                    assert resumed.digest() == model.digest()
            # Score the exact visible prefix on a copy. Updates occur only at full blocks.
            snapshot = copy.deepcopy(model)
            for y in stream[model.sites:stop]:
                snapshot.transition(snapshot.feature(y))
            scored = score(snapshot, panel)
            rows.append({'arm': name, 'presentations': dose, 'symbols': stop, 'updates': model.updates,
                         'cpu_seconds_cumulative_including_probes': time.process_time()-clock,
                         'loss_nats_on_committed_blocks': loss, 'outputs': scored,
                         'exact_arithmetic': sum(x['exact'] for x in scored[:4]),
                         'exact_facts': sum(x['exact'] for x in scored[4:])})
            print(name, dose, rows[-1]['exact_arithmetic'], rows[-1]['exact_facts'], flush=True)
    profiles = []
    for width, topology in [(32,'connected'),(64,'connected'),(64,'dense')]:
        for method in ('forward','bp'):
            model = Learner(SEED, width, topology)
            start = time.process_time()
            for i in range(0, 256, 32):
                model.train(stream[i:i+32], method)
            cpu = time.process_time() - start
            profiles.append({'method': method, 'width': width, 'topology': topology,
                             'symbols': 256, 'cpu_seconds': cpu,
                             'sites_per_cpu_second': 256/cpu,
                             'model_arrays_bytes': sum(a.nbytes for a in model.arrays().values())})
    native = Learner(SEED, 32, 'connected', 'kc')
    for start in range(0, 128, 32):
        native.train(stream[start:start+32], 'forward')
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'forward.npz'
        native.save(path)
        resumed = Learner.load(path)
        assert resumed.digest() == native.digest()
    for start in range(128, 256, 32):
        native.train(stream[start:start+32], 'forward')
        resumed.train(stream[start:start+32], 'forward')
    assert resumed.digest() == native.digest()
    save('CYCLE3.json', {'status': 'PASS_REDUCED_REHEARSAL', 'seed': SEED, 'stream_sha256': exposure_digest(records),
                         'rows': rows, 'profiles': profiles, 'checkpoint_resume': 'Bit-identical for all five BPTT arms and the separate native forward KC fixture',
                         'single_process_peak_rss_bytes_mac': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         'stage1_three_world_screen': 'NOT_RUN', 'lifetime': 'NOT_RUN'})


if __name__ == '__main__':
    torch.set_num_threads(1)
    cycle1()
    cycle2()
    cycle3()
