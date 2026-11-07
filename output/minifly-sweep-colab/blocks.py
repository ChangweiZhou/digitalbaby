"""Portable pilot and development runners for the MiniFly sweep."""
from itertools import islice
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def panel_hash(values):
    from hashlib import sha256
    return sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()


def cortical(arm, seed, profile):
    import mod97_local as m

    dose, pair_count = {'smoke': (8, 8), 'pilot': (512, 32), 'screen': (12288, 64)}[profile]
    fit, monitor, held = m.panels(seed)
    canonical = lambda rows: {tuple(sorted(row)) for row in rows}
    assert not canonical(fit) & canonical(held)
    assert not canonical(monitor) & canonical(held)
    groups = np.asarray(sorted({(a, b) for a, b in held if a < b}))
    rng = np.random.default_rng(seed + 106)
    wrap_groups = groups[groups.sum(axis=1) >= 97]
    ordinary_groups = groups[groups.sum(axis=1) < 97]
    pair_set = np.concatenate([g[rng.choice(len(g), pair_count // 2, replace=False)]
                               for g in (wrap_groups, ordinary_groups)])
    pairs = [(int(a), int(b)) for a, b in pair_set]
    net, feedback, optimizer = m.make(seed, arm)
    update = m.update_bp if arm == 'bp' else m.update_dfa
    loss = [update(net, feedback, optimizer, rows) for rows in islice(m.stream(fit, seed), dose)]
    prompts = [m.arithmetic(p) for a, b in pairs for p in ((a, b), (b, a))]
    net.eval()
    out = m.respond(net, [row[:6] for row in prompts])
    correct = [actual == row[6:] for actual, row in zip(out, prompts)]
    return dict(exact_pairs=sum(x and y for x, y in zip(correct[::2], correct[1::2])),
                n=pair_count, exact_answers=sum(correct), wrap_exact_pairs=sum(
                    correct[2*i] and correct[2*i+1] for i, (a, b) in enumerate(pairs) if a+b >= 97),
                carry_exact_pairs=sum(correct[2*i] and correct[2*i+1] for i, (a, b) in enumerate(pairs)
                                      if a % 10 + b % 10 >= 10),
                carry_pairs=sum(a % 10 + b % 10 >= 10 for a, b in pairs),
                updates=dose, target_bytes=dose * m.BATCH * len(prompts[0]),
                params=sum(p.numel() for p in net.parameters()), last_loss=loss[-1],
                held_sha256=panel_hash(prompts), outputs=out, held=pairs)


def attention(arm, seed, profile):
    import torch as t
    from torch import nn

    dose, count = {'smoke': (8, 8), 'pilot': (512, 32), 'screen': (12288, 64)}[profile]
    t.set_num_threads(1)
    t.manual_seed(seed)
    rng = np.random.default_rng(seed + 37)
    # Keys A and B, query A or B, and two distinct printable values.
    groups = np.array([(a, b) for a in range(97, 123) for b in range(a + 1, 123)])
    rng.shuffle(groups)
    held_groups, train_groups = groups[:64], groups[64:]
    assert not set(map(tuple, held_groups)) & set(map(tuple, train_groups))

    def render(groups, query, flip):
        a, b = groups.T
        x = np.column_stack((65 + query, np.where(flip, 66, 65), np.where(flip, b, a),
                            np.where(flip, 65, 66), np.where(flip, a, b)))
        return t.from_numpy(x), t.from_numpy(np.where(query, b, a))

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(257, 32)
            self.features = nn.Linear(64, 64)
            self.query = nn.Linear(32, 64)
            self.gate = nn.Linear(128, 64)
            self.output = nn.Sequential(nn.Linear(160, 128), nn.Tanh(), nn.Linear(128, 257))

        def forward(self, inputs, previous):
            context = self.query(self.embed(inputs[:, 0]))
            features = self.features(self.embed(inputs[:, 1:]).reshape(-1, 2, 64)).tanh()
            if arm != 'register':
                # Give pooling and attention an item projection of the same size as the write gate.
                features = features + self.gate(t.cat((features, features.roll(1, -1)), -1)).tanh()
            if arm == 'attention':
                weight = (features * context[:, None]).sum(-1).div(8).softmax(1)
                state = (features * weight[:, :, None]).sum(1)
            elif arm == 'register':
                state = t.zeros_like(context)
                for item in features.unbind(1):
                    gain = self.gate(t.cat((context, item), 1)).sigmoid()
                    state = t.lerp(state, item, gain)
            else:
                state = features.mean(1)
            return self.output(t.cat((context, state, self.embed(previous)), 1))

    net = Reader()
    trainable = list(net.parameters())
    opt = t.optim.AdamW(trainable, lr=.003)
    for _ in range(dose):
        pair = train_groups[rng.integers(len(train_groups), size=64)]
        pair = np.where(rng.integers(2, size=(64, 1)), pair, pair[:, ::-1])
        inputs, target = render(pair, rng.integers(2, size=64), rng.integers(2, size=64))
        start = t.full_like(target, 61)
        logits = net(inputs.repeat(2, 1), t.cat((start, target)))
        loss = nn.functional.cross_entropy(logits, t.cat((target, t.full_like(target, 256))))
        opt.zero_grad()
        loss.backward()
        opt.step()
    pairs = np.repeat(held_groups[:count], 2, axis=0)
    query = np.tile([0, 1], count)
    inputs, target = render(pairs, query, np.repeat(rng.integers(2, size=count), 2))
    with t.no_grad():
        answer = net(inputs, t.full_like(target, 61)).argmax(1)
        eos = net(inputs, answer).argmax(1)
    exact = (answer == target) & (eos == 256)
    return dict(exact_pairs=int(exact.reshape(-1, 2).all(1).sum()), n=count,
                exact_answers=int(exact.sum()), updates=dose, target_bytes=dose * 64 * 2,
                stored_params=sum(p.numel() for p in net.parameters()), mutable_params=sum(p.numel() for p in trainable),
                held_sha256=panel_hash(inputs.tolist()), held=inputs.tolist(),
                outputs=t.stack((answer, eos), 1).tolist(), last_loss=float(loss.detach()),
                scope='supervised reference block')


def central_complex(arm, seed, profile):
    sys.path.insert(0, str(ROOT / 'cx'))
    from qualify_dynamics import Compass, FILES, Interface, Steering, bump, wrap

    root = ROOT / 'cx'
    with np.load(root / FILES[0], allow_pickle=False) as source:
        graph = dict(source)
    model = Interface(graph, *(json.loads((root / name).read_text()) for name in FILES[1:]))
    prior = json.loads((root / 'DYNAMICS_QUALIFICATION.json').read_text())
    graph_arm = 'rewired' if arm == 'rewired' else 'native'
    compass = Compass(model, 'observed_contacts', graph_arm)
    controller = Steering(model, graph_arm)
    turn = prior['compass_separate_timescales']['observed_contacts'][graph_arm]
    steering = prior['steering'][graph_arm]['params']
    count, steps = {'smoke': (8, 16), 'pilot': (32, 80), 'screen': (64, 80)}[profile]
    rng = np.random.default_rng(seed + 31)
    starts = rng.uniform(-np.pi, np.pi, count)
    offsets = rng.uniform(.6, 2.1, count) * np.resize([-1, 1], count)
    noise = rng.normal(0, np.deg2rad(2), size=(steps, count))
    goals = wrap(starts + offsets)
    e = bump(model.phase, starts)
    position = starts.copy()
    traces = []
    for step in range(steps):
        x, y = (e @ model.observer.T).T
        heading = np.arctan2(y, x)
        yaw = np.clip(.14 * controller.command(heading, goals, steering), -.054, .054)
        position += yaw
        e = compass.step(e, yaw, turn['params'], 'opponent', turn['rest_rate'])
        if (step + 1) % 8 == 0:
            # A stationary beacon at zero phase yields an egocentric observation.
            observed_bearing = wrap(-position + noise[step])
            if arm == 'displaced':
                observed_bearing += np.pi / 2
            e = .75 * e + .25 * bump(model.phase, -observed_bearing)
        traces.append(position.tolist())
    error = np.rad2deg(abs(wrap(goals - position)))
    x, y = (e @ model.observer.T).T
    compass_error = np.rad2deg(abs(wrap(np.arctan2(y, x) - position)))
    return dict(n=count, within_20_deg=int((error <= 20).sum()), median_error_deg=float(np.median(error)),
                compass_error_deg=float(np.median(compass_error)), steps=steps, target_bytes=0, updates=0,
                held_sha256=panel_hash([starts.tolist(), goals.tolist(), noise.tolist()]),
                goals=goals.tolist(), starts=starts.tolist(), trace=traces,
                scope='fixed v630 weights and prior calibration, engineered landmark cue, supplied angular goal')
