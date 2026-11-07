"""Eight compact MiniFly development blocks and their matched controls."""
from itertools import product
from pathlib import Path

import numpy as np

from blocks import panel_hash

ROOT = Path(__file__).resolve().parent


def dose(profile, smoke, pilot, screen):
    return dict(smoke=smoke, pilot=pilot, screen=screen)[profile]


def logistic(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def cerebellar(arm, seed, profile):
    """Plastic lag features with local observed-event errors and a supervised readout."""
    rng = np.random.default_rng(seed)
    init = np.random.default_rng(seed + 71)
    updates = dose(profile, 8, 512, 12288)
    count = dose(profile, 8, 32, 64)
    # Each item is a two-cue history. In the held set a cue appears at a new lag.
    groups = np.array(list(product(range(32), range(8))))
    rng.shuffle(groups)
    held, train = groups[:64], groups[64:]
    channels = 34
    same = init.normal(0, .08, (8 * channels, 128))
    masks = np.zeros((8, 128), dtype=bool)
    for i in range(128):
        masks[i % 8, i] = True
        masks[(i + 1) % 8, i] = True
    if arm == 'shuffled_lags':
        masks[np.arange(8)] = masks[np.array([0, 4, 2, 6, 1, 5, 3, 7])]
    if arm == 'ordinary':
        masks[:] = False
        for i in range(128):
            masks[i % 8, i] = True
            masks[(i + 4) % 8, i] = True
    mask = np.repeat(masks, channels, axis=0)
    w = same * mask
    read = init.normal(0, .02, (128, 8))

    def make(pairs, other):
        cue, lag = pairs.T
        x = np.eye(channels, dtype=np.float32)[other]
        rows = np.arange(len(x))
        x[rows, lag] = np.eye(channels)[cue]
        x[rows, lag, 32] = 1
        next_lag = (lag + 1) % 8
        x[rows, next_lag] = np.eye(channels)[(cue + 1) % 32]
        x[rows, next_lag, 33] = 1
        return x.reshape(len(x), -1), (cue + lag) % 8

    held_x, target = make(held[:count], rng.integers(32, size=(count, 8)))
    for _ in range(updates):
        x, y = make(train[rng.integers(len(train), size=64)], rng.integers(32, size=(64, 8)))
        h = np.tanh(x @ w)
        # The feature teacher is the ordinary observation at its preferred lag.
        observed = x.reshape(-1, 8, channels)[:, np.arange(128) % 8, np.arange(128) % channels]
        w += .015 * (x.T @ ((observed - h) * (1 - h*h))) / len(x) * mask
        z = h @ read
        p = np.exp(z-z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        read -= .06 * h.T @ p / len(x)
    pred = (np.tanh(held_x @ w) @ read).argmax(1)
    return dict(exact_pairs=int((pred == target).sum()), n=count, updates=updates,
                target_bytes=updates * 64, params=int(mask.sum()+read.size),
                outputs=pred.tolist(), target=target.tolist(),
                held_sha256=panel_hash([held_x.tolist(), target.tolist()]),
                scope='plastic temporal expansion with local observation targets')


def thalamocortical(arm, seed, profile):
    """Order-sensitive byte composition with matched recurrent and feedforward passes."""
    import torch as t
    from torch import nn

    t.set_num_threads(1)
    t.manual_seed(seed)
    rng = np.random.default_rng(seed)
    updates = dose(profile, 8, 512, 12288)
    count = dose(profile, 8, 32, 64)
    ops = np.stack([rng.permutation(32) for _ in range(5)])
    # Every individual operator is present in teaching. Remove both orders of the held templates.
    templates = list(product(range(5), repeat=3))
    keys = [(0, 1, 2), (0, 3, 4), (1, 2, 4), (1, 3, 0)]
    banned = set(keys) | {k[::-1] for k in keys}
    train = np.asarray([(a, *p) for a in range(32)
                        for p in templates + [(5, 5, op) for op in range(5)] if p not in banned])
    ops = np.vstack((ops, np.arange(32)))

    def target(rows):
        answer = rows[:, 0]
        for col in range(1, 4):
            answer = ops[rows[:, col], answer]
        return answer

    choices = []
    for key in keys:
        for state in range(32):
            a, b = (state, *key), (state, *key[::-1])
            if len(set(target(np.asarray([a, b])))) == 2:
                choices.append((a, b))
    rng.shuffle(choices)
    held = np.asarray(choices[:count]).reshape(-1, 4)
    assert len(held) == 2*count and not set(map(tuple, held)) & set(map(tuple, train))

    class Loop(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(257, 16)
            self.input = nn.Linear(16, 96)
            self.to_relay = nn.Linear(96, 96)
            self.from_relay = nn.Linear(96, 96)
            self.out = nn.Sequential(nn.Linear(112, 128), nn.Tanh(), nn.Linear(128, 257))

        def forward(self, rows, previous, delay=None):
            tokens = rows + t.tensor([65, 97, 97, 97])
            if delay is not None:
                tokens = t.cat((tokens[:, :1], delay, tokens[:, 1:]), 1)
            inputs = self.input(self.embed(tokens)).tanh()
            state = t.zeros_like(inputs[:, 0])
            for inp in inputs.unbind(1):
                state = state + inp
                for _ in range(1 if arm == 'one_pass' else 3):
                    feedback = self.to_relay(state)
                    relay = (feedback + inp if arm != 'feedforward' else feedback).tanh()
                    state = self.from_relay(relay).tanh()
            return self.out(t.cat((state, self.embed(previous)), 1))

    net = Loop()
    opt = t.optim.AdamW(net.parameters(), lr=.002)
    for _ in range(updates):
        rows = train[rng.integers(len(train), size=64)]
        x, y = t.tensor(rows), t.tensor(target(rows)+65)
        start = t.full_like(y, 61)
        logits = net(x.repeat(2, 1), t.cat((start, y)))
        loss = nn.functional.cross_entropy(logits, t.cat((y, t.full_like(y, 256))))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with t.no_grad():
        x, y = t.tensor(held), t.tensor(target(held)+65)
        answer = net(x, t.full_like(y, 61)).argmax(1)
        eos = net(x, answer).argmax(1)
        exact = (answer == y) & (eos == 256)
        delay = t.tensor(np.repeat(rng.integers(113, 118, (count, 6)), 2, axis=0))
        delayed_answer = net(x, t.full_like(y, 61), delay).argmax(1)
        delayed_eos = net(x, delayed_answer, delay).argmax(1)
        delayed_exact = (delayed_answer == y) & (delayed_eos == 256)
    return dict(exact_pairs=int(exact.reshape(-1, 2).all(1).sum()), n=count, updates=updates,
                target_bytes=updates*64*2, params=sum(p.numel() for p in net.parameters()),
                passes=1 if arm == 'one_pass' else 3, outputs=t.stack((answer, eos), 1).tolist(),
                delayed_exact_pairs=int(delayed_exact.reshape(-1, 2).all(1).sum()),
                delayed_outputs=t.stack((delayed_answer, delayed_eos), 1).tolist(),
                target=y.tolist(), held_sha256=panel_hash([held.tolist(), ops.tolist(), delay.tolist()]),
                scope='supervised compact two-population composition probe')


def reward_panels(seed, profile):
    rng = np.random.default_rng(seed)
    count = dose(profile, 8, 32, 64)
    episodes = dose(profile, 32, 512, 4096)
    mapping = rng.integers(2)
    # Each context is followed by thirteen training or twenty held distractors.
    context = rng.integers(2, size=episodes)
    train = rng.integers(2, 4, (episodes, 14))
    train[:, 0] = context
    held = np.repeat(rng.integers(2, 4, (count, 21)), 2, axis=0)
    held[:, 0] = np.tile([0, 1], count)
    return int(mapping), train, held


def working_memory_gate(arm, seed, profile):
    """Rewarded write policy with a frozen state representation and a yoked trace."""
    mapping, train, held = reward_panels(seed, profile)
    rng = np.random.default_rng(seed + 106)
    # A and C encode one sign, B and D the other. Relevance comes from the observed token.
    token_value = np.array([-1, 1, -1, 1])

    def history(rows, mode, random, tape=None):
        gate = np.zeros(4)
        readout = np.zeros(2)
        baseline = .5
        gates, rewards = [], []
        for j, row in enumerate(rows):
            ps = logistic(gate[row])
            draws = random.random(len(row))
            if mode == 'ungated':
                writes = np.ones(len(row), dtype=bool)
            elif mode == 'yoked':
                writes = tape[j]
            else:
                writes = draws < ps
            gates.append(writes)
            last = np.flatnonzero(writes)
            state = token_value[row[last[-1]]] if len(last) else 0
            feature = np.array([1, state])
            action_p = logistic(readout @ feature)
            action = int(random.random() < action_p)
            reward = float(action == (row[0] ^ mapping))
            delta = reward-baseline
            baseline += .025*delta
            readout += .12*delta*(action-action_p)*feature
            if mode == 'learned':
                np.add.at(gate, row, .07*delta*(writes-ps))
            rewards.append(reward)
        return gate, readout, np.asarray(gates), rewards

    independent = np.random.default_rng(seed + 201)
    tape, shadow_gate = None, None
    if arm == 'yoked':
        shadow = train.copy()
        shadow[:, 0] = independent.integers(2, size=len(shadow))
        shadow_gate, _, tape, _ = history(shadow, 'learned', independent)
    gate, readout, writes, rewards = history(train, arm, rng, tape)
    predicted, targets = [], []
    yoke_masks = None
    if shadow_gate is not None:
        shadow_held = held[::2].copy()
        shadow_held[:, 0] = independent.integers(2, size=len(shadow_held))
        yoke_masks = np.repeat(independent.random(shadow_held.shape) < logistic(shadow_gate[shadow_held]), 2, axis=0)
    for i, row in enumerate(held):
        if arm == 'ungated':
            indices = np.arange(len(row))
        elif arm == 'yoked':
            indices = np.flatnonzero(yoke_masks[i])
        else:
            indices = np.flatnonzero(logistic(gate[row]) >= .5)
        state = token_value[row[indices[-1]]] if len(indices) else 0
        predicted.append(int(readout @ np.array([1, state]) >= 0))
        targets.append(int(row[0] ^ mapping))
    ok = np.asarray(predicted) == targets
    reset_action = int(readout[0] >= 0)
    return dict(exact_pairs=int(ok.reshape(-1, 2).all(1).sum()), n=len(held)//2, updates=len(train),
                reward_bits=len(train), auxiliary_yoke_episodes=len(train) if tape is not None else 0,
                writes=int(writes.sum()), reward_mean=float(np.mean(rewards)),
                reset_exact_pairs=int((np.asarray(targets)==reset_action).reshape(-1, 2).all(1).sum()),
                outputs=predicted, target=targets, held_sha256=panel_hash([mapping, held.tolist()]),
                scope='compact scalar-outcome register gate')


def actor_critic(arm, seed, profile):
    mapping, train, held = reward_panels(seed, profile)
    rng = np.random.default_rng(seed + 63)
    actor = np.zeros(4)
    critic = np.zeros(4)
    rewards, nonzero = [], 0
    for row in train:
        # A causal choice on the cue, then distractor actions. Only the earlier action is scored.
        e = np.zeros(4)
        trace = 0.
        initial = 0
        for j, token in enumerate(row):
            p = logistic(actor[token])
            action = int(rng.random() < p)
            e *= .93
            if j == 0:
                initial = action
                e[token] += action-p
            trace = action-p
        if arm == 'immediate':
            e[:] = 0
            e[row[-1]] = trace
        elif arm == 'shifted':
            # Keep the magnitude and move it to an irrelevant observation.
            e[row[-1]] = e[row[0]]
            e[row[0]] = 0
        reward = float(initial == (int(row[0]) ^ mapping))
        err = reward-critic[row[0]]
        critic[row[0]] += .06*err
        actor += .25*err*e
        nonzero += np.count_nonzero(err*e)
        rewards.append(reward)
    target = held[:, 0] ^ mapping
    actions = (actor[held[:, 0]] >= 0).astype(int)
    return dict(exact_pairs=int((actions==target).reshape(-1, 2).all(1).sum()), n=len(held)//2,
                updates=len(train), reward_bits=len(train), writes=int(nonzero),
                reward_mean=float(np.mean(rewards)), value=critic.tolist(), outputs=actions.tolist(),
                target=target.tolist(), held_sha256=panel_hash([mapping, held.tolist()]),
                scope='compact actor–critic delayed scalar-outcome probe')


def continual_fixture(seed, profile):
    rng = np.random.default_rng(seed)
    n = dose(profile, 8, 32, 64)
    feature = rng.normal(size=(3*n, 64))
    feature /= np.linalg.norm(feature, axis=1, keepdims=True)
    target = rng.integers(2, size=3*n)
    a, b, edit = np.arange(2*n), np.arange(2*n, 3*n), np.arange(n, 2*n)
    counts = dose(profile, (32, 16, 8), (256, 128, 64), (1024, 256, 128))
    streams = [np.tile(group, (count+len(group)-1)//len(group))[:count].copy()
               for group, count in zip((a, b, edit), counts)]
    for stream in streams:
        rng.shuffle(stream)
    return n, feature, target, streams


def retention_result(n, target, before, after, final, extra):
    prior = before[:n] == target[:n]
    old = after[:n] == target[:n]
    return dict(exact_pairs=int(old.sum()), n=n, a_acquired=int((before[:2*n]==target[:2*n]).sum()),
                retained_previously_correct=int((prior & old).sum()), previously_correct=int(prior.sum()),
                old_after_correction=int((final[:n]==target[:n]).sum()),
                b_after_correction=int((final[2*n:]==target[2*n:]).sum()),
                corrected=int((final[n:2*n]!=target[n:2*n]).sum()),
                target=target.tolist(), old_pre=before.tolist(), after_b=after.tolist(),
                after_correction=final.tolist(), no_write_old=int(prior.sum()), **extra)


def retention(arm, seed, profile, mechanism):
    n, x, y, (a, b, edit) = continual_fixture(seed, profile)
    w = np.zeros(x.shape[1])
    fast, slow = w[:32], w[32:]
    dual = mechanism == 'fast_slow' and arm != 'single'
    writes = 0

    def read():
        return (x @ w >= 0).astype(int)

    def write(i, target, replay=False):
        nonlocal writes
        if dual:
            xf, xs = x[i, :32], x[i, 32:]
            if replay:
                teaching = logistic(xf @ fast) if arm == 'coupled' else target
                slow[:] += .35*(teaching-logistic(xs@slow))*xs
            else:
                fast[:] += 2*(target-logistic(xf@fast))*xf
                slow[:] += .035*(target-logistic(xs@slow))*xs
        else:
            w[:] += (2 if not replay else .35)*(target-logistic(x[i]@w))*x[i]
        writes += 1

    for i in a:
        write(i, y[i])
    before = read()
    # Same chronological capped observations, never accessed for inference.
    replay = a[-min(128, len(a)):].copy()
    buckets = np.array_split(replay, len(b))
    for j, i in enumerate(b):
        write(i, y[i])
        if mechanism == 'fast_slow' or arm == 'interleaved':
            for old in buckets[j]:
                write(old, y[old], replay=True)
    if mechanism == 'sleep_replay' and arm == 'blocked':
        for old in replay:
            write(old, y[old], replay=True)
    if mechanism == 'sleep_replay':
        w *= .99
    after = read()
    for i in edit:
        write(i, 1-y[i])
    final = read()
    return retention_result(n, y, before, after, final,
                            dict(updates=writes, target_bytes=len(a)+len(b)+len(edit),
                                 replay_updates=writes-len(a)-len(b)-len(edit), params=len(w),
                                 held_sha256=panel_hash([x.tolist(), y.tolist()]), scope=mechanism))


def mushroom_body(arm, seed, profile):
    """Use observed PN/KC, KC/MBON, DAN/KC and MBON/DAN contacts."""
    from sys import path
    path.insert(0, str(ROOT / 'cx'))
    from qualify import Local
    with np.load(ROOT / 'cx' / 'EXTRACT.npz', allow_pickle=False) as data:
        g = dict(data)
    local = Local(g)
    rng = np.random.default_rng(seed + 64)
    n, _, targets, (a, b, edit) = continual_fixture(seed, profile)
    nodes = g['roles']
    u, v = g['u'], g['v']
    dan = np.flatnonzero(nodes == 'DAN')
    mb = local.mbons
    relay = g['usable'] & (nodes[u]=='MBON') & (nodes[v]=='DAN')
    assert relay.any()
    loop = np.zeros((len(dan), len(mb)))
    np.add.at(loop, (np.searchsorted(dan, v[relay]), np.searchsorted(mb, u[relay])),
              g['sign'][relay]*g['w'][relay])
    loop /= np.abs(loop).sum(1, keepdims=True).clip(1)
    da_kc = (nodes[u]=='DAN') & (nodes[v]=='KC')
    address = np.zeros((len(nodes), len(dan)))
    address[v[da_kc], np.searchsorted(dan, u[da_kc])] = 1
    address /= address.sum(1, keepdims=True).clip(1)
    if arm == 'permuted':
        address = address[:, rng.permutation(len(dan))]
    edge_address = address[local.u]
    odor = np.asarray([local.observe(seed*7+i)[0][local.u] for i in range(3*n)])
    groups = np.arange(len(mb)) % 2
    idx = np.searchsorted(mb, local.v)
    w = local.base.copy()
    rewards, writes = [], 0

    def response(i):
        y = np.bincount(idx, weights=w*odor[i], minlength=len(mb))/local.post_norm[mb]
        score = np.bincount(groups, weights=y)
        return y, logistic(30*(score[1]-score[0]))

    def read():
        return np.asarray([int(response(i)[1]>=.5) for i in range(3*n)])

    def train(stream, edited=False):
        nonlocal writes
        for i in stream:
            y, prob = response(i)
            action = int(rng.random() < prob)
            target = int(targets[i]) ^ edited
            reward = float(action == target)
            rewards.append(reward)
            gain = 1 if arm == 'open' else (1+.5*np.tanh(4*loop@y))
            mod = edge_address @ np.broadcast_to(gain, (len(dan),))
            elig = odor[i] * (2*groups[idx]-1) * (action-prob)
            delta = .75*(reward-.5)*elig*mod
            w[:] = np.clip(w+delta, .5*local.base, 1.5*local.base)
            writes += int(np.count_nonzero(delta))

    train(a)
    before = read()
    train(b)
    after = read()
    train(edit, True)
    return retention_result(n, targets, before, after, read(),
                            dict(reward_bits=len(rewards), reward_mean=float(np.mean(rewards)),
                                 updates=len(rewards), writes=writes, params=len(w), observed_loop_edges=int(relay.sum()),
                                 held_sha256=panel_hash([seed, targets.tolist()]),
                                 scope='v630 contact-based MBON–DAN gain analogue with scalar outcomes'))


def grid_head_direction(arm, seed, profile):
    """Fixed toroidal transport with or without attractor sharpening."""
    rng = np.random.default_rng(seed)
    n = dose(profile, 8, 32, 64)
    steps = dose(profile, 32, 80, 160)
    # Even the maximum allowed path fits within the unambiguous half-period.
    bins, span = 16, 8.
    angles = 2*np.pi*np.arange(bins)/bins
    kernel = np.exp(5*(np.cos(angles)-1))
    kernel /= kernel.sum()
    xgrid = np.exp(1j*angles)
    true_yaw = rng.uniform(-.12, .12, (steps,n))
    true_distance = rng.uniform(.012,.022,(steps,n))
    pairs = (n-max(1,n//8))//2
    true_yaw[:, pairs:2*pairs] = true_yaw[::-1, :pairs]
    true_distance[:, pairs:2*pairs] = true_distance[::-1, :pairs]
    true_yaw[:, -max(1,n//8):] = 0
    true_distance[:, -max(1,n//8):] = 0
    observed_yaw = true_yaw+rng.normal(0,.004,true_yaw.shape)
    observed_distance = true_distance*(1+rng.normal(0,.04,true_distance.shape))
    state = np.tile(kernel,(n,1))
    grid = np.tile(np.outer(kernel,kernel),(n,1,1))
    true_heading = np.zeros(n)
    position = np.zeros((n,2))
    indices = np.arange(bins)
    path_len = true_distance.sum(0)

    def shift(values, offset):
        low = np.floor(offset).astype(int)
        f = (offset-low)[:,None]
        a = (indices-low[:,None]) % bins
        b = (a-1)%bins
        row = np.arange(len(values))[:, None]
        return (1-f)*values[row,a]+f*values[row,b]

    def diffuse(values):
        return .7*values+.15*(np.roll(values,1,axis=1)+np.roll(values,-1,axis=1))

    def sharpen(values):
        shape = diffuse(values)
        if arm == 'attractor':
            shape = shape**1.10
        return shape / shape.sum(axis=1,keepdims=True).clip(1e-12)

    for yaw, distance, oy, od in zip(true_yaw,true_distance,observed_yaw,observed_distance):
        true_heading += yaw
        position += distance[:,None]*np.stack((np.cos(true_heading),np.sin(true_heading)),1)
        if arm == 'reset':
            state[:] = kernel
            grid[:] = np.outer(kernel,kernel)
        state = sharpen(shift(state, oy*bins/(2*np.pi)))
        heading = np.angle(state@xgrid)
        # Operate on the two-dimensional population state, not on a position accumulator.
        for axis, amount in [(1, od*np.cos(heading)), (2, od*np.sin(heading))]:
            moved = np.moveaxis(grid, axis, -1)
            plane = moved.reshape(n*bins,bins)
            plane = shift(plane, np.repeat(amount*bins/span,bins))
            grid = np.moveaxis(plane.reshape(moved.shape),-1,axis)
        if arm == 'attractor':
            grid = grid**1.045
        grid /= grid.sum(axis=(1,2),keepdims=True).clip(1e-12)
    decoded = np.stack((np.angle(grid.sum(axis=2)@xgrid),np.angle(grid.sum(axis=1)@xgrid)),1)*span/(2*np.pi)
    diff = decoded-position
    error = np.linalg.norm(diff,axis=1)
    moving = path_len >= .25
    return dict(exact_pairs=int(np.sum(moving & (error <= .2*path_len))), n=int(moving.sum()),
                median_normalized_error=float(np.median(error[moving]/path_len[moving])),
                order_pairs=pairs, median_order_separation=float(np.median(
                    np.linalg.norm(decoded[:pairs]-decoded[pairs:2*pairs], axis=1))),
                zero_motion_drift=float(np.median(error[path_len==0])), updates=0, steps=steps,
                target_bytes=0, state_size=bins*bins+bins, outputs=decoded.tolist(), target=position.tolist(),
                held_sha256=panel_hash([true_yaw.tolist(),true_distance.tolist(),
                                        observed_yaw.tolist(),observed_distance.tolist()]),
                scope='engineered toroidal transport, attractor sharpening versus diffuse and reset controls')
