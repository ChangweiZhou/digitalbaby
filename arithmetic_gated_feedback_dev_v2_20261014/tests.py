"""Cycle-1 generic implementation qualification; not arithmetic science."""
import argparse
import ast
import json
import math
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import Learner, EOS, MODELS
from protocol import ROOT, D, atomic

CHECKS = []


def test(fn):
    CHECKS.append(fn)
    return fn


def near(a, b, atol=2e-12, rtol=2e-11):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def rejects(fn):
    try:
        fn()
    except (ValueError, TypeError, AssertionError):
        return
    raise AssertionError('forbidden input accepted')


@test
def architecture_and_boundary():
    a, b = Learner('KEEP', 301), Learner('G62', 301)
    near(a.p['E'], b.p['E'], atol=0, rtol=0)
    for c in (a, b):
        assert sum(p.numel() for p in c.p.values()) == D['models'][c.model]['parameters']
        assert sum(p.numel() * p.element_size() for p in c.p.values()) * 2 == D['models'][c.model]['parameter_and_RMS_bytes']
        assert 'C' in c.p if c.model == 'KEEP' else 'C' not in c.p
        rejects(lambda: c.encode(b'x' * 257))
        rejects(lambda: c.encode('not bytes'))
    assert len(a.initial()[0]) * 2 == 128
    assert sum(x.numel() for x in b.initial()) == 124
    near(b.p['bf'], torch.ones(62, dtype=torch.float64), atol=0, rtol=0)
    tree = ast.parse((ROOT / 'core.py').read_text())
    imports = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {x.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for x in n.names}
    assert imports <= {'copy', 'hashlib', 'json', 'math', 'time', 'pathlib', 'numpy', 'torch'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('eval', 'exec', '__import__') for n in ast.walk(tree))
    return dict(parameters={c.model: sum(p.numel() for p in c.p.values()) for c in (a, b)}, imports=sorted(imports))


@test
def keep_historical_parity():
    from legacy_keep_reference import Learner as Historical
    new, old = Learner('KEEP', 823), Historical('KEEP', 823)
    for k, p in new.p.items():
        near(p, old.net.p[k], atol=0, rtol=0)
    # Nonzero C and head exercise the historical decoder path, not just birth.
    rng = np.random.default_rng(702)
    with torch.no_grad():
        for k in ('V', 'C'):
            z = torch.from_numpy(rng.normal(0, .03, new.p[k].shape))
            new.p[k].copy_(z)
            old.net.p[k].copy_(z)
    for i, phase in enumerate(('WARM', 'WARM', 'HALF', 'SPARSE', 'SCALAR', 'SCALAR', 'HALF', 'WARM')):
        prompt = bytes((0, i + 1, 255, 61))
        answer = None if phase == 'SCALAR' else bytes((128 + i, 0))
        actual, status, cache = new.act(prompt, i + 919, phase != 'WARM')
        oa, os, oc = old.act(prompt, i + 919, phase != 'WARM')
        assert (actual, status, cache['actions']) == (oa, os, oc['actions'])
        near(cache['baseline'], oc['baseline'], atol=0, rtol=0)
        u = new.update(prompt, answer, i % 2, cache, phase)
        v = old.update(prompt, answer, i % 2, oc, phase)
        for k, p in new.p.items():
            near(p, old.net.p[k])
            near(new.mom[k], old.mom[k])
        for k in ('ce', 'pg', 'critic'):
            assert u[k] is None and v[k] is None or math.isclose(u[k], v[k], abs_tol=1e-12, rel_tol=1e-11)
        assert (new.actor_steps, new.critic_steps) == (old.actor_steps, old.critic_steps)
    assert new.generate_batch([b'', b'\x00\xff', b'abc']) == old.generate_batch([b'', b'\x00\xff', b'abc'])
    return dict(records=8, optimizer_and_sampling_parity=True)


@test
def lstm_independent_cell():
    c = Learner('G62', 127)
    ref = torch.nn.LSTMCell(24, 62, dtype=torch.float64)
    with torch.no_grad():
        for name, prefix in (('weight_ih', 'W'), ('weight_hh', 'U'), ('bias_ih', 'b')):
            getattr(ref, name).copy_(torch.cat([c.p[prefix + k] for k in ('i', 'f', 'g', 'o')], 0))
        ref.bias_hh.zero_()
    state = c.initial(3)
    for t in range(11):
        token = torch.tensor([(t * 11 + j) % 259 for j in range(3)])
        h, m = ref(c.p['E'][token], (state[1], state[0]))
        state = c.step(state, token)
        near(state[0], m)
        near(state[1], h)
    return dict(independent_reference='torch.nn.LSTMCell', steps=11, batch=3)


@test
def objective_gradients():
    errors = []
    for model in MODELS:
        c = Learner(model, 78)
        with torch.no_grad():
            c.p['V'].copy_(torch.from_numpy(np.random.default_rng(18).normal(0, .1, c.p['V'].shape)))
        prompt, answer = b'\x00\xffQ', b'\x80\x00'
        for kind in ('CE', 'PG'):
            _, _, cache = c.act(prompt, 77, True, cap=3)
            baseline = float(cache['baseline'])
            actions = cache['actions']
            def loss():
                if kind == 'CE':
                    return c.ce_loss(prompt, answer)
                logs = c.trace_logps(prompt, actions)
                return -(1 - baseline) * logs[torch.arange(len(actions)), torch.tensor(actions)].sum()
            values = list(c.actor().values())
            grads = torch.autograd.grad(loss(), values)
            for (name, p), g in zip(c.actor().items(), grads):
                idx = int(g.abs().argmax())
                expected = float(g.reshape(-1)[idx])
                old = float(p.detach().reshape(-1)[idx])
                with torch.no_grad():
                    p.reshape(-1)[idx] = old + 1e-5
                    plus = float(loss())
                    p.reshape(-1)[idx] = old - 1e-5
                    minus = float(loss())
                    p.reshape(-1)[idx] = old
                numeric = (plus - minus) / 2e-5
                assert math.isclose(expected, numeric, abs_tol=3e-8, rel_tol=4e-5), (model, kind, name, expected, numeric)
                errors.append(abs(expected - numeric))
            critic_grad = torch.autograd.grad(loss(), c.p['critic'], allow_unused=True)[0]
            assert critic_grad is None
    return dict(component_objective_derivatives=len(errors), max_absolute_error=max(errors))


@test
def byte_output_purity_and_roundtrip():
    prompts = [b'', bytes((0, 10, 32, 61, 255)), bytes(range(256)), b'\xff', b'longer arbitrary bytes']
    for model in MODELS:
        c = Learner(model, 190)
        global_rng = torch.random.get_rng_state().clone()
        before = c.digest()
        batched = c.generate_batch(prompts)
        assert batched == [c.generate_batch([p])[0] for p in prompts]
        assert before == c.digest()
        near(global_rng, torch.random.get_rng_state(), atol=0, rtol=0)
        assert all((len(a) == 16 and s == 'TRUNCATED') for a, s in batched)
        with torch.no_grad():
            c.p['V'].copy_(torch.from_numpy(np.random.default_rng(381).normal(0, .5, c.p['V'].shape)))
        # Nonzero, prompt-dependent heads also exercise padding/decoder parity.
        assert c.generate_batch(prompts) == [c.generate_batch([p])[0] for p in prompts]
        state_before = c.digest()
        c.act(prompts[1], 277, True)
        assert state_before == c.digest()
        near(global_rng, torch.random.get_rng_state(), atol=0, rtol=0)
        with torch.no_grad():
            c.p['V'][-1, EOS] = 1000
        assert c.generate_batch(prompts) == [(b'', 'EOS')] * len(prompts)
        _, _, cache = c.act(b'\x00\xff', 123, False)
        c.update(b'\x00\xff', b'\x80\x00', 0, cache, 'WARM')
        with tempfile.TemporaryDirectory(dir=ROOT / 'ops') as td:
            p = Path(td) / 'state.npz'
            c.save(p)
            d = Learner.load(p)
            assert c.digest() == d.digest() and c.resource == d.resource
            assert c.generate_batch(prompts) == d.generate_batch(prompts)
            with torch.no_grad():
                d.p['E'][0, 0] += 1
            assert c.digest() != d.digest()
    return dict(raw_payload_values=256, roundtrip_models=2, serial_batch_parity=True, independent_clone=True)


@test
def feedback_permissions_and_cache():
    for model in MODELS:
        c = Learner(model, 82)
        for phase in ('WARM', 'HALF', 'SPARSE', 'SCALAR'):
            before = c.digest()
            _, _, cache = c.act(b'p', 19, False)
            u = c.update(b'p', b'answer' if phase != 'SCALAR' else None, 0, cache, phase, 'SAFE', False)
            assert before == c.digest() and u['actor_steps'] == u['critic_steps'] == 0
        _, _, cache = c.act(b'p', 21, True)
        rejects(lambda: c.update(b'p', b'hidden', 0, cache, 'SCALAR', 'SAFE'))
        # The invalid request must not consume the actual-action cache.
        assert not cache['used']
        c.update(b'p', None, 0, cache, 'SCALAR', 'SAFE')
        rejects(lambda: c.update(b'p', None, 0, cache, 'SCALAR', 'SAFE'))
        _, _, stale = c.act(b'p', 22, True)
        _, _, current = c.act(b'p', 23, False)
        u = c.update(b'p', b'x', 1, current, 'SCALAR', 'CE_ONLY')
        assert u['pg'] is None and u['ce'] is not None
        rejects(lambda: c.update(b'p', None, 0, stale, 'SCALAR', 'SAFE'))
    return dict(all_four_no_write_phases=True, scalar_hidden_answer_rejected=True, consumed_and_stale_cache_rejected=True)


@test
def safe_independent_math_and_atomicity():
    reports = []
    for model in MODELS:
        c = Learner(model, 89)
        with torch.no_grad():
            c.p['V'].copy_(torch.from_numpy(np.random.default_rng(91).normal(0, .2, c.p['V'].shape)))
        prompt = b'\xff\x00B'
        _, _, cache = c.act(prompt, 200, True, cap=4)
        oldlog = cache['full_logp'].clone()
        oldbaseline = float(cache['baseline'])
        oldloss = float((c.ce_loss(prompt, b'xy') - (1 - cache['baseline']) * cache['logp']).detach())
        before = c.clone()
        u = c.update(prompt, b'xy', 1, cache, 'HALF', 'SAFE')
        with torch.no_grad():
            newlog = c.trace_logps(prompt, cache['actions'])
            measured = float(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=oldlog), torch.distributions.Categorical(logits=newlog)).sum())
            actual_loss = float(c.ce_loss(prompt, b'xy') - (1 - oldbaseline) * newlog[torch.arange(len(cache['actions'])), torch.tensor(cache['actions'])].sum())
        assert math.isclose(u['old_actor_loss'], oldloss, abs_tol=1e-12)
        assert u['accepted'] is True and measured <= .01 + 1e-12 and actual_loss <= oldloss + 1e-12 * (1 + abs(oldloss))
        assert math.isclose(measured, u['candidates'][-1]['kl'], abs_tol=3e-14)
        assert math.isclose(actual_loss, u['candidates'][-1]['loss'], abs_tol=1e-12)
        assert c.critic_steps == before.critic_steps + 1
        # Forced all-reject is a synthetic atomicity fixture, not a science arm.
        d = before.clone()
        _, _, dc = d.act(prompt, 201, True, cap=4)
        saved_actor = d.digest(actor_only=True)
        saved_critic = d.p['critic'].detach().clone()
        d._candidate_admissible = lambda *args: False
        v = d.update(prompt, None, 0, dc, 'SCALAR', 'SAFE')
        assert v['accepted'] is False and len(v['candidates']) == 8
        assert saved_actor == d.digest(actor_only=True) and d.critic_steps == before.critic_steps + 1
        assert not torch.equal(saved_critic, d.p['critic'])
        assert d.resource['safe_rejections'] == 1 and d.resource['candidate_evaluations'] == 8
        reports.append(dict(model=model, accepted_alpha=u['alpha'], measured_KL=measured, rejected_scales=8))
    return reports


@test
def real_backtracking_and_fixed_proposal():
    result = []
    for model in MODELS:
        c = Learner(model, 89)
        with torch.no_grad():
            c.p['V'].copy_(torch.from_numpy(np.random.default_rng(91).normal(0, .2, c.p['V'].shape)))
        # Synthetic optimizer state amplifies the known RMS proposal; no data,
        # learning-rate tuning or arithmetic outcome is involved.
        c.actor_steps = 100000
        prompt = b'\xff\x00B'
        _, _, cache = c.act(prompt, 200, True, cap=4)
        loss = c.ce_loss(prompt, b'xy') - (1 - cache['baseline']) * cache['logp']
        grads = torch.autograd.grad(loss, list(c.actor().values()), retain_graph=True)
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        proposed = {}
        parameters = {k: p.detach().clone() for k, p in c.actor().items()}
        moments = {}
        for (k, p), g in zip(c.actor().items(), grads):
            g = g * min(1., 1 / max(norm, 1e-30))
            moments[k] = .99 * c.mom[k] + .01 * g.square()
            proposed[k] = .001 * g / (torch.sqrt(moments[k] / (1 - .99 ** 100001)) + 1e-8)
        oldlog, baseline, actions = cache['full_logp'].clone(), float(cache['baseline']), cache['actions']
        u = c.update(prompt, b'xy', 1, cache, 'HALF', 'SAFE')
        assert len(u['candidates']) > 1 and u['accepted']
        for k in parameters:
            near(c.p[k], parameters[k] - u['alpha'] * proposed[k], atol=1e-14, rtol=1e-13)
            near(c.mom[k], moments[k], atol=1e-14, rtol=1e-13)
        assert c.actor_steps == 100001
        # Rebuild every tried displacement and independently measure full-vocab KL.
        for trial in u['candidates']:
            d = c.clone()
            with torch.no_grad():
                for k in parameters:
                    d.p[k].copy_(parameters[k] - trial['alpha'] * proposed[k])
                lp = d.trace_logps(prompt, actions)
                kl = float(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=oldlog), torch.distributions.Categorical(logits=lp)).sum())
                value = float(d.ce_loss(prompt, b'xy') - (1 - baseline) * lp[torch.arange(len(actions)), torch.tensor(actions)].sum())
            assert math.isclose(kl, trial['kl'], abs_tol=3e-13)
            assert math.isclose(value, trial['loss'], abs_tol=2e-12)
        result.append(dict(model=model, tried_scales=len(u['candidates']), accepted_alpha=u['alpha']))
    return result


@test
def generic_acquisition():
    # Fixed non-arithmetic fixtures and dose, specified before observing results.
    prompts = [bytes((0, 255, k, 61)) for k in range(8)]
    answers = [bytes((128 + k, 0)) for k in range(8)]
    result = []
    for model in MODELS:
        c = Learner(model, 671)
        initial = c.generate_batch(prompts)
        start = time.process_time()
        for i in range(4096):
            k = i % 8
            _, status, cache = c.act(prompts[k], 9001 + i, False)
            c.update(prompts[k], answers[k], 0, cache, 'WARM')
        final = c.generate_batch(prompts)
        n = sum(x == (a, 'EOS') for x, a in zip(final, answers))
        result.append(dict(model=model, presentations=4096, initial_exact=sum(x == (a, 'EOS') for x, a in zip(initial, answers)), final_exact=n, cases=8, cpu=time.process_time() - start))
        assert n == 8, result[-1]
        p = ROOT / 'cycles/cycle1/generic_states' / (model + '.npz')
        c.save(p)
        atomic(ROOT / 'cycles/cycle1' / ('GENERIC_' + model + '.json'), result[-1])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', action='append')
    a = p.parse_args()
    stamp = str(time.time_ns())
    report = dict(schema='GATED_FEEDBACK_GENERIC_QUALIFICATION', technical=True, science_jobs=0, tests=[], accepted=False)
    out = ROOT / 'cycles/cycle1' / ('TESTS_' + stamp + '.json')
    for fn in CHECKS:
        if a.only and fn.__name__ not in a.only:
            continue
        start = time.process_time()
        try:
            detail = fn()
            row = dict(name=fn.__name__, accepted=True, cpu=time.process_time() - start, detail=detail)
        except BaseException:
            row = dict(name=fn.__name__, accepted=False, cpu=time.process_time() - start, error=traceback.format_exc())
            report['tests'].append(row)
            atomic(out, report)
            print(json.dumps(row), flush=True)
            raise
        report['tests'].append(row)
        atomic(out, report)
        print(json.dumps(row), flush=True)
    assert report['tests']
    report['accepted'] = True
    atomic(out, report)
    print(json.dumps(dict(accepted=True, tests=len(report['tests']), path=str(out))), flush=True)


if __name__ == '__main__':
    main()
