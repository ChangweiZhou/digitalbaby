"""Real source parity, byte/state and derivative tests. No arithmetic training."""
import ast
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
from core import Learner, Kernel, ARMS, EOS, BEGIN, REPLY
from reference_oracles import old_gru, original_elm, install_elm


def close(a, b, tolerance=2e-10):
    a = np.asarray(a); b = np.asarray(b)
    err = float(np.max(np.abs(a-b))) if a.size else 0.
    assert err <= tolerance, (err, tolerance)
    return err


def checks():
    rows = []
    generic_records = 0
    def ok(name, **extra):
        rows.append(dict(check=name, accepted=True, **extra))
    expected = dict(B16=36401, B24=40009, B69=43664, E6=43913, E6_FIX=43529)
    for arm in ARMS:
        c = Learner(arm, 202610138901)
        count = sum(p.numel() for p in c.net.parameters() if p.requires_grad)
        assert count == expected[arm], (arm, count)
        assert all(p.dtype == torch.float64 for p in c.net.parameters())
        ok(arm+' exact count/dtype', parameters=count)
        before = c.digest()
        c.observe(bytes(range(256)), bytes((0, 10, 32, 61, 255)), False)
        generic_records += 1
        assert c.digest() == before and c.steps == 0
        ok(arm+' full no-write arbitrary payload and long input')
        rng = torch.random.get_rng_state().clone()
        before = c.digest()
        prompts = [b'', b'\x00\n =\xff', b'\x01'*75, b'\x05\x02']
        batched = c.generate_batch(prompts)
        serial = [c.generate(p) for p in prompts]
        assert batched == serial and c.digest() == before
        assert torch.equal(rng, torch.random.get_rng_state())
        ok(arm+' batched/serial own generation parity and purity')
        twin = c.clone()
        with torch.no_grad():
            twin.net.p['V'][0, 0] += 1.
        assert c.digest() == before and twin.digest() != before
        assert all(c.net.p[k].data_ptr() != twin.net.p[k].data_ptr() for k in c.net.p)
        assert all(c.mom[k].data_ptr() != twin.mom[k].data_ptr() for k in c.mom)
        ok(arm+' independent mutable clone')
        for payload in (0, 10, 32, 61, 255):
            with torch.no_grad():
                c.net.p['V'].zero_(); c.net.p['V'][-1, payload] = 10.
            assert c.generate(b'\x00\xff', 2) == (bytes((payload, payload)), 'TRUNCATED')
            assert c.generate(b'', 0) == (b'', 'TRUNCATED')
        with torch.no_grad():
            c.net.p['V'].zero_(); c.net.p['V'][-1, EOS] = 10.
        assert c.generate(b'\x00=\n', 2) == (b'', 'EOS')
        ok(arm+' no vocabulary mask, ownEOS vs cap')

        # Nonzero random head is a derivative fixture, not a trained birth or
        # a claim that all components receive useful native arithmetic credit.
        c = Learner(arm, 771)
        with torch.no_grad():
            c.net.p['V'].copy_(torch.randn(c.net.p['V'].shape, generator=torch.Generator().manual_seed(82), dtype=torch.float64)*.04)
        pred, loss = c.record_loss(b'\x00\x7f\xff\n=', b'\x02\xfe', True)
        named = [(k, p) for k, p in c.net.p.items() if p.requires_grad]
        grads = torch.autograd.grad(loss, [p for _, p in named])
        for (name, p), g in zip(named, grads):
            assert torch.isfinite(g).all() and float(g.norm()) > 0, (arm, name)
            index = int(g.abs().reshape(-1).argmax())
            original = float(p.reshape(-1)[index]); h = 2e-6
            with torch.no_grad(): p.reshape(-1)[index] = original+h
            _, plus = c.record_loss(b'\x00\x7f\xff\n=', b'\x02\xfe', True)
            with torch.no_grad(): p.reshape(-1)[index] = original-h
            _, minus = c.record_loss(b'\x00\x7f\xff\n=', b'\x02\xfe', True)
            with torch.no_grad(): p.reshape(-1)[index] = original
            fd = float((plus-minus).detach())/(2*h)
            analytic = float(g.reshape(-1)[index])
            assert abs(fd-analytic) <= 1e-7+1e-5*abs(analytic), (arm, name, fd, analytic)
            ok(arm+' derivative '+name, analytic=analytic, finite_difference=fd)
        # Distinguish payload and EOS paths, rather than a merged norm alone.
        for target in (19, EOS):
            state = c.net.initial()
            for token in (BEGIN, 0, 255, REPLY):
                state = c.net.step(state, torch.tensor([token]))
            ls = torch.nn.functional.cross_entropy(c.net.logits(state), torch.tensor([target]))
            gradient = torch.autograd.grad(ls, [p for _, p in named])
            assert all(float(g.norm()) > 0 for g in gradient)
            ok(arm+' separate target gradient '+str(target))
        if arm == 'E6_FIX':
            assert not c.net.p['kappa_raw'].requires_grad and 'kappa_raw' not in c.mom
        for i in range(8):
            c.observe(bytes((i, 255-i, 0)), bytes((17+i, i)), True)
            generic_records += 1
        before = c.digest()
        c.observe(b'\x00'*40, b'\xff\n', False)
        generic_records += 1
        assert before == c.digest() and c.steps == 8
        ok(arm+' trained no-write freezes full learning state')

    for arm in ('B16', 'B24', 'B69'):
        c = Learner(arm, 42)
        o = old_gru(42, c.net.width, c.net.embedding)
        for k, p in c.net.p.items():
            close(p.detach().numpy(), o.parameters()[k], 0.)
        ok(arm+' original birth exactly reproduced')
        for i, (prompt, reply, allowed) in enumerate([
            (b'\x00\xff=', b'\x03\x7f', True), (b'\x00\xff=', b'\x03\x7f', True),
            (b'\x02\x03\n', b'\x07', True), (bytes(range(70)), bytes(range(12)), True),
            (b'\x00', b'\xff', False)]):
            o.begin()
            for b in prompt: o.observe(b, False)
            o.start_reply()
            pred = []
            for b in [*reply, EOS]: pred.append(o.observe(b, allowed))
            got = c.observe(prompt, reply, allowed)
            generic_records += 1
            assert [x['choice'] for x in got] == [x['choice'] for x in pred]
            error = close([x['loss'] for x in got], [x['loss'] for x in pred])
            for k, p in c.net.p.items():
                error = max(error, close(p.detach().numpy(), o.parameters()[k]))
                error = max(error, close(c.mom[k].numpy(), o.mom[k]))
            assert c.steps == o.steps
            ok(arm+' original RMS/state/long-credit record '+str(i), maximum_error=error)

    c = Learner('E6', 76)
    original = original_elm()
    install_elm(original, c.net)
    tokens = torch.tensor([[BEGIN, 13, 0, 255, REPLY, 5], [BEGIN, 255, 0, 18, REPLY, 7]])
    sensory = [torch.cat((c.net.p['E'][tokens[:, t]], torch.zeros(2, 760, dtype=torch.float64)), 1) for t in range(tokens.shape[1])]
    original.make_episode = lambda images, delay_override=None: sensory
    reference = original(torch.zeros(2, 28, 28, dtype=torch.float64))
    state = c.net.initial(2); activities, memories = [], []
    for t in range(tokens.shape[1]):
        state = c.net.step(state, tokens[:, t]); memories.append(state[0]); activities.append(state[1])
    error = max(close(torch.stack(activities, 1).detach().numpy(), reference['activity'].detach().numpy()),
                close(torch.stack(memories, 1).detach().numpy(), reference['memory'].detach().numpy()))
    ok('ELM original forward state at every step', maximum_error=error)
    reference_loss = reference['final_activity'].square().sum()+.01*reference['memory'][:, -1].square().sum()
    new_loss = state[1].square().sum()+.01*state[0].square().sum()
    names = [k for k in c.net.p if k not in ('E', 'V')]
    a = torch.autograd.grad(reference_loss, [getattr(original, k) for k in names], retain_graph=True)
    b = torch.autograd.grad(new_loss, [c.net.p[k] for k in names])
    for name, x, y in zip(names, a, b):
        error = close(x.numpy(), y.numpy())
        ok('ELM original gradient '+name, maximum_error=error)
    fix = Learner('E6_FIX', 76)
    assert all(torch.equal(c.net.p[k], fix.net.p[k]) for k in c.net.p)
    assert torch.equal(c.net.source_idx, fix.net.source_idx)
    ok('ELM train/fixed initial state and topology parity')
    for arm in ('E6', 'E6_FIX'):
        net = Learner(arm, 14).net
        for i in range(64):
            assert len(set(net.source_idx[i, :, :3].reshape(-1).tolist())) == 24
            rec = net.source_idx[i, :, 3]-24
            assert len(set(rec.tolist())) == 8 and i not in rec.tolist()
        ok(arm+' fixed nonsemantic graph fan-in')
    tree = ast.parse((Path(__file__).parent/'core.py').read_text())
    imports = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imports |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imports <= {'copy','hashlib','math','pickle','time','collections','numpy','torch'}
    forbidden = {'MultiheadAttention','Transformer','scaled_dot_product_attention','softmax'}
    assert not any(isinstance(n, ast.Attribute) and n.attr in forbidden for n in ast.walk(tree))
    ok('operative core imports and information-flow whitelist')
    return {'accepted': True, 'scope': 'source/state/gradient/byte qualification only', 'checks': rows,
            'check_count': len(rows), 'generic_fixture_records': generic_records,
            'native_arithmetic_records': 0, 'scientific_jobs': 0,
            'derivative_scope': 'one maximum-gradient coordinate per trainable component, plus original-source all-array gradient parity; not full Jacobian or arithmetic capability'}


if __name__ == '__main__':
    result = checks()
    target = Path(__file__).parent/'CONTRACT_TESTS.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))
