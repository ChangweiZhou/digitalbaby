"""Three qualification cycles for E16-WL."""
import copy
import math
import subprocess
import sys
from pathlib import Path

from e16wr2_common import OUT, read, save
from e16wl import StageLog, all_update, byte_masks, configure


def equal(torch, left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(torch, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(torch, a, b) for a, b in zip(left, right))
    return left == right


def main():
    model = configure('C', 'answer')
    x, t = model.X, model.T
    items = [([99, 48, 61], [48, 49, 59, 48, 50, x.EOS]),
             ([114, 61], [49, 50, x.EOS])]
    inputs, targets, answer = x.batch_tensor(items)
    prompt, answer = byte_masks(targets, answer, x.BOS)
    assert prompt.sum().item() == 5 and answer.sum().item() == 9
    assert (targets[answer] == x.EOS).sum().item() == 2
    assert t.equal(prompt | answer, targets != x.BOS) and not (prompt & answer).any()
    print('Cycle 1 PASS: variable lengths, padding exclusion, prompt boundary and EOS', flush=True)

    for local in (True, False):
        for mode in ('answer', 'all'):
            net = x.new_net(73129)
            ref = copy.deepcopy(net)
            opt = t.optim.AdamW(net.parameters(), lr=.001, weight_decay=.25, betas=(.9, .98))
            ref_opt = t.optim.AdamW(ref.parameters(), lr=.001, weight_decay=.25, betas=(.9, .98))
            store, ref_store = x.Store(73129), x.Store(73129)
            rng = t.get_rng_state()
            for _ in range(20):
                if mode == 'answer':
                    a = StageLog(x, mode, [20]).update(net, opt, items, local, store)
                    b = x.update(ref, ref_opt, items, local, ref_store)
                    assert a == b
                else:
                    a, ps, ans, pn, an = all_update(x, net, opt, items, local, store)
                    logits = x.forward(ref, inputs, local)
                    joined = targets != x.BOS
                    expected_prompt = x.cross_entropy(logits[prompt], targets[prompt], reduction='sum')
                    expected_answer = x.cross_entropy(logits[answer], targets[answer], reduction='sum')
                    assert math.isclose(ps, float(expected_prompt.detach()), rel_tol=1e-6)
                    assert math.isclose(ans, float(expected_answer.detach()), rel_tol=1e-6)
                    assert (pn, an) == (5, 9) and math.isclose(a, (ps + ans)/(pn + an), rel_tol=1e-6)
                    loss = x.cross_entropy(logits[joined], targets[joined], reduction='none').mean()
                    ref_opt.zero_grad()
                    loss.backward()
                    ref_opt.step()
                    ref_store.write(inputs, targets, joined)
                assert equal(t, net.state_dict(), ref.state_dict())
                assert equal(t, opt.state_dict(), ref_opt.state_dict())
                assert store.mem == ref_store.mem and store.order == ref_store.order
                assert t.equal(t.get_rng_state(), rng)
    print('Cycle 2 PASS: 20-update exact default parity and all-byte reference updates for both rules', flush=True)

    root = OUT / 'e16wl_technical'
    root.mkdir()
    for arm in ('C', 'D'):
        fixture = root / arm
        subprocess.run([sys.executable, 'e16wl.py', arm, '73129', str(fixture)],
                       env=dict(__import__('os').environ, E8_LOSS='all'), check=True)
        receipt, = [p for p in fixture.glob('*.json') if p.name.endswith('_73129.json')]
        stages = read(receipt)['stages']
        logs = read(receipt.with_suffix('.loss.json'))['stages']
        assert len(stages) == 10 and stages[-1]['task'] == 'compose'
        assert [r['updates'] for r in logs] == [2] * 8 + [4]
        assert all(r['prompt_bytes'] > 0 and r['answer_bytes'] > 0 for r in logs)
        assert all(math.isfinite(r[k]) for r in logs for k in ('prompt_nll', 'answer_nll'))
        if arm == 'C':
            assert [r['replay_count'] for r in read(receipt.with_suffix('.replay.json'))] == [0] + [64] * 7 + [128]
    save(Path('E16WL_CHECKS.json'), dict(status='PASS', cycles=3, scientific_lives=0,
         scope='Mixed-length masks, 20-update state parity for both rules, full nine-stage short fixtures'))
    print('Cycle 3 PASS: complete short lives, observed replay and stage loss receipts', flush=True)


if __name__ == '__main__':
    main()
