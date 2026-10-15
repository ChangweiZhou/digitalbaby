"""Cycle-2 independent auditor tamper suite; no model execution or training."""
import copy
import gzip
import json
import tempfile
from pathlib import Path
import numpy as np
from auditor import audit
from protocol import ROOT, atomic


def main():
    files = sorted((ROOT / 'cycles/cycle2/receipts').glob('*.json.gz'))
    assert len(files) == 2
    with gzip.open(files[0], 'rt') as f:
        original = json.load(f)
    source = original['source']
    results = []

    def attempt(name, change):
        r = copy.deepcopy(original)
        change(r)
        try:
            audit(r, source)
        except (AssertionError, ValueError, KeyError):
            results.append(dict(name=name, rejected=True))
            return
        raise AssertionError('auditor accepted tamper: ' + name)

    def scalar(r, branch='SAFE'):
        return r['branches'][branch]['segments'][-1]['records'][0]

    def safe(r):
        return next(x['update'] for seg in r['branches']['SAFE']['segments'] for x in seg['records']
                    if x['update']['safe'] and x['update']['accepted'])

    attempt('reward_flip', lambda r: scalar(r).__setitem__('reward', 1 - scalar(r)['reward']))
    attempt('hidden_scalar_answer', lambda r: scalar(r).__setitem__('observed', scalar(r)['expected']))
    attempt('wrong_recipe', lambda r: scalar(r).__setitem__('recipe', 'CE_ONLY'))
    attempt('no_write_critic', lambda r: r['prefixes']['N_WARM']['segment']['records'][0]['update'].__setitem__('critic_steps', 1))
    attempt('no_write_state', lambda r: r['prefixes']['N_WARM']['segment'].__setitem__('after', 'tampered'))
    attempt('missing_CE_ONLY_answer', lambda r: scalar(r, 'CE_ONLY').__setitem__('observed', None))
    attempt('CE_ONLY_policy_gradient', lambda r: scalar(r, 'CE_ONLY')['update'].__setitem__('pg', .1))
    attempt('wrong_SAFE_alpha', lambda r: safe(r)['candidates'][0].__setitem__('alpha', .2))
    attempt('overlimit_SAFE_KL', lambda r: safe(r)['candidates'][-1].__setitem__('kl', .02))
    attempt('negative_SAFE_KL', lambda r: safe(r)['candidates'][-1].__setitem__('kl', -.02))
    attempt('wrong_SAFE_actor_steps', lambda r: safe(r).__setitem__('actor_steps', 1 - safe(r)['actor_steps']))
    attempt('SAFE_loss_increase', lambda r: safe(r)['candidates'][-1].__setitem__('loss', safe(r)['old_actor_loss'] + 1))
    attempt('wrong_critic_permission', lambda r: scalar(r).__setitem__('allowed', False))
    attempt('freeze_wrong_parent', lambda r: r['branches']['FREEZE_SAFE'].__setitem__('fork_state', r['initial']))
    attempt('missing_checkpoint', lambda r: r['branches']['OLD']['segments'][0].__setitem__('curve', []))
    attempt('wrong_held_exact_score', lambda r: r['branches']['SAFE']['curve'][-1]['scores']['binary'][0].__setitem__('exact', not r['branches']['SAFE']['curve'][-1]['scores']['binary'][0]['exact']))
    attempt('wrong_source', lambda r: r['source'].__setitem__('core.py', 'tampered'))
    attempt('wrong_count', lambda r: r.__setitem__('executed_records', r['executed_records'] + 1))
    attempt('wrong_architecture_size', lambda r: r['parameters'].__setitem__('E', r['parameters']['E'] + 1))
    # Actual array alteration with unchanged digest, isolated from accepted data.
    with tempfile.TemporaryDirectory(dir=ROOT / 'ops') as td:
        path = Path(td) / 'bad.npz'
        original_path = ROOT / original['archives']['BIRTH.npz']
        with np.load(original_path, allow_pickle=False) as source_arrays:
            arrays = {k: source_arrays[k].copy() for k in source_arrays.files}
        arrays['p_E'][0, 0] += .1
        np.savez_compressed(path, **arrays)
        attempt('altered_parameter_array', lambda r: r['archives'].__setitem__('BIRTH.npz', str(path.relative_to(ROOT))))
        arrays['p_E'][0, 0] -= .1
        arrays['m_E'][0, 0] = -1.
        np.savez_compressed(path, **arrays)
        attempt('negative_RMS_array', lambda r: r['archives'].__setitem__('BIRTH.npz', str(path.relative_to(ROOT))))
    atomic(ROOT / 'cycles/cycle2/TAMPER_TESTS.json', dict(accepted=True, cases=results,
        rejected=len(results), native_records_executed=0, receipt_mutations_are_synthetic=True))
    print(json.dumps(dict(accepted=True, rejected=len(results))))


if __name__ == '__main__':
    main()
