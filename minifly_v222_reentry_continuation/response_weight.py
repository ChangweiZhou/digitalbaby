"""Development intervention on the objective of the unchanged R8 learner."""
import argparse
import copy
import json
import time

import numpy as np

from run import CORE, ROOT, SEED, Learner, evaluate, layout, lock, save_model, write_json
from learner import EOS

MULTIPLIER = 4.0


class WeightedUpdates:
    """Apply a prefix-defined loss weight, with each block normalized to total weight 32."""
    def __init__(self, model, multiplier):
        self.model = model
        self.multiplier = multiplier
        self.in_response = False
        self.response_sites = 0
        self.total_response_sites = 0
        self.response_theta = np.zeros_like(model.theta)
        self.response_head = np.zeros_like(model.head)
        self.alignment = []

    def step(self, y):
        model = self.model
        if self.in_response:
            previous_theta, previous_head = model.gtheta.copy(), model.ghead.copy()
        loss = model.step(y, update=False)
        if self.in_response:
            self.response_theta += model.gtheta - previous_theta
            self.response_head += model.ghead - previous_head
            self.response_sites += 1
            self.total_response_sites += 1
        if y == ord('='):
            self.in_response = True
        elif y == EOS:
            self.in_response = False
        if model.sites % 32 == 0:
            if 0 < self.response_sites < 32:
                response = np.r_[self.response_theta.ravel(), self.response_head.ravel()]
                other = np.r_[(model.gtheta - self.response_theta).ravel(),
                              (model.ghead - self.response_head).ravel()]
                denom = float(np.linalg.norm(response) * np.linalg.norm(other))
                if denom > 0:
                    self.alignment.append(float(response @ other / denom))
            if self.multiplier != 1:
                extra = self.multiplier - 1
                scale = 32 / (32 + extra * self.response_sites)
                model.gtheta += extra * self.response_theta
                model.ghead += extra * self.response_head
                model.gtheta *= scale
                model.ghead *= scale
            model.apply_update()
            self.response_theta.fill(0)
            self.response_head.fill(0)
            self.response_sites = 0
        return loss


def check(stream):
    model = Learner(SEED)
    baseline = copy.deepcopy(model)
    weighted = WeightedUpdates(model, 1)
    for y in stream[:64]:
        assert weighted.step(y) == baseline.step(y)
    assert model.digest() == baseline.digest()
    assert weighted.total_response_sites > 0


def main():
    from worlds import tokens
    p = argparse.ArgumentParser()
    p.add_argument('--panel', choices=('acquisition',), default='acquisition')
    p.parse_args()
    plan = lock()
    records, doses, stops, panels, _ = layout('acquisition')
    stream = tokens(records)
    check(stream)
    out = ROOT/'results'/'response_weight_acquisition'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'PLAN.json', {
        'base_stream_sha256': plan['schedule']['acquisition']['stream_sha256'],
        'multiplier': MULTIPLIER, 'scope': 'One separate R8 repeated-record development pilot',
        'response_condition': 'Observed equals sign through genuine EOS, inclusive of EOS loss',
        'block_weight': 'Rescale weighted gradient to total weight 32 before the same optimizer update',
        'compared_with': 'Uniform-objective R8 at the same birth and exposures',
        'memory_and_sensor': 'All observations advance ordinary memory and sensor dynamics',
        'multiplier_one_check': 'Bit-identical to original R8 on the 64-symbol fixture',
    })
    model = Learner(SEED)
    weighted = WeightedUpdates(model, MULTIPLIER)
    results = []
    start = time.process_time()
    for dose, stop in zip(doses, stops):
        loss = sum(weighted.step(y) for y in stream[model.sites:stop])
        raw, metrics = evaluate(model, panels)
        row = {'dose': dose, 'symbols': stop, 'optimizer_updates': model.updates,
               'training_loss_nats_this_leg_unweighted': loss,
               'weighted_sites': weighted.total_response_sites, 'metrics': metrics,
               'state_sha256': model.digest()}
        write_json(out/f'{dose}_outputs.json', raw)
        results.append(row)
        write_json(out/'PROGRESS.json', results)
        print(dose, {k: (v['taught_repeat']['exact'], v['taught_repeat']['first_byte_correct'])
                    for k,v in metrics.items()}, flush=True)
    save_model(model, out/'256.npz')
    write_json(out/'SUMMARY.json', {'rows': results, 'cpu_seconds': time.process_time()-start,
                                  'mean_gradient_cosine_mixed_blocks': float(np.mean(weighted.alignment)),
                                  'negative_gradient_cosine_blocks': sum(v < 0 for v in weighted.alignment),
                                  'mixed_blocks': len(weighted.alignment)})


if __name__ == '__main__':
    main()
