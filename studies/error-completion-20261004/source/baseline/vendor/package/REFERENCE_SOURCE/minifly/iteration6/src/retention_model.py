"""Iteration-6 allocation perturbation and diagnostic write controls.

GPL-3.0-or-later, derived from the unchanged iteration-5 Huang-based learner.
Uses existing arrays only; the diagnostic control is an external intervention.
"""
from pathlib import Path
import sys, json, copy
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
sys.path.insert(0, str(BASE / 'iteration5/src'))
from hybrid import HybridLearner, advance_step
import numpy as np

CONFIG = json.loads((ROOT / 'config.json').read_text())


class RetentionLearner(HybridLearner):
    def __init__(self, allocation='parent', mode='full'):
        super().__init__('bi', mode)
        if allocation not in ('parent', 'equal_allocation'):
            raise ValueError('Only parent and the one declared allocation are available')
        self.allocation = allocation
        self.kernel = dict(self.kernel)
        if allocation == 'equal_allocation':
            self.kernel['fraction'] = CONFIG['candidate_fraction']

    def step(self, seconds, pn_activity=None, punishment=0., control='FULL'):
        if control == 'FULL':
            return super().step(seconds, pn_activity, punishment)
        if control not in ('REST', 'SENSORY', 'TEACHER_NO_WRITE', 'GAMMA_ONLY'):
            raise ValueError('Unknown diagnostic control')
        if not np.isfinite(seconds) or seconds < 0 or not np.isfinite(punishment):
            raise ValueError('Finite duration and punishment required')
        p = np.zeros(self.input_channels) if pn_activity is None or control == 'REST' else np.asarray(pn_activity, float)
        if control == 'REST':
            punishment = 0.
        x = self.encode(p)
        alpha_fast, alpha_slow = self.fast[:, 1].copy(), self.slow.copy()
        k = self.kernel
        rates, payload = advance_step(x, float(seconds), float(punishment), self.Q, self.T,
            self.F, self.MM, self.k0, self.fw0, self.fwd, self.ta, self.tg,
            k['fast_tau'], k['slow_tau'], k['fraction'], self.fast, self.slow,
            self.adapt, control == 'GAMMA_ONLY', control in ('GAMMA_ONLY', 'TEACHER_NO_WRITE'))
        if control == 'GAMMA_ONLY':
            self.fast[:, 1] = alpha_fast * np.exp(-seconds / k['fast_tau'])
            self.slow[:] = alpha_slow * np.exp(-seconds / k['slow_tau'])
        self.elapsed = np.float64(self.elapsed + seconds)
        self.event_count += np.uint64(1)
        self.presentation_count += np.uint64(np.any(p > 0))
        return dict(rates=rates, teaching_payload=payload, active_KCs=int(x.sum()))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fast=self.fast, slow=self.slow, adapt=self.adapt,
            elapsed=self.elapsed, event_count=self.event_count, presentation_count=self.presentation_count,
            interface_sha=self.interface_sha, allocation=self.allocation, mode=self.mode,
            kernel_json=json.dumps(self.kernel, sort_keys=True))

    @classmethod
    def restore(cls, path):
        with np.load(path, allow_pickle=False) as state:
            m = cls(str(state['allocation']), str(state['mode']))
            if str(state['interface_sha']) != m.interface_sha or json.loads(str(state['kernel_json'])) != m.kernel:
                raise ValueError('Checkpoint interface/kernel differs from declared model')
            for key in ('fast', 'slow', 'adapt'):
                if state[key].shape != getattr(m, key).shape:
                    raise ValueError('Checkpoint shape differs')
                setattr(m, key, state[key].copy())
            for key, dtype in [('elapsed', np.float64), ('event_count', np.uint64), ('presentation_count', np.uint64)]:
                setattr(m, key, dtype(state[key]))
        return m

    @classmethod
    def from_parent_checkpoint(cls, path):
        parent = HybridLearner.restore(path)
        m = cls('parent')
        for key in ('fast', 'slow', 'adapt'):
            setattr(m, key, getattr(parent, key).copy())
        for key in ('elapsed', 'event_count', 'presentation_count'):
            setattr(m, key, getattr(parent, key))
        return m
