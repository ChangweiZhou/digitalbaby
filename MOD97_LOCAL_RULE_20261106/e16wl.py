"""E16-WL: all-byte objective over the existing E8 window learner."""
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

DOSES = [6144] * 8 + [8192]


def configure(arm, mode):
    if mode not in ('answer', 'all'):
        raise ValueError('E8_LOSS must be answer or all')
    os.environ.update(EXP45_HIDDEN='512', E8_BATCH='64', E8_BUFFER_CAP='1024',
                      E8_REPLAY_SAMPLING='uniform', E8_REPLAY_SHARE='1/2',
                      E8_SCHEDULE='blocked', E8_REPLAY_TARGETS='observed' if arm == 'C' else 'self',
                      E8_WD='.25' if arm == 'C' else '.3', E8_LR='.001',
                      E8_SAVE_STAGES='0', E8_COMPOSE_CHECKPOINTS='0', E8_DIAGNOSTICS='0',
                      E8_RUN_PREFIX='e16wl_', E8_LOSS=mode)
    import exp8_e16wr2 as model
    import e14_fast_inference as fast
    model.X.respond = fast.respond
    return model


def byte_masks(y, answer, bos):
    valid = y != bos
    return valid & ~answer, answer


def all_update(native, model, optimizer, items, local, store):
    x, y, answer = native.batch_tensor(items)
    prompt, answer = byte_masks(y, answer, native.BOS)
    mask = prompt | answer
    logits = native.forward(model, x, local)
    losses = native.cross_entropy(logits[mask], y[mask], reduction='none')
    loss = losses.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if store is not None:
        store.write(x, y, mask)
    prompt_in_valid = prompt[mask]
    prompt_sum = float(losses.detach()[prompt_in_valid].sum())
    answer_sum = float(losses.detach()[~prompt_in_valid].sum())
    prompt_count = sum(len(p) for p, _ in items)
    answer_count = sum(len(a) for _, a in items)
    return float(loss.detach()), prompt_sum, answer_sum, prompt_count, answer_count


class StageLog:
    def __init__(self, native, mode, doses):
        self.native = native
        self.original = native.update
        self.mode = mode
        self.doses = doses
        self.reset()
        self.stages = []

    def reset(self):
        self.within = self.prompt_count = self.answer_count = self.store_steps = 0
        self.prompt_sum = self.answer_sum = 0.0
        self.input_hash = hashlib.sha256()

    def update(self, model, optimizer, items, local, store=None):
        if self.mode == 'answer':
            return self.original(model, optimizer, items, local, store)
        loss, prompt_sum, answer_sum, prompt_count, answer_count = all_update(
            self.native, model, optimizer, items, local, store)
        self.within += 1
        self.prompt_sum += prompt_sum
        self.answer_sum += answer_sum
        self.prompt_count += prompt_count
        self.answer_count += answer_count
        self.store_steps += store is not None
        raw = json.dumps([prompt for prompt, _ in items], separators=(',', ':')).encode()
        self.input_hash.update(len(raw).to_bytes(8, 'little') + raw)
        if self.within == self.doses[len(self.stages)]:
            self.stages.append(dict(stage=len(self.stages) + 1, updates=self.within,
                                    prompt_nll=self.prompt_sum / self.prompt_count,
                                    answer_nll=self.answer_sum / self.answer_count,
                                    prompt_bytes=self.prompt_count, answer_bytes=self.answer_count,
                                    store_write_updates=self.store_steps,
                                    prompt_sha256=self.input_hash.hexdigest()))
            self.reset()
        return loss


def run(arm, seed, mode='all', fixture=None):
    model = configure(arm, mode)
    doses = DOSES
    if fixture:
        fixture.mkdir()
        (fixture / f'e8_dev_{seed}.pt').symlink_to(model.OUT / f'e8_dev_{seed}.pt')
        model.OUT = fixture
        model.STAGE_UPDATES, model.COMPOSE_UPDATES = 2, 4
        doses = [2] * 8 + [4]
        split = model.split
        score = model.X.score

        def fixture_split(task, fraction, seed):
            training, panel = split(task, fraction, seed)
            return training[:32], panel

        model.split = fixture_split
        model.X.score = lambda net, items, store=None: score(net, items[:8], store)
    logger = StageLog(model.X, mode, doses)
    model.X.update = logger.update
    start = time.process_time()
    model.life(arm, seed)
    if mode == 'all':
        assert len(logger.stages) == 9 and not logger.within
        assert [s['store_write_updates'] for s in logger.stages] == (doses[:8] + [0] if arm == 'C' else [0] * 9)
        if arm == 'C' and fixture is None:
            reference = json.loads(Path('E16WR2_B_INPUT_REFERENCE.json').read_text())
            matching = next(row for row in reference['seeds'] if row['seed'] == seed)
            assert [s['prompt_sha256'] for s in logger.stages] == [s['prompt_sha256'] for s in matching['stages']]
        path = model.OUT / f'e16wl_e8_life_{arm}{model.TAG}_{seed}.loss.json'
        from e16wr2_common import save
        save(path, dict(arm=arm, seed=seed, loss=mode, stages=logger.stages,
                        data_policy='observed replay' if arm == 'C' else 'joint observed training pool',
                        worker_cpu_seconds=time.process_time() - start,
                        peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        peak_rss_units='bytes' if sys.platform == 'darwin' else 'KiB'))


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), os.environ.get('E8_LOSS', 'answer'),
        Path(sys.argv[3]) if len(sys.argv) > 3 else None)
