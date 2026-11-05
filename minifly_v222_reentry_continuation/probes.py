"""Read-only diagnostics on disposable recurrent states."""
from contextlib import contextmanager
import math

import numpy as np

EOS = 256
LN2 = math.log(2)


@contextmanager
def snapshot(model):
    state = model.h.copy(), model.memory.copy(), model.fe.copy(), model.counters()
    try:
        yield
    finally:
        model.h, model.memory, model.fe, counters = state
        model.sites, model.updates, model.writes, model.caps, model.write_norm_sum = counters


def prediction(model):
    result = model.predict(model.sensory())
    return result[0], result[2], result[3]


def reveal(model, symbol, q, read, causal):
    if causal:
        model.write(symbol, q, read)
    model.advance_sensor(symbol)


def rank(logp, symbol):
    return 1 + int(np.count_nonzero(logp > logp[symbol])) + int(np.count_nonzero(logp[:symbol] == logp[symbol]))


def diagnose(model, prompt, answer, causal, cap=12):
    """Prompt ingestion followed by independent autonomous and teacher-forced branches."""
    with snapshot(model):
        prompt_loss, operand_loss, delimiter_loss = [], [], []
        for symbol in prompt:
            logp, q, read = prediction(model)
            loss = -float(logp[symbol]) / LN2
            prompt_loss.append(loss)
            (operand_loss if 48 <= symbol <= 57 else delimiter_loss).append(loss)
            reveal(model, symbol, q, read, causal)
        with snapshot(model):
            emitted = []
            for _ in range(cap):
                logp, q, read = prediction(model)
                symbol = int(logp.argmax())
                emitted.append(symbol)
                reveal(model, symbol, q, read, causal)
                if symbol == EOS:
                    break
        targets = [*answer, EOS]
        loss, ranks = [], []
        for symbol in targets:
            logp, q, read = prediction(model)
            loss.append(-float(logp[symbol]) / LN2)
            ranks.append(rank(logp, symbol))
            reveal(model, symbol, q, read, causal)
    return {
        'prompt': prompt.decode('ascii'), 'expected': targets, 'emitted': emitted,
        'exact': emitted == targets, 'own_eos': emitted[-1] == EOS,
        'first_byte_correct': emitted[0] == answer[0],
        'immediate_eos': emitted[0] == EOS,
        'payload_correct': emitted[:len(answer)] == list(answer),
        'first_probability': 2 ** -loss[0], 'first_rank': ranks[0],
        'answer_bits': loss[:-1], 'answer_ranks': ranks[:-1],
        'eos_bits': loss[-1], 'eos_rank': ranks[-1],
        'prompt_bits': prompt_loss,
        'operand_bits': operand_loss if b'+' in prompt else [],
        'delimiter_bits': delimiter_loss if b'+' in prompt else [],
    }


def state_trace(model, prompt, causal):
    """Capture sampled KCs, writer value, workspace and reads along a visible prefix."""
    trace = []
    with snapshot(model):
        for symbol in prompt:
            x = model.sensory()
            logp, _, q, read, *_ = model.predict(x)
            trace.append((x.copy(), model.h.copy(), read.copy(), model.memory.copy(), logp.copy(), model.values[symbol].copy()))
            reveal(model, symbol, q, read, causal)
        x = model.sensory()
        logp, _, q, read, *_ = model.predict(x)
        trace.append((x.copy(), model.h.copy(), read.copy(), model.memory.copy(), logp.copy(), None))
    return trace


def paired_trace(model, a, b, causal):
    left, right = state_trace(model, a, causal), state_trace(model, b, causal)
    names = ('sampled_kc', 'workspace', 'read', 'memory', 'logp', 'writer_value')
    return [{name: float(np.linalg.norm(p - q)) for name, p, q in zip(names, xs, ys) if p is not None}
            for xs, ys in zip(left, right)]
