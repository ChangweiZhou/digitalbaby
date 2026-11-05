"""Create a human-readable report from completed continuation receipts."""
from collections import Counter
from pathlib import Path
from run import ROOT, ARMS, SEED, make_world, addition, DOSES
from summarize import summarize, load


def first_byte_baseline():
    world = make_world(SEED)
    counts = {width: Counter() for width in (2, 3, 4)}
    held = [addition(*item[2]) for item in world['primary']]
    result = {}
    next_dose = iter(DOSES)
    stop = next(next_dose)
    seen = 0
    for prompt, answer in world['phases'][0]:
        if b'+' not in prompt:
            continue
        width = prompt.index(b'+')
        counts[width][answer[0]] += 1
        seen += 1
        if seen != stop:
            continue
        mode = {width: min(dist, key=lambda k: (-dist[k], k)) for width, dist in counts.items()}
        correct = sum(mode[p.index(b'+')] == a[0] for p, a in held)
        result[str(stop)] = {'correct': correct, 'n': len(held), 'by_width_mode': {k: chr(v) for k,v in mode.items()}}
        stop = next(next_dose, None)
    return result


def report():
    summarize()
    summary = load(ROOT/'SUMMARY.json')
    assert summary['status'] == 'COMPLETE'
    input_audit = load(ROOT/'INPUT_AUDIT.json')
    assert all(arm in input_audit for arm in ARMS)
    curves = summary['studies']['curve']
    repeat = summary['studies']['acquisition']
    baseline = first_byte_baseline()
    lines = [
        '# MiniFly re-entry continuation — causal memory and training dose',
        '',
        '**2026-11-05 · Completed paired development study.**',
        '',
        'The R8 learner and the longer-credit R32-BP diagnostic trained on the same one-world stream, through all five A doses. Each received 65,536 distinct addition records and 4,096 fact presentations, totaling 967,507 symbols. The earlier core, data generator, uniform next-symbol objective, and engineering receipts were preserved. BPTT32 is a diagnostic with a different credit horizon.',
        '',
        'Both probe modes freeze the slow parameters and optimizer and discard all query state afterward. Frozen probes retain the original fixed matrix. Causal probes apply the normal matrix writer to observed prompt bytes and the model’s own emitted symbols. The true-prefix diagnostic has a separate disposable state. The probe checks confirm frozen-output parity with the original code, unchanged training digests, and identical autonomous output when the evaluator’s gold answer is altered.',
        '',
        'At the 65,536-record endpoint, all four arm/probe combinations scored zero exact answers on the taught and held panels. The longer-credit causal probe produced one exact held answer at 32,768 records. Causal memory greatly improved EOS under the correct prefix, while later digits remained weak. The tested longer credit horizon did not restore acquisition or held arithmetic on this world.',
        '',
        '## Learning curve',
        '',
        'Cells give exact autonomous answers including EOS. The two numbers in each cell are frozen / causal. The same 128 taught and 512 never-taught prompts were scored at each dose.',
        '',
        '| Addition records | R8 taught | R8 held | R32-BP taught | R32-BP held |',
        '|---:|---:|---:|---:|---:|',
    ]
    def cell(row, panel):
        a, b = [row['metrics'][mode][panel] for mode in ('frozen','causal')]
        return f'{a["exact"]}/{a["n"]} / {b["exact"]}/{b["n"]}'
    for a,b in zip(curves['R8'], curves['R32-BP'], strict=True):
        lines.append(f'| {a["dose"]:,} | {cell(a,"taught")} | {cell(a,"held")} | {cell(b,"taught")} | {cell(b,"held")} |')
    lines += ['', '## Where held answers break at the final dose', '',
              '| Arm and probe | Exact | First byte correct | Mean first-byte rank | Later answer bits/byte | EOS top-1 at true boundary | Correct payload, wrong termination |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for mode in ('frozen','causal'):
            row=curves[arm][-1]['metrics'][mode]['held']
            lines.append(f'| {arm} {mode} | {row["exact"]}/512 | {row["first_byte_correct"]}/512 | {row["mean_first_rank"]:.2f} | {row["later_answer_bits_per_byte"]:.3f} | {row["eos_top1_at_true_boundary"]}/512 | {row["failure_types"]["wrong_termination"]} |')
    prior = baseline[str(DOSES[-1])]
    lines += ['', f'The evaluator’s width-conditioned first-byte frequency baseline gets **{prior["correct"]}/512** at the final dose. This baseline measures how far answer priors can go without tracking the specific operands.',
              '', 'True-prefix losses by operand bytes, delimiters, answer bytes, and EOS are stored separately in SUMMARY.json. A correct EOS under the true prefix uses the correct prior bytes. It is distinct from stopping at the correct place during free generation.',
              '', '## Repeated-record acquisition', '',
              'The independent acquisition fixture cycles through four addition and four fact records, with a seeded reshuffle each round. It uses a new learner with the same birth and no state shared with the main history.',
              '', '| Presentations per record | R8 exact frozen / causal | R32-BP exact frozen / causal |', '|---:|---:|---:|']
    for a,b in zip(repeat['R8'], repeat['R32-BP'], strict=True):
        lines.append(f'| {a["dose"]} | {cell(a,"taught_repeat")} | {cell(b,"taught_repeat")} |')
    lines += ['', '## Complete input path', '',
              'Across 128 held one-digit carry pairs, the audit compares sampled sensory inputs, the byte-dependent writer target, workspace, matrix, read, and output logits. It records both-correct answer pairs on the final checkpoints.',
              '', '| Model and probe | Distinct sampled KCs at any prompt step | Distinct workspace at answer onset | Distinct memory read at onset | Both answers exact |',
              '|---|---:|---:|---:|---:|']
    for label, item in input_audit.items():
        for mode,row in item['summary'].items():
            both = f'{row["both_exact"]}/128' if 'both_exact' in row else '—'
            lines.append(f'| {label} {mode} | {row["sampled_kc_diff_at_any_step"]}/128 | {row["workspace_different_at_onset"]}/128 | {row["read_different_at_onset"]}/128 | {both} |')
    lines += ['', 'Detectable state differences establish that the edits reach this learner on these probes. The audit leaves open how accessible the required arithmetic transformation is to its output head.',
              '', '## Execution scope and reproduction', '',
              'The paired curves share one development world and birth. The many held questions give item-level behavior within that world. Independent replication across worlds remains open.',
              '',
              'BPTT checkpoints use exact record boundaries for scoring. If a dose stops inside a 32-symbol block, its uncommitted prefix is replayed on a copy for evaluation and later replayed with the complete block for training. Both arms update only on the locked block boundaries. Save files include the restart state and exact-dose reference copy when these differ.',
              '',
              'The original learner directory sits alongside this continuation. Use the environment from its engineering report. Run `run.py check`, then `run.py acquisition --arm R8` and `run.py curve --arm R8`. Repeat with `R32-BP`. Run `response_weight.py` for the independent objective pilot. Run `input_audit.py` and `report.py` to build the tables from the receipts. [Dose curve](DOSE_CURVE.png)',
              '',
              'The experiment covers acquisition and held behavior during A. The B/A′ retention campaign, full-dose response weighting, and remaining route/credit arms are separate follow-ups.']
    intervention = ROOT/'results'/'response_weight_acquisition'/'SUMMARY.json'
    if intervention.exists():
        weighted = load(intervention)
        lines += ['', '## Response-weight intervention',
                  '',
                  'A separate R8 pilot gives four times the loss weight to response bytes and EOS on the same repeated-record history. The observed equals sign opens the weighted span. Each 32-symbol block is normalized to total weight 32 before the unchanged optimizer update. All observations still advance the sensor and fast memory. Multiplier one was bit-identical to the core on a 64-symbol test.',
                  '',
                  '| R8 at 256 presentations per record | Exact | First byte | Later answer bits/byte | EOS top-1 at true boundary |',
                  '|---|---:|---:|---:|---:|']
        for objective, result in [('Uniform', repeat['R8'][-1]), ('4× response', weighted['rows'][-1])]:
            for mode in ('frozen', 'causal'):
                row = result['metrics'][mode]['taught_repeat']
                lines.append(f'| {objective}, {mode} | {row["exact"]}/8 | {row["first_byte_correct"]}/8 | {row["later_answer_bits_per_byte"]:.3f} | {row["eos_top1_at_true_boundary"]}/8 |')
        lines += ['', f'The response and other-site gradients had negative cosine in {weighted["negative_gradient_cosine_blocks"]}/{weighted["mixed_blocks"]} mixed update blocks along the weighted trajectory (mean cosine {weighted["mean_gradient_cosine_mixed_blocks"]:.3f}). This diagnostic concerns that small repeated history. The objective pilot has no held generalization test.']
    cpu=sum(r['worker_cpu_seconds_this_invocation'] for study in summary['worker_totals'].values() for r in study.values())
    lines += ['', f'Total recorded worker CPU was {cpu/60:.1f} minutes across the two curves and two acquisition histories, including their probe evaluations.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from run import write_json
    write_json(ROOT/'FIRST_BYTE_BASELINE.json', baseline)


if __name__ == '__main__':
    report()
