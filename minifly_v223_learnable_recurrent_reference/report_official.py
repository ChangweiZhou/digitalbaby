"""Reduce the locked V223 campaign without changing the recorded decisions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'official'


def read(path):
    return json.loads(path.read_text())


def exact(row, panel):
    v = row['metrics'][panel]
    return f'{v["exact"]}/{v["n"]}'


def retained(seed, method, before, after, old_name, new_name):
    folder = OUT/f'{seed}_{method}'
    a = read(folder/before)[old_name]
    b = read(folder/after)[new_name]
    eligible = {r['prompt'] for r in a if r['exact']}
    prompts = {r['prompt'] for r in b}
    correct = {r['prompt'] for r in b if r['exact']}
    eligible &= prompts
    return f'{len(eligible & correct)}/{len(eligible)}'


def main():
    summary = read(OUT/'SUMMARY.json')
    s = summary['screen']
    config = summary['config']
    worlds = config['worlds']
    assert len(summary['A']) == 6
    for seed in worlds:
        pair = [r for r in summary['A'] if r['seed'] == seed]
        assert {r['method'] for r in pair} == {'forward', 'bp'}
        assert len({r['stream_sha256'] for r in pair}) == 1
        for run in pair:
            assert [r['dose'] for r in run['rows']] == config['doses']
            for row in run['rows']:
                assert row['updates'] == row['symbols'] // 32
                raw = read(OUT/f'{seed}_{run["method"]}'/f'outputs_A_{row["dose"]}.json')
                for panel, answers in raw.items():
                    assert row['metrics'][panel]['n'] == len(answers)
                    assert row['metrics'][panel]['exact'] == sum(
                        ans['output'] == ans['expected'] and ans['output'][-1] == 256
                        for ans in answers
                    )
    facts = {method:sum(r['rows'][-1]['metrics']['old_all']['exact'] for r in summary['A']
                        if r['method']==method) for method in ('forward','bp')}
    held = {method:sum(r['rows'][-1]['metrics']['primary']['exact'] for r in summary['A']
                       if r['method']==method) for method in ('forward','bp')}
    lines = ['# V223 — official staged development run','',
             '**2026-11-05 · ' + ('Completed A, B and A′.**' if summary['later'] else 'Completed all locked A doses.**'),
             '', 'The three-world acquisition screen selected the connected width-32 GRU with direct byte input. The main comparison used exact forward sensitivities and BPTT32 on the same graph, objective, exposures, and paired births.',
             '', f'At the final A dose, exact forward credit recalled **{facts["forward"]}/192** facts and answered **{held["forward"]}/1,536** held sums. BPTT32 recalled **{facts["bp"]}/192** facts and answered **{held["bp"]}/1,536** held sums. These totals pool items for readability. The paired worlds are the independent units.',
             '', '## Acquisition screen', '',
             'Cells give exact fact and arithmetic answers with own EOS, out of four each. Every arm received 256 presentations of each record.',
             '', '| Arm | 66101 facts / arithmetic | 66202 facts / arithmetic | 66203 facts / arithmetic |',
             '|---|---:|---:|---:|']
    for arm in ('dense64','connected32','connected64','isolated','kc'):
        r = sorted((r for r in s['rows'] if r['arm']==arm),key=lambda r:r['seed'])
        if r:
            lines.append('| '+arm+' | '+' | '.join(f'{w["curve"][-1]["facts"]}/4 · {w["curve"][-1]["arithmetic"]}/4' for w in r)+' |')
    lines += ['', 'The direct-input matched isolated and connected masks tied on all three final fixture scores. KC input missed the fact floor. The fixture supports acquisition on the direct-byte interface and leaves a connectivity advantage on more demanding streams open.',
              '', '## Main dose curves', '',
              'Each cell gives exact initial fact counts on the three fresh worlds in order. The primary arithmetic panel contains 512 held questions per world.',
              '', '| Addition dose | Forward facts | BPTT32 facts | Forward held arithmetic | BPTT32 held arithmetic |',
              '|---:|---:|---:|---:|---:|']
    paired = {(r['seed'],r['method']):r for r in summary['A']}
    for i,dose in enumerate(config['doses']):
        def cell(method,panel):
            return ' · '.join(exact(paired[w,method]['rows'][i],panel) for w in worlds)
        lines.append(f'| {dose:,} | '+ ' | '.join([cell('forward','old_all'),cell('bp','old_all'),cell('forward','primary'),cell('bp','primary')])+' |')
    decision = read(OUT/'A_DECISION.json')
    lines += ['', f'The locked A gate required at least 52/64 facts on two reference worlds. **{decision["reference_worlds_passing"]}/3 reference worlds cleared it.**',
              '', '## Final A behavior and shortcuts', '',
              '| World | Arm | Taught | Held | Longer operands | Both carry answers | Held first byte | Held bits/answer byte | Held EOS under true prefix |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for w in worlds:
        for method in ('forward','bp'):
            row=paired[w,method]['rows'][-1]
            held=row['metrics']['primary']
            lines.append(f'| {w} | {method} | '+ ' | '.join(exact(row,p) for p in ('taught','primary','extrapolation','both_correct'))+f' | {held["first_correct"]}/512 | {held["answer_bits_per_byte"]:.3f} | {held["eos_top1"]}/512 |')
    lines += ['', '| World | Frequency exact held | Nearest exact held | No-carry exact held |',
              '|---:|---:|---:|---:|']
    for w in worlds:
        m=paired[w,'bp']['arithmetic_shortcuts_at_A_end']['primary']
        lines.append(f'| {w} | {m["frequency"]}/512 | {m["nearest"]}/512 | {m["no_carry"]}/512 |')
    lines += ['', 'The latest-value fact reference retrieves the exact exposed label for every taught fact. The arithmetic shortcuts are fitted on the A exposure stream and use explicit EOS on their complete outputs. The both-correct and length-extrapolation shortcut counts are saved in the per-world A receipts.',
              '', '## Interference and correction', '']
    if summary['later']:
        lines += ['| World | Arm | New facts at B | Unchanged A facts at A′ | B facts at A′ | Corrected | Obsolete | Acquired A facts retained | Acquired B facts retained |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for life in summary['later']:
            w,m=life['seed'],life['method']
            b,f=life['rows'][0],life['rows'][-1]
            old=retained(w,m,'outputs_A_65536.json','outputs_A_prime_32.json','old_all','unchanged')
            new=retained(w,m,'outputs_B_8192.json','outputs_A_prime_32.json','new','new')
            lines.append(f'| {w} | {m} | {exact(b,"new")} | '+ ' | '.join(exact(f,p) for p in ('unchanged','new','corrected','obsolete'))+f' | {old} | {new} |')
    else:
        lines += ['B and A′ were skipped under the locked acquisition rule. The receipts establish A-phase learning behavior. Retention and revision under adequate acquisition remain untested.']
    screen_cpu=sum(r['cpu_seconds'] for r in s['rows'])
    a_cpu=sum(r['cpu_seconds'] for r in summary['A'])
    later_cpu=sum(r['cpu_seconds'] for r in summary['later'])
    rates=config['profile']['rows']
    lines += ['', '## Cost and evidence scope','',
              f'The selected graph was profiled over 8,192 symbols per method before opening the main worlds. Throughput was {rates["forward"]["sites_per_cpu_second"]:,.0f} forward and {rates["bp"]["sites_per_cpu_second"]:,.0f} BPTT symbols per CPU second. The planned training and probe allocation with reserve was {config["profile"]["training_and_probe_cpu_seconds_with_reserve"]/60:.1f} CPU minutes.',
              '', f'Recorded worker CPU was {screen_cpu/60:.1f} minutes for screening, {a_cpu/60:.1f} for A and {later_cpu/60:.1f} for B/A′. These receipts exclude separate arithmetic-shortcut calculation and supervisor overhead. Per-worker peak RSS is recorded in the run receipts.',
              '', 'The comparison uses three paired development worlds. The exact-forward learner begins at random birth and computes its own updates. Its finite-window gradient and the reverse diagnostic target the same computation, with numerical differences possible over long runs. This campaign supplies no independent sealed confirmation.',
              '', 'Run `screen.py`, then `official.py` and `report_official.py` using the single-thread environment from CYCLES.md. The output directory contains locks, raw native outputs, phase checkpoints and the decision receipt.']
    (ROOT/'OFFICIAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='OFFICIAL_MANIFEST.json')
    (ROOT/'OFFICIAL_MANIFEST.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print(summary['status'],len(files),'artifacts')


if __name__=='__main__':
    main()
