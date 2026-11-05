"""Render the held-out accuracy and predictive loss by dose."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT/'SUMMARY.json').read_text())
assert summary['status'] == 'COMPLETE'
baseline = json.loads((ROOT/'FIRST_BYTE_BASELINE.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout='constrained')
colors = {'R8': '#286985', 'R32-BP': '#ab5b2c'}
for arm, rows in summary['studies']['curve'].items():
    doses = [row['dose'] for row in rows]
    for mode in ('frozen', 'causal'):
        metrics = [row['metrics'][mode]['held'] for row in rows]
        for ax, ys in zip(axes, (
            [100 * m['first_byte_correct'] / m['n'] for m in metrics],
            [m['later_answer_bits_per_byte'] for m in metrics],
        )):
            ax.plot(doses, ys, color=colors[arm], linestyle='-' if mode == 'causal' else '--',
                    marker='o', markersize=4, linewidth=1.7, label=f'{arm} {mode}')
axes[0].axhline(100 * baseline[str(doses[-1])]['correct']/512, color='#777777', linestyle=':', label='First-digit frequency')
axes[0].set(ylabel='First answer byte correct (%)', title='Autonomous first byte')
axes[1].set(ylabel='Bits per later answer byte', title='Continuation under a true prefix')
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_xticks(doses, labels=[f'{d//1024}k' for d in doses])
    ax.set_xlabel('Addition records observed')
    ax.grid(axis='y', alpha=0.22)
axes[0].legend(loc='best', fontsize=8, frameon=False)
fig.suptitle('MiniFly held arithmetic · one paired development world', fontsize=13)
fig.savefig(ROOT/'DOSE_CURVE.png', dpi=180)
