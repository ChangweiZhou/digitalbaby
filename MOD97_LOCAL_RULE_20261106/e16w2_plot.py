"""Scientific figures from completed receipts, no training."""
from pathlib import Path
import json,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
def main():
 start=time.process_time();t=time.monotonic();m=json.loads((R/'E16W2_METRICS.json').read_text());assert m['complete']
 roles=[('C_uni','Local uniform','#c3473a'),('C_share','Local 25% composition','#d88b16'),('D_matched','BP exact local batches','#3263a8'),('D_ceiling','BP joint-data ceiling','#348358')]
 fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
 for c,seed in enumerate((73129,93683,116089)):
  for role,label,color in roles:
   ss=m['arms'][f'{role}_{seed}']['stages'][5:]
   for row,key in enumerate(('held_final','mod97_mean')):axes[row,c].plot(range(5),[s[key] for s in ss],marker='o',markersize=4,label=label,color=color,lw=2)
  axes[0,c].set_title(f'Seed {seed}');axes[1,c].set_xticks(range(5),['After\ncompose','After\nadd73','After\nsub73','After\nmax73','After\nmin73'])
 for ax in axes.flat:ax.set_ylim(0,102);ax.grid(alpha=.22)
 axes[0,0].set_ylabel('Held composition: final answer (%)');axes[1,0].set_ylabel('Earlier mod97 operations: mean exact (%)')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
 fig.suptitle('E16W: cross-form retention under four replay/update conditions',fontsize=14)
 fig.savefig(R/'E16W2_RETENTION_CURVES.png',dpi=160);fig.savefig(R/'E16W2_RETENTION_CURVES.svg');plt.close(fig)
 (R/'E16W2_FIGURE_COSTS.json').write_text(json.dumps(dict(instrumented_plotting_cpu_seconds=time.process_time()-start,instrumented_plotting_wall_seconds=time.monotonic()-t,scope='This figure generation only. Earlier reporting/verification process CPU not fully measured; training costs remain lower bounds.'),indent=2)+'\n')
if __name__=='__main__':main()
