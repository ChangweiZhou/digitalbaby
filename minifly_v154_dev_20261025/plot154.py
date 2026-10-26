"""Scientific plots of saved native curves only; no model queries."""
from pathlib import Path
import json, os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/minifly_v154_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())

def main():
    assert load(H/'FINAL_AUDIT.json')['status']=='ACCEPTED'
    groups=[load(p) for p in sorted((H/'groups').glob('*/ACQUISITION.json'))]
    worlds=sorted({g['world'] for g in groups})
    configs=sorted({g['config']['id'] for g in groups})
    colors=dict(zip(configs,plt.get_cmap('tab10').colors))
    for axis,filename,xlabel in [
        ('observed_targets','ACQUISITION_TARGETS','Observed reply/EOS targets in W history (including repeats)'),
        ('training_CPU','ACQUISITION_CPU','Cumulative group training CPU seconds (includes preceding controls)')]:
        fig,axs=plt.subplots(2,2,figsize=(12,8),sharey=True,layout='constrained')
        for row,world in enumerate(worlds):
            for g in [z for z in groups if z['world']==world]:
                old=[];new=[]
                for c in g['curves']:
                    if c['branch']!='W' or c['tag']=='POST_IDLE':continue
                    if c['tag'].startswith('OLD') or c['tag']=='INITIAL_OLD':old.append((c[axis],c['metrics']['correct']/64))
                    if c['tag']=='BEFORE_NEW':new.append((c[axis],c['metrics']['correct']/64))
                    if c['tag'].startswith('NEW'):
                        old.append((c[axis],sum(z['correct'] for z in c['rows'][:64])/64))
                        new.append((c[axis],sum(z['correct'] for z in c['rows'][64:])/64))
                for col,points in enumerate([old,new]):
                    axs[row,col].plot(*zip(*points),marker='o',markersize=3,
                                      label=g['config']['id'].replace('N_',''),color=colors[g['config']['id']])
            for col,title in enumerate(['OLD acquisition and retention','NEW acquisition']):
                a=axs[row,col];a.set_title(f"World {str(world)[-4:]} — {title}");a.set_xlabel(xlabel)
                a.set_ylim(-.015,1.015);a.grid(alpha=.25);a.set_ylabel('Complete own-byte/EOS accuracy')
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=5)
        fig.savefig(H/(filename+'.png'),dpi=160,bbox_inches='tight')
        fig.savefig(H/(filename+'.svg'),bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
