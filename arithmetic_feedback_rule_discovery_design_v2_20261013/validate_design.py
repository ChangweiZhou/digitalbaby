"""Static document/count audit only. Imports no learner or evaluator, executes no trajectories."""
import json, math, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
m=json.loads((HERE/'DESIGN.json').read_text())
checks={}
def check(name,condition):
    checks[name]=bool(condition)
    if not condition: raise AssertionError(name)
check('design_only',not m['implemented'] and not m['science_authorized'] and m['scientific_jobs_executed']==0 and m['technical_cycles_executed']==0)
check('eight_fixed_arms',len(m['configs'])==8 and len({c['id'] for c in m['configs']})==8)
check('two_selectable_ce',sorted(c['id'] for c in m['configs'] if c['selectable'])==['G-CE','L-CE'])
check('dev_counts',m['selection']['dev_jobs']==m['selection']['dev_worlds']*m['selection']['dev_configs']==64)
check('confirmation_counts',m['confirm']['jobs']==m['confirm']['worlds']*m['confirm']['configs']==288)
n=m['counts']
check('per_job_record_count',n['records_per_full_job']==3*512+3*6*32==2112)
check('max_counts',n['conditional_max_jobs']==64+288 and n['conditional_max_records']==352*2112==743424)
check('phase_records',n['dev_records']==64*2112 and n['conditional_confirm_records']==288*2112)
check('history_counts',n['conditional_max_roots']==352*3 and n['conditional_max_cohort_forks']==352*18 and n['conditional_max_correlated_trajectories']==352*21)
check('independent_world_count',n['conditional_max_independent_worlds']==8+48)
g=m['readers']['GRU'];l=m['readers']['LAG32_MLP']
check('model_parameters',g['total_model_parameters']==19696+65*257==36401 and l['total_model_parameters']==259*16+64*512+64+65*257==53681)
check('learned_array_bytes',g['learned_model_and_moment_bytes_float64']==36401*16 and l['learned_model_and_moment_bytes_float64']==53681*16)
check('frozen_array_bytes',g['frozen_model_and_head_moment_bytes_float64']==(36401+16705)*8 and l['frozen_model_and_head_moment_bytes_float64']==(53681+16705)*8)
check('primary_family',sum(m['confirm']['primary_groups'].values())==m['confirm']['primary_tests']==16)
check('mean_not_sign_fallback',m['confirm']['zero_variance_inference']['sign_test_not_used_as_mean_test'])
# Algebraic sanity of the declared conservative mean method; not a scientific test.
p=lambda mean,n: math.exp(-n*mean*mean/2) if mean>0 else 1.
check('bounded_mean_sanity',p(0,48)==1 and p(-.1,48)==1 and 0<p(1,48)<p(.1,48)<1)
inv_path=ROOT/'arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.json'
inv=json.loads(inv_path.read_text())
wanted={'historical108':108,'catalogue136':136,'versions254':254,'subset71':71,'first16':16}
actual={k:len(v['rows']) for k,v in inv['views'].items()}
check('full_inventory_coverage',actual==wanted and inv['all_original_rows_preserved'])
links=[]
for path in HERE.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#'):continue
        resolved=(path.parent/target.split('#')[0]).resolve()
        links.append({'document':path.name,'target':target,'exists':resolved.exists()})
check('local_document_links',all(x['exists'] for x in links))
doc=(HERE/'EXPERIMENT.md').read_text()
check('document_inference_repaired','bounded-mean Hoeffding' in doc and 'constant positive gives p=2^-n' not in doc)
check('no_launch_claim','Revised design only' in doc)
check('shortcut_scope','protected shortcut references' in doc)
out={'schema':'RAW_RULE_V2_STATIC_DOCUMENT_AUDIT','passed':True,'scope':'static counts/links/manifest only; no native science/learning or three-cycle qualification',
'checks':checks,'inventory_counts':actual,'links':links,'planned_jobs':352,'planned_records':743424,'trajectories_executed':0}
(HERE/'STATIC_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'passed':True,'static_checks':len(checks),'inventory_counts':actual,'trajectories_executed':0}))

