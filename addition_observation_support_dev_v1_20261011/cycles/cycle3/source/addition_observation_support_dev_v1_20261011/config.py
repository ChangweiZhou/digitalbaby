"""Only reserved technical worlds; no science admission."""
from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parent
DESIGN=ROOT.parent/'addition_observation_support_v1_design_20261011'
PRIOR=ROOT.parent/'addition_feedback_efficiency_dev_v1_20261011'
CONTRACT=json.loads((DESIGN/'DESIGN_CONTRACT.json').read_text())
ARMS=tuple(CONTRACT['roster'])
CHECKPOINTS=tuple(CONTRACT['old_checkpoints'])
NEW_CHECKPOINTS=tuple(CONTRACT['new_checkpoints'])
HISTORIES=('W','N_OLD','SHUFFLED_OLD')
OPERANDS=b'01234ABCDEFGHIJ'; ANSWERS=b'012345678'
KAPPA=15/45**.5
TECH_PLAN={1:((1004201,'PATH_LOOP','STANDARD',False),),
2:((1004202,'STAR_LOOP','BLURRY',False),(1004203,'PATH_LOOP','STANDARD',True)),
3:((1004204,'PATH_LOOP','STANDARD',False),(1004205,'PATH_LOOP','BLURRY',False),
   (1004206,'STAR_LOOP','STANDARD',False),(1004207,'STAR_LOOP','BLURRY',False))}
TECH_TASKS={r[0]:r[1:] for rows in TECH_PLAN.values() for r in rows}
def admit(world,family,noise=None,ordinary=None):
    if type(world) is not int or world not in TECH_TASKS or family!=TECH_TASKS[world][0]:
        raise ValueError('reserved technical namespace only; science not authorized')
    if noise is not None and (noise,ordinary)!=TECH_TASKS[world][1:]:
        raise ValueError('technical stratum mismatch')
def lock():
    paths=list(ROOT.glob('*.py'))+[DESIGN/n for n in ('EXPERIMENT.md','DESIGN_CONTRACT.json')]
    paths+=[PRIOR/'cores.py',PRIOR/'decision.py',PRIOR/'worlds.py']
    paths+=[ROOT.parent/'addition_sample_generalization_v2_design_20261011'/'benchmark_fixtures.py']
    return {str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
