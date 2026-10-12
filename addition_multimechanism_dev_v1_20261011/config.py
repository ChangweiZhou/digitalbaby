"""Reserved technical admission; no science execution permitted by this module."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
DESIGN=ROOT.parent/'addition_multimechanism_v1_design_20261011'
PRIOR=ROOT.parent/'addition_observation_support_dev_v1_20261011'
BASELINES=('COV-R','COV-SUPPORT')
CANDIDATES=('COVER','RETRO','ROLE','KCI')
PEERS={'COVER':('COVER-UNIFORM','COVER-RESIDUAL'),
       'RETRO':('RETRO-FIXED','RETRO-RANDOM'),
       'ROLE':('ROLE-FIXED','ROLE-PERMUTED'), 'KCI':('KCI-GLOBAL-NORM',)}
ARMS=BASELINES+CANDIDATES+tuple(a for p in PEERS.values() for a in p)
CHECKPOINTS=(10,20,40,80,160,320)
NEW_CHECKPOINTS=(20,40,80)
HISTORIES=('W','N_OLD','SHUFFLED_OLD')
KCI_LAMBDA=1.0
OPERANDS=b'01234ABCDEFGHIJ';ANSWERS=b'012345678';KAPPA=15/45**.5
TECH_PLAN={1:((1304701,'PATH_LOOP','STANDARD',False),),
 2:((1304702,'STAR_LOOP','BLURRY',False),(1304703,'PATH_LOOP','STANDARD',True)),
 3:((1304704,'PATH_LOOP','STANDARD',False),(1304705,'PATH_LOOP','BLURRY',False),
    (1304706,'STAR_LOOP','STANDARD',False),(1304707,'STAR_LOOP','BLURRY',False),
    (1304708,'TRIANGLE_TAIL','BLURRY',False),(1304709,'TRIANGLE_TAIL','STANDARD',True))}
TECH_TASKS={r[0]:r[1:] for rows in TECH_PLAN.values() for r in rows}
def admit(world,family,noise=None,ordinary=None):
    if type(world) is not int or world not in TECH_TASKS or family!=TECH_TASKS[world][0]:
        raise ValueError('technical namespace only; science not authorized')
    if noise is not None and (noise,ordinary)!=TECH_TASKS[world][1:]:
        raise ValueError('technical stratum mismatch')
def lock():
    paths=list(ROOT.glob('*.py'))+[DESIGN/'EXPERIMENT.md']
    if (DESIGN/'DESIGN_CONTRACT.json').exists():paths.append(DESIGN/'DESIGN_CONTRACT.json')
    paths += [PRIOR/'cores.py',ROOT.parent/'addition_sample_generalization_v2_design_20261011'/'benchmark_fixtures.py']
    return {str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
