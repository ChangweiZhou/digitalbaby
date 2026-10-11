"""Reserved technical admission. No DEV/confirmation launch permission."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
DESIGN = ROOT.parent/'addition_feedback_efficiency_v1_design_20261011'
PRIOR = ROOT.parent/'addition_sample_generalization_dev_v2_20261011'
CONTRACT = json.loads((DESIGN/'DESIGN_CONTRACT.json').read_text())
ARMS = tuple(CONTRACT['roster'])
CHECKPOINTS = tuple(CONTRACT['old_checkpoints'])
NEW_CHECKPOINTS = tuple(CONTRACT['new_checkpoints'])
HISTORIES = ('W','N_OLD','SHUFFLED_OLD')
OPERANDS = b'01234ABCDEFGHIJ'
ANSWERS = b'012345678'
KAPPA = 15/45**.5
TECH_PLAN = {
  1: ((941001,'PATH_LOOP','STANDARD',False),),
  2: ((941002,'STAR_LOOP','BLURRY',False),(941003,'PATH_LOOP','STANDARD',True)),
  3: ((941004,'PATH_LOOP','STANDARD',False),(941005,'PATH_LOOP','BLURRY',False),
      (941006,'STAR_LOOP','STANDARD',False),(941007,'STAR_LOOP','BLURRY',False)),
}
TECH_TASKS = {row[0]:row[1:] for rows in TECH_PLAN.values() for row in rows}

def admit(world,family,noise=None,ordinary=None):
    if type(world) is not int or world not in TECH_TASKS or family != TECH_TASKS[world][0]:
        raise ValueError('technical namespace/task only; science not authorized')
    if noise is not None and (noise,ordinary) != TECH_TASKS[world][1:]:
        raise ValueError('technical task stratum mismatch')

def lock():
    out={}
    for p in sorted(ROOT.glob('*.py')):
        out[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('EXPERIMENT.md','DESIGN_CONTRACT.json','geometry_check.py','MECHANISM_REVIEW.md'):
        p=DESIGN/name;out[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('cores.py','config.py'):
        p=PRIOR/name;out[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT.parent/'addition_sample_generalization_v2_design_20261011'/'benchmark_fixtures.py'
    out[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return out
