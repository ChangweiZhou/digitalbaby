"""Technical admission only. No DEV or confirmation IDs are admitted."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
DESIGN = ROOT.parent / 'addition_sample_generalization_v2_design_20261011'
CONTRACT = json.loads((DESIGN / 'BENCHMARK_CONTRACT.json').read_text())
ARMS = tuple(row['id'] for row in CONTRACT['roster'])
CHECKPOINTS = tuple(CONTRACT['old_sample_checkpoints'])
HISTORIES = ('W', 'N_OLD', 'SHUFFLED')
OPERANDS = b'01234ABCDEFGHIJ'
ANSWERS = b'012345678'
KAPPA = 15 / 45 ** .5
TECH_PLAN = {
    1: ((835008, 'PATH_LOOP', 'STANDARD', False),),
    2: ((835002, 'STAR_LOOP', 'BLURRY', False),
        (835003, 'PATH_LOOP', 'STANDARD', True)),
    3: ((835004, 'PATH_LOOP', 'STANDARD', False),
        (835005, 'PATH_LOOP', 'BLURRY', False),
        (835006, 'STAR_LOOP', 'STANDARD', False),
        (835007, 'STAR_LOOP', 'BLURRY', False)),
}
TECH_WORLDS = {r[0] for rows in TECH_PLAN.values() for r in rows}


def admit(world, family):
    if type(world) is not int or world not in TECH_WORLDS:
        raise ValueError('technical namespace only; science not authorized')
    if family not in ('PATH_LOOP', 'STAR_LOOP'):
        raise ValueError('ODD_CYCLE learner runs forbidden before confirmation')


def lock():
    result = {}
    for base, names in ((ROOT, ('config.py', 'assets.py', 'cores.py', 'reference.py',
                               'worlds.py', 'runner.py', 'technical.py', 'decision.py',
                               'operations.py', 'cycles.py')),
                        (DESIGN, ('EXPERIMENT.md', 'BENCHMARK_CONTRACT.json',
                                  'benchmark_fixtures.py'))):
        for name in names:
            path = base / name
            result[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
