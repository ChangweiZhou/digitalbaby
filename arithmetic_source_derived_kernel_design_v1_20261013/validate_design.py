"""Static source/data/ledger checks only. Does not import or run a learner."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    d = json.loads((HERE / 'DESIGN.json').read_text())
    lineage = json.loads((HERE / 'SOURCE_LINEAGE.json').read_text())
    checks = []
    for item in lineage['files']:
        p = ROOT / item['path']
        assert p.is_file(), item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
    checks.append('source references exist and match recorded identity')
    assert not d['implementation_completed'] and not d['science_authorized']
    assert d['scientific_jobs_executed'] == d['technical_cycles_executed'] == 0
    assert not d['attention_allowed'] and not d['transformers_allowed']
    checks.append('design-only status and architectural exclusions explicit')

    arms = {a['id']: a for a in d['configurations']}
    assert list(arms) == ['B16', 'B24', 'B69', 'E6', 'E6_FIX']
    for name in ('B16', 'B24', 'B69'):
        a = arms[name]
        n, e = a['width'], a['embedding']
        count = 259*e + 3*(n*e+n*n+n) + (n+1)*257
        assert count == a['parameters_expected'] == a['parameters_formula']
    n, r, s, m, h = 64, 8, 4, 6, 12
    cell = n*r*s + 2*n*r + n*(r+m)*h + n*h + n*h*m + n*m + n*m + n*m + n*r + 2*n
    assert cell == 20992
    total = cell + 259*24 + 65*257
    assert total == arms['E6']['parameters_expected'] == 43913
    assert total-n*m == arms['E6_FIX']['parameters_expected'] == 43529
    def gru_count(width):
        return 259*24 + 3*(width*24+width*width+width) + (width+1)*257
    assert min(range(1, 257), key=lambda width: abs(gru_count(width)-total)) == 69
    assert d['selectable'] == ['E6', 'E6_FIX']
    checks.append('parameter formulas, nearest GRU width, roster and selectable recipes')

    b, m = d['data']['binary_final'], d['data']['mixed_final']
    applications = 3*b + 4*m
    updates = 2*b + 3*m
    assert applications == d['histories']['full_job_external_records'] == 32768
    assert updates == d['histories']['full_job_permitted_optimizer_steps'] == 22528
    dev, confirm = d['DEV'], d['CONFIRM']
    assert len(dev['world_ids'])*len(dev['configurations']) == dev['full_jobs'] == 20
    assert len(confirm['world_ids'])*3 == confirm['full_jobs'] == 72
    assert dev['full_jobs'] + confirm['full_jobs'] == d['maximum_scientific_jobs'] == 92
    assert applications*92 == d['maximum_scientific_record_applications'] == 3014656
    assert 4*92 == d['maximum_correlated_final_histories'] == 368
    assert len(confirm['tests']) == confirm['primary_tests'] == 8
    assert d['metrics']['checkpoints_external_W'] == [0,256,512,1024,2048,4096,8192,8256,8320,8448,8704,9216,10240]
    checks.append('conditional science roster, causal-prefix reuse, updates and curve coordinates')

    tech = {c['world'] for c in d['technical_cycles_planned']}
    all_ids = dev['world_ids'] + confirm['world_ids'] + sorted(tech)
    assert len(all_ids) == len(set(all_ids))
    # Existing structured source/report files only; no binary receipt decoding or
    # science revalidation. This verifies planned IDs are not in prior manifests.
    needles = tuple(str(i) for i in all_ids)
    prior_collisions = []
    for folder in ROOT.iterdir():
        if not folder.is_dir() or folder == HERE or folder.name == '.git':
            continue
        if not (folder.name.startswith('arithmetic_') or folder.name == 'PROJECT_DOCUMENTS'):
            continue
        for p in folder.rglob('*'):
            if not p.is_file() or p.suffix not in ('.json', '.md'):
                continue
            if any(part in ('.venv', '__pycache__') for part in p.parts):
                continue
            if p.name == 'CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json':
                continue
            text = p.read_text(errors='replace')
            if any(needle in text for needle in needles):
                prior_collisions.append(str(p.relative_to(ROOT)))
    assert not prior_collisions, prior_collisions
    checks.append('new technical/DEV/confirmation ID namespaces do not overlap prior structured records')

    spec = importlib.util.spec_from_file_location('source_cell_static_worlds', ROOT / 'arithmetic_kernel_no_attention_dev_v3_20261013/worlds.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifests = []
    for world_id in all_ids:
        world = module.make_world(world_id)
        binary = module.schedule(world, 'binary', b)
        mixed = module.schedule(world, 'mixed', m)
        old = module.probes(world['binary_held'], None)
        new = module.probes(world['mixed_held'], None)
        assert len(old) == 128 and len(new) == 256
        assert len(world['mixed_train']) == 512
        train = binary + mixed
        held = old + new
        assert {r['prompt'] for r in train}.isdisjoint(r['prompt'] for r in held)
        assert {r['group'] for r in train}.isdisjoint(r['group'] for r in held)
        assert len(set(world['permutation'])) == 256
        max_events = max(2 + len(bytes.fromhex(r['prompt'])) + len(bytes.fromhex(r['reply'])) + 1 for r in train)
        assert max_events <= 32, (world_id, max_events)
        assert max(len(bytes.fromhex(r['reply'])) for r in train + held) <= 16
        manifests.append({'world': world_id, 'data_digest': world['data_digest'], 'max_training_events': max_events, 'binary_held_groups': len(old), 'mixed_held_groups': len(new)})
    checks.append('all planned worlds: unchanged generator, disjoint groups/bytes, full vocabulary, finite frame/output support')

    result = {'schema': 'SOURCE_CELL_STATIC_VALIDATION_V1', 'accepted': True,
              'scope': 'Source/data/formula validation only; no model implementation, qualification or training',
              'checks': checks, 'worlds_generated_without_learning': len(manifests),
              'new_forward_passes': 0, 'new_training_records': 0, 'scientific_jobs': 0,
              'planned_worlds': manifests}
    (HERE / 'STATIC_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('planned_worlds','checks')}))


if __name__ == '__main__':
    main()
