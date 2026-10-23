"""Read-only document/provenance checks. Does not import or run any learner."""
import hashlib
import json
import pathlib
import subprocess

D = pathlib.Path(__file__).resolve().parent
R = D.parents[1]
catalog = json.loads((D / 'EXPERIMENT_CATALOG.json').read_text())
assert len(catalog['experiment_deliveries']) == 53
assert len({x['package'] for x in catalog['experiment_deliveries']}) == 53
checked = 0
for row in catalog['experiment_deliveries']:
    for e in row['evidence']:
        b = subprocess.check_output(['git', 'show', 'HEAD:' + e['path']], cwd=R)
        assert hashlib.sha256(b).hexdigest() == e['sha256'], e['path']
        checked += 1
versions = json.loads((D / 'MECHANISMS_254_REVIEW.json').read_text())['rows']
source = json.loads((R / 'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/VERSION_REGISTER_20261011/VERSIONS.json').read_text())['versions']
assert {x['id'] for x in versions} == {x['version_id'] for x in source}
assert len(versions) == len({x['id'] for x in versions}) == 254
for x, original in zip(versions, source):
    assert x['original_register_row'] == original
assert len(json.loads((D / 'SOURCE_HISTORICAL108.json').read_text())['rows']) == 108
assert len(json.loads((D / 'SOURCE_CATALOGUE136.json').read_text())['rows']) == 136
local = json.loads((D / 'WORKFOLDER_REPORT_INDEX.json').read_text())
assert len(local['reports']) == local['distinct_report_contents'] == 158
assert sum(len(x['locations']) for x in local['reports']) == local['report_files'] == 187
branches = json.loads((D / 'BRANCH_ARCHIVE.json').read_text())['branches']
assert len(branches) == 18
for row in branches:
    subprocess.run(['git', 'merge-base', '--is-ancestor', row['sha'], 'HEAD'], cwd=R, check=True)
    for name in ['archive_tag', 'legacy_link_tag']:
        assert subprocess.check_output(['git', 'rev-parse', 'refs/tags/' + row[name]], cwd=R).decode().strip() == row['sha']
old = subprocess.check_output(['git', 'show', '43d04b5e:minifly_v4_design_20261020/DESIGN.md'], cwd=R)
now = subprocess.check_output(['git', 'show', 'HEAD:minifly_v4_design_20261020/DESIGN.md'], cwd=R)
assert old == now
claude = subprocess.check_output(['git', 'show', '19107b68:minifly_v4_design_20261020/DESIGN.md'], cwd=R)
preserved = subprocess.check_output(['git', 'show', 'HEAD:minifly_v4_design_20261020/CLAUDE_BRANCH_ORIGINAL_DESIGN.md'], cwd=R)
assert claude == preserved
print(json.dumps({'status': 'PASS', 'experiment_deliveries': 53, 'report_objects_hashed': checked, 'mechanism_rows': 254, 'historical_items': 108, 'catalogue_rows': 136, 'local_report_files': 187, 'local_report_hashes': 158, 'original_remote_tips_reachable': 18, 'v4_both_designs_preserved_exactly': True, 'training_or_scientific_validation_executed': False}, indent=2))
