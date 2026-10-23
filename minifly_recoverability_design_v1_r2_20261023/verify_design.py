"""Static document checks only: no native imports, fitting or data generation."""
from pathlib import Path
import hashlib
import json
import re

P = Path(__file__).resolve().parent
R = P.parent
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()


def main():
    spec = json.loads((P / 'SPEC.json').read_text())
    lock = json.loads((P / 'SOURCE_LOCK.json').read_text())
    links = json.loads((P / 'MECHANISM_LINKS.json').read_text())
    for name, value in lock['source_files'].items():
        assert sha(R / name) == value, name
    for name, value in lock['outside_repo_sources'].items():
        assert sha(Path(name)) == value, name
    for name, value in lock.get('documents', {}).items():
        assert sha(P / name) == value, name
    for name in ('DESIGN.md', 'MECHANISM_DELTA.md'):
        for target in re.findall(r'\]\(([^)]+)\)', (P / name).read_text()):
            if '://' not in target:
                assert (P / target).exists(), (name, target)
    A, B, roster = spec['A'], spec['B'], spec['roster']
    assert A['parameters'] == 171 * 32 + 32 + 32 + 1 == 5537
    lives = len(roster['pairs']) * roster['value_lives_per_pair']
    assert lives == 8
    assert A['fresh_roster_native_records'] == lives * 2 * (3 * 504 + 252) == 28224
    assert B['additional_native_records'] == lives * 3 * (504 + 252) == 18144
    assert roster['base_histories'] == lives * (2 * 3 + 3) == 72
    assert roster['passive_continuations'] == lives * (2 + 3) == 40
    assert roster['main_native_records'] == A['fresh_roster_native_records'] + B['additional_native_records'] == 46368
    assert roster['main_native_events'] == 5 * roster['main_native_records'] == 231840
    assert roster['shared_OLD_logical_equivalents'] == 40 * 252 == 10080
    assert roster['logical_record_equivalents'] == 46368 + 10080 == 56448
    assert B['mutable_credit_and_timestamp_bytes'] == 9 * 5177 * 16 == 745488
    assert A['fits'] == len(A['storage']) * len(A['head_seeds']) == 6
    assert not B['wait_for_A_success']
    assert not A['final_OLD_rescue_required_for_component']
    assert spec['constraints']['gradual_native_schedule_unchanged']
    assert spec['constraints']['arithmetic_track_unchanged']
    assert all(v == 0 for v in spec['executed_this_delivery'].values())
    assert spec['authorization']['design']
    assert not any(v for k, v in spec['authorization'].items() if k != 'design')
    assert len(links['rows']) == len(set(r['version_id'] for r in links['rows'])) == 254
    assert sha(R / links['inventory']) == links['inventory_sha256']
    assert all(not r['new_run_authorized'] and not r['original_scientific_status_changed'] for r in links['rows'])
    assert sha(R / 'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json') == spec['arithmetic_pointer_sha256']
    pointer = json.loads((R / 'PROJECT_DOCUMENTS/CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json').read_text())
    assert pointer['version'] == spec['version']
    assert not pointer['implemented'] and not pointer['execution_authorized']
    review = dict(status='STATIC_DOCUMENT_CHECKS_PASS', date='2026-10-23',
        evidence='Design consistency only; not E0 native qualification or scientific efficacy',
        independent_reviewer=False, native_or_head_code_implemented=False,
        new_native_records=0, head_fits=0, arithmetic_pointer_unchanged=True,
        source_hashes_verified=True, local_markdown_links_resolve=True,
        complete_inventory_ID_coverage=254, counts_and_head_parameters_verified=True,
        A_not_gated_on_impossible_OLD_recovery=True, B_has_fixed_readout=True,
        remaining_preexecution_requirements=[
            'Implement and qualify the one gate without editing frozen parent sources',
            'Lock full runtime dependency closure, birth assets and actual stream/key collision audits',
            'Verify g=1/g=0, credit bounds/ordering, no-write/label isolation, restore and EOS payload invariants',
            'Meter all existing-artifact fitting, native/control work and resource admission'],
        unresolved_design_risks=[
            'Credit is a direct-alpha proxy, not complete native causal attribution',
            'Fixed credit/gate constants are engineering hypotheses, not inferred optimal values',
            'Attenuation-only event L1 yoking may become infeasible in diverged closed-loop states',
            'Shuffling preserves the gate multiset but not necessarily weighted write budgets',
            'Protection can impair NEW acquisition or later revision; no general-core claim follows',
            'Four pairs support descriptive component decisions only'])
    (P / 'STATIC_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(dict(status=review['status'], main_records=roster['main_native_records'],
                         head_fits_planned=A['fits'], executed_native_records=0,
                         arithmetic_pointer_unchanged=True)))


if __name__ == '__main__':
    main()
