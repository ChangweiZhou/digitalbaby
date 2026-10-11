"""Pure task enumeration. No learner imports, updates or efficacy estimates."""
from fractions import Fraction
from itertools import permutations, combinations_with_replacement
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
EDGES = ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4))


def exact_rank(rows):
    a = [[Fraction(x) for x in row] for row in rows]
    rank = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [x / scale for x in a[rank]]
        for i in range(len(a)):
            if i != rank and a[i][col]:
                scale = a[i][col]
                a[i] = [x - scale*y for x, y in zip(a[i], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def check():
    masks = {tuple(sorted(tuple(sorted((p[i], p[j]))) for i, j in EDGES))
             for p in permutations(range(5))}
    assert len(masks) == 60
    ranks = set()
    all_pairs = set(combinations_with_replacement(range(5), 2))
    witness_rows = 0
    for mask in masks:
        train = set(mask)
        held = all_pairs - train
        assert len(train) == 5 and len(held) == 10
        assert all(i != j for i, j in train)
        orientations = [(a, b) for i, j in mask for a, b in ((i, j), (j, i))]
        assert len(orientations) == 10
        assert {i for i, j in orientations} == set(range(5))
        assert {j for i, j in orientations} == set(range(5))
        assert all(tuple(sorted((i, j))) not in train
                   for a, b in held for i, j in ((a, b), (b, a)))
        rows = []
        for i, j in orientations:
            row = [0] * 11
            row[i] = row[5+j] = row[10] = 1
            rows.append(row)
        ranks.add(exact_rank(rows))
        for pi in permutations(range(5)):
            # Environment-side structural witness; never learner initialization.
            w = list(pi) + list(pi) + [0]
            for i, j in all_pairs:
                assert w[i] + w[5+j] + w[10] == pi[i] + pi[j]
                assert 0 <= pi[i] + pi[j] <= 8
                witness_rows += 1
    assert ranks == {9}
    jobs_dev = 12 * 6
    jobs_confirm_max = 128 * 3
    records_per_job = 3*320 + 3*6*80 + 2*80
    assert records_per_job == 2560
    result = {
        'status': 'PASS_GEOMETRY_ONLY',
        'triangle_tail_distinct_labelled_masks': len(masks),
        'exact_independent_role_design_ranks': sorted(ranks),
        'clean_equation_witness_rows': witness_rows,
        'taught_semantic_pairs': 5,
        'held_semantic_pairs': 10,
        'dev_jobs': jobs_dev,
        'confirmation_jobs_max': jobs_confirm_max,
        'total_jobs_max': jobs_dev + jobs_confirm_max,
        'external_records_per_job': records_per_job,
        'external_records_max': (jobs_dev + jobs_confirm_max)*records_per_job,
        'learner_updates': 0,
        'scientific_worlds_run': 0,
        'implementation_qualified': False,
        'three_cycles_completed': False,
    }
    (ROOT/'GEOMETRY_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    check()
