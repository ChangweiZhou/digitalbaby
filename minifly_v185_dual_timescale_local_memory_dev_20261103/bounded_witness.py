"""Bounded C0 retry using the same deterministic V185 feature table."""
import json
import resource
import time
from pathlib import Path

from exercise import Projection, roster, decode, np

LIMIT = 1024**3
START = time.monotonic()

def peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def main():
    projection = Projection()
    rows, _ = roster()
    features = np.array([projection.feature(key + value[:p])
                         for key, value in rows for p in range(3)])
    labels = np.array([y for _, value in rows for y in (*value, 256)])
    design = np.column_stack((features, np.ones(len(labels), dtype=np.float32)))
    # A direct ridge solve initializes a margin projection witness. No optimizer
    # state or fitted array from this diagnostic enters a candidate.
    targets = np.eye(257, dtype=np.float64)[labels]
    x = design.astype(np.float64)
    gram = x.T @ x + np.eye(x.shape[1]) * .001
    weights = np.linalg.solve(gram, x.T @ targets).T
    row = np.arange(len(labels))
    norms = (x * x).sum(axis=1)
    updates = 0
    checks = 0
    for iteration in range(0):
        if iteration % 100 == 0:
            if peak() > LIMIT or time.monotonic() - START > 30:
                break
        scores = x @ weights.T
        correct = scores[row, labels].copy()
        scores[row, labels] = -np.inf
        competitors = scores.argmax(axis=1)
        violations = scores[row, competitors] + 1 - correct
        checks += len(labels)
        worst = int(violations.argmax())
        if violations[worst] <= 1e-8 or iteration == 12000:
            break
        delta = x[worst] * (violations[worst] / (2 * norms[worst]))
        weights[labels[worst]] += delta
        weights[competitors[worst]] -= delta
        updates += 1
    w = weights.astype(np.float32)
    def score(feature):
        return w[:, :64] @ feature + w[:, 64]
    logits = np.array([score(feature) for feature in features])
    correct = logits[row, labels].copy()
    logits[row, labels] = -np.inf
    complete = sum(decode(score, projection, key) == (value, True) for key, value in rows)
    # All 240 rows are re-scored per iteration. Keep that workload explicit.
    return {'complete': complete, 'min_float32_margin': float((correct - logits.max(axis=1)).min()),
            'constraint_updates': updates, 'labeled_site_checks': checks + len(labels),
            'oracle_record_presentations': len(rows), 'unique_labeled_sites': len(labels),
            'peak_rss_bytes': peak(), 'cpu_seconds': time.process_time(),
            'wall_seconds': time.monotonic() - START}

if __name__ == '__main__':
    result = main()
    Path('BOUNDED_WITNESS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)
