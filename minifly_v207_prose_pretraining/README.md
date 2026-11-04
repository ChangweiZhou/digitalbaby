# MiniFly V207 — Prose Pretraining

The official author-disjoint prose gate passed. Across eight held-out authors and three training orders, frozen V9.2 scored 2.786 bits per byte after 1 MiB, 2.708 after 10 MiB, and 4.736 after 10 MiB shuffled. The train-only Markov-3 baseline scored 2.492. The QA transfer roster stayed unopened because the 200-question development check exercised no predictor fallback.

The bundle includes nine scored checkpoints, their SHA-256 checksums, source and corpus manifests, results, the design PDF and development checks. Corpus text can be retrieved using the Gutenberg IDs and verified against the original hashes in OFFICIAL_PLAN.json.

## Reproduction

Use Python 3 with NumPy and SciPy and a C compiler. From the repository root:

```sh
ln -s minifly_mixed_stream_integration minifly-integration
ln -s minifly_v205_round12_qa1_qa2 minifly-v205
ln -s minifly_v202_babi_qa1 minifly-v202
ln -s minifly_qa1_frozen_generalization minifly-qa-generalization
(cd minifly-integration && cc -O3 -ffp-contract=off -dynamiclib predictor.c -o predictor.dylib)
cd minifly_v207_prose_pretraining
mkdir -p books
python -c 'import json; print(*(b["id"] for b in json.load(open("OFFICIAL_PLAN.json"))["books"]),sep="\n")' |
while read -r id; do
  curl -fLsS "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o "books/$id.txt" || exit 1
done
python -c 'import json,hashlib,pathlib; p=json.load(open("OFFICIAL_PLAN.json")); assert all(hashlib.sha256(pathlib.Path("books",str(b["id"])+".txt").read_bytes()).hexdigest()==b["raw_sha256"] for b in p["books"])'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B official.py
```

The compiler command targets macOS. Copy the published plan and results elsewhere if you want to retain them before rerunning. The published engineering runner uses a repository-relative V9.2 source extract and reads its Alice fixture from MINIFLY_ALICE. Its source hash therefore differs from the local engineering receipt. The official runner is unchanged.
