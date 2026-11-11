# How a cloud worker should clone this repository

Measured by the storage audit of 11 November (`PROJECT_DOCUMENTS/REPO_STORAGE_AUDIT_20261111.md`): a full clone with its checkout is about 21 GiB, more than a cloud container has free. Almost none of that is needed for work in `MOD97_LOCAL_RULE_20261106/`. Use the tested shallow sparse clone, about 275 MiB:

```bash
git clone --depth 1 --filter=blob:none --sparse --single-branch --branch codex/minifly-consolidated-20261023 https://github.com/ChangweiZhou/digitalbaby.git
```

```bash
git -C digitalbaby sparse-checkout set --no-cone '/*' '!/*/' '/MOD97_LOCAL_RULE_20261106/' '!*.pt'
```

To read project documents as well (about 340 MiB), add `'/.claude/' '/PROJECT_DOCUMENTS/'` before `'!*.pt'`.

**Weights are not checked out.** Fetch the ones your job loads, by name, before the first run, for example:

```bash
git -C digitalbaby sparse-checkout add '/MOD97_LOCAL_RULE_20261106/exp45/e17_dev_bank2_nbd_73129.pt'
```

The two clone commands were tested by the audit; the fetch-by-name line follows git's pattern rules and was not, so check that the file arrived. A script that lists `*.pt` will find only what you fetched; a life stops by itself if its birth is missing or is not the expected file. Commit weights only at completion, never for dry runs, and never commit checkpoint files.
