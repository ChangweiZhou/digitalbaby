"""Verify V144-R2's pinned local source/asset closure; restore missing Git files on request."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def verify(lock_path, restore=False):
    lock = json.loads(Path(lock_path).read_text())
    problems = []
    for rel, item in sorted(lock["files"].items()):
        path = ROOT / rel
        if not path.is_file() and restore and item.get("git_ref"):
            try:
                data = subprocess.check_output(
                    ["git", "show", f"{item['git_ref']}:{rel}"], cwd=ROOT)
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise RuntimeError("Git object hash differs from lock")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                problems.append(f"{rel}: restoration failed: {exc}")
        if not path.is_file():
            problems.append(f"{rel}: missing; expected SHA256 {item['sha256']} ({item['origin']})")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            problems.append(f"{rel}: hash mismatch; expected {item['sha256']}")
    return {"status": "QUALIFIED" if not problems else "INVALID", "problems": problems,
            "checked_files": len(lock["files"]), "source_commit": lock["source_commit"],
            "parent_commit": lock["parent_commit"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default=str(Path(__file__).with_name("SOURCE_LOCK.json")))
    ap.add_argument("--restore-missing-git-files", action="store_true")
    args = ap.parse_args()
    result = verify(args.lock, args.restore_missing_git_files)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
