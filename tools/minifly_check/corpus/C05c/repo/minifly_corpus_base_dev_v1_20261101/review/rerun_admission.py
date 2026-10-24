"""Reviewer script (rerun_admission) for the corpus base; committed, never run by the checker."""
import json
import sys


def main():
    sys.stdout.write(json.dumps({"kind": "rerun_admission"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
