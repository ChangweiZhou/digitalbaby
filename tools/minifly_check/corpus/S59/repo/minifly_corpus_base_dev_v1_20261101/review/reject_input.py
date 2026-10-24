"""Reviewer script (reject_input) for the corpus base; committed, never run by the checker."""
import json
import sys


def main():
    sys.stdout.write(json.dumps({"kind": "reject_input"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
