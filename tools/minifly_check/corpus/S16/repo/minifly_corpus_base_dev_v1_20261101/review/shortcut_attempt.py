"""Reviewer script (shortcut_attempt) for the corpus base; committed, never run by the checker."""
import json
import sys


def main():
    sys.stdout.write(json.dumps({"kind": "shortcut_attempt"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
