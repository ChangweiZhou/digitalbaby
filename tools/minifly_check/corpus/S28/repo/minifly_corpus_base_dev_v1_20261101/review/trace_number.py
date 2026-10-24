"""Reviewer script (trace_number) for the corpus base; committed, never run by the checker."""
import json
import sys


def main():
    sys.stdout.write(json.dumps({"kind": "trace_number"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
