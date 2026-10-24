"""Fail-closed command-line validity checker. Not a behavioral success gate."""
import argparse
import json
from v144_audit import validate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',required=True);ap.add_argument('--input',required=True)
    a=ap.parse_args()
    with open(a.lock) as f:lock=json.load(f)
    with open(a.input) as f:value=json.load(f)
    result=validate(lock,value['rows'],value['required'],value.get('closure_ok') is True,
                    value.get('expected_settings'))
    print(json.dumps(result))
    return 0 if result['valid'] else 2


if __name__=='__main__':raise SystemExit(main())
