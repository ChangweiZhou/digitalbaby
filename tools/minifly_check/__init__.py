"""MiniFly research-loop kit checker (MRK-1, schema v1).

Run it as ``python3 -I tools/minifly_check/minifly_check.py <command> ...`` (abbreviated MFC).
Kit modules import each other top-level (``import common``); importing this package only puts
the kit directory on sys.path so that convention also works for package-style imports.
Nothing is imported eagerly, so startup stays fast and a broken module fails only the
commands that need it (CHECKER_ERROR, exit 3).
"""
import os as _os
import sys as _sys

_KIT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _KIT_DIR not in _sys.path:
    _sys.path.insert(0, _KIT_DIR)

CHECKER_VERSION = '1.0.0'
