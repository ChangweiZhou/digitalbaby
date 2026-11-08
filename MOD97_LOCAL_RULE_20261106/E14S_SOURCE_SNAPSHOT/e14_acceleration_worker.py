"""Qualification-only adapter; never changes an already running/locked E14 worker."""
import os,sys,runpy
from pathlib import Path
os.environ['EXP45_WINDOW']='18';os.environ.setdefault('EXP45_HIDDEN','512')
import exp45 as X
from e14_fast_inference import respond
X.respond=respond
script=sys.argv[1];sys.argv=[script,*sys.argv[2:]]
runpy.run_path(str(Path(__file__).resolve().parent/script),run_name='__main__')
