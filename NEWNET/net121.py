"""Alias so that drivers which select a net by number (importlib "net%d") can run net12_rc: net=121 -> net12_rc.Tissue.
Added for RCGENERAL_REPORT.md (23 November 2026); nothing else changes. Options rc=, rc_gate= etc. pass through to net12_rc."""
from net12_rc import Tissue  # noqa: F401
