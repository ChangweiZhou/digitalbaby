#!/bin/bash
# Starts the trial's fourth run (place, sweep) when one of the three running E47 processes has ended, keeping at most three at once.
cd "$(dirname "$0")"
until [ "$(pgrep -fc 'exp47_local.py run 32')" -lt 3 ]; do sleep 60; done
exec ./launch_trial.sh place:S
