#!/bin/bash
# E43 addendum 1: the three rate runs (seed 73129, backpropagation), one process each.
cd "$(dirname "$0")/../../.."
for lr in 0.01 0.03 0.1; do python3 e43_addendum.py plain B 73129 $lr > exp45/e43/addendum1/logs/plain_lr${lr}_B_73129.log 2>&1 & done
wait
