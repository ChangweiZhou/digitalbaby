#!/bin/sh
# runs the second batch after the first: chain.sh <first batch tag> <second batch file> <script>
cd "$(dirname "$0")"
until [ -f out/$1.done ]; do sleep 10; done
SCRIPT=$3 sh run_batch.sh $2
