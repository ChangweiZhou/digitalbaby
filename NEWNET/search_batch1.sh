#!/bin/bash
# first batch of architectures, same size (L=8, M=16); one candidate at a time, its four tests in parallel
cd "$(dirname "$0")"
D="mesh_base=0 mesh_dag=1"
python3 search_arch.py A0 "L=8 joint=1"
python3 search_arch.py A1 "L=8 joint=1 mesh=2 $D"
python3 search_arch.py A2 "L=8 joint=1 mesh=1 $D"
python3 search_arch.py A3 "L=8 joint=0 mesh=2 $D"
python3 search_arch.py A4 "L=8 joint=0"
python3 search_arch.py A5 "L=8 joint=1 mesh=2 $D hold=0"
python3 search_arch.py A6 "L=8 joint=1 mesh=4 $D"
python3 search_arch.py A7 "L=8 joint=1 mesh=2 mesh_base=1"
