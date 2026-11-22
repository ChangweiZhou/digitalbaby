#!/bin/bash
cd "$(dirname "$0")"
python3 search_arch.py F1 "L=16 cC=64 joint=1 mesh=2 mesh_base=0 mesh_dag=5"
python3 search_arch.py F2 "L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5"
python3 search_arch.py F3 "L=16 joint=1 mesh=2 mesh_base=0 mesh_dag=5 M=8"
