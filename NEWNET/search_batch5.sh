#!/bin/bash
# batch 5: architectures from biological principles: parallel streams, feedback, sparseness
cd "$(dirname "$0")"
python3 search_arch.py E1 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=3"
python3 search_arch.py E2 "L=16 cC=64 joint=1 mesh=2 mesh_base=0 mesh_dag=3"
python3 search_arch.py E3 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=4"
python3 search_arch.py E4 "L=8 joint=1 c_k_win=4"
python3 search_arch.py E5 "L=8 joint=1 c_k_win=16"
python3 search_arch.py E6 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4"
