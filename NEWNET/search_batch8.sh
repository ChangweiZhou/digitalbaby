#!/bin/bash
cd "$(dirname "$0")"
T="tests=moby,code,keep,keep4,keep2,mem"
python3 search_arch.py H1 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 seed=2" $T
python3 search_arch.py H2 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 seed=3" $T
python3 search_arch.py H3 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 c_th_on=0.9 c_th_match=0.75" $T
python3 search_arch.py H4 "L=12 M=11 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4" $T
python3 search_arch.py H5 "L=6 M=21 joint=1 mesh=2 mesh_base=0 mesh_dag=1 c_k_win=4" $T
python3 search_arch.py H6 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=1" $T
