#!/bin/bash
cd "$(dirname "$0")"
python3 search_arch.py G1 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=2"
python3 search_arch.py G2 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=1 c_k_win=4"
python3 search_arch.py G3 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 M=32"
python3 search_arch.py G4 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=4 c_k_win=4"
