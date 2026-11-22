#!/bin/bash
cd "$(dirname "$0")"
python3 search_arch.py A8 "L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2"
python3 search_arch.py A9 "L=8 joint=1 mesh=3 mesh_base=0 mesh_dag=2"
python3 search_arch.py A10 "L=12 M=12 joint=1 mesh=2 mesh_base=0 mesh_dag=2"
