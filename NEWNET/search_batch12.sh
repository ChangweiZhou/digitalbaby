#!/bin/bash
cd "$(dirname "$0")"
T="tests=moby,code,keep,keep4,keep2,mem"
K8="L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75"
N1="L=8 cC=64 M=32 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4"
for s in 2 3; do
python3 search_arch.py P-K8-s$s "$K8 seed=$s" $T
python3 search_arch.py P-N1-s$s "$N1 seed=$s" $T
done
