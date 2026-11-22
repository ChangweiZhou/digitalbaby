#!/bin/bash
cd "$(dirname "$0")"
T="tests=moby,code,keep,keep4,keep2,mem"
B="L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=1"
python3 search_arch.py I1 "$B h_g_k=0.005 h_g_fan=16" $T
python3 search_arch.py I2 "$B h_g_k=0.002 h_g_fan=16" $T
python3 search_arch.py I3 "$B h_g_k=0.005 h_g_fan=16 nG=8192" $T
python3 search_arch.py I4 "$B h_g_k=0.005 h_g_fan=16 c_k_win=4" $T
