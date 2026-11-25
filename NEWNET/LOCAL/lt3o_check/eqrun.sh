cd "$(dirname "$0")"
for n in p2_118 m3_144 p2_137 m3_024 m3_016 m3_050; do for code in old new; do for seed in 1 2; do echo "$n $code $seed"; done; done; done | xargs -P 6 -n 3 sh -c 'python3 eq.py "$0" "$1" "$2" >> eq.jsonl 2>> eq.err'
