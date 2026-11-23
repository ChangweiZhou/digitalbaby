# runs the job list 4 at a time (OMP_NUM_THREADS=1); each job: name|command; log rccarry_runs/<name>.log
cd /home/user/digitalbaby/NEWNET; export OMP_NUM_THREADS=1
grep -v '^#' ${1:-rccarry_runs/jobs.txt} | xargs -P 4 -I{} bash -c 'j="{}"; n=${j%%|*}; c=${j#*|}; $c > rccarry_runs/$n.log 2>&1'
