set -x

total_count=$1

for ((i=0; i<total_count; i++)); do
    nohup python src/rank-resumes-rand.py >/dev/null 2>&1 &
    sleep 0.5
done
