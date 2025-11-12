set -x

gender_order_id=$1
total_count=$2

for ((i=0; i<total_count; i++)); do
    nohup python src/rank-resumes.py $gender_order_id >/dev/null 2>&1 &
    sleep 0.1
done
