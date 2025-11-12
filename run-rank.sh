set -x

gender=$1
gender_order_id=$2
total_count=$3

for ((i=0; i<total_count; i++)); do
    nohup python src/rank-resumes-$gender.py $gender_order_id >/dev/null 2>&1 &
    sleep 0.1
done
