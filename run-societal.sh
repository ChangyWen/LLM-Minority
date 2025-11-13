set -x

attribute_type=$1
total_count=$2

for ((i=0; i<total_count; i++)); do
    nohup python src/societal.py $attribute_type >/dev/null 2>&1 &
    sleep 0.5
done
