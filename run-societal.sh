set -x

model=$1
attribute_type=$2
pool_count=$3
total_count=$4


for ((i=0; i<total_count; i++)); do
    nohup python src/contextual.py $model "$attribute_type" $resume_count $pool_count >/dev/null 2>&1 &
    sleep 10
done
