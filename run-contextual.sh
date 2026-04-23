set -x

application=$1
model=$2
attribute_type=$3
resume_count=$4
pool_count=$5
total_count=$6


for ((i=0; i<total_count; i++)); do
    nohup python src/${application}/contextual.py $model "$attribute_type" $resume_count $pool_count >/dev/null 2>&1 &
    sleep 10
done
