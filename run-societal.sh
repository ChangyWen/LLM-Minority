set -x

application=$1
model=$2
attribute_type=$3
total_count=$4


for ((i=0; i<total_count; i++)); do
    nohup python src/$application/societal.py $model "$attribute_type" >/dev/null 2>&1 &
    sleep 10
done
