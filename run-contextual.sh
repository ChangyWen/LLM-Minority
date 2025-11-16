set -x

attribute_type=$1
resume_count=$2
total_count=$3

for ((i=0; i<total_count; i++)); do
    nohup python src/contextual.py "$attribute_type" $resume_count >/dev/null 2>&1 &
    sleep 10
done
