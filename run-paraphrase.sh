set -x

total_count=$1


for ((i=0; i<total_count; i++)); do
    nohup python src/paraphrase.py $total_count $i >/dev/null 2>&1 &
    sleep 5
done
