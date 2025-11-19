set -x

######### start vllm server #########
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --port 8000 &

echo "*********** Waiting for vllm server to start ***********"
sleep 360
echo "*********** Done waiting ***********"

######### run the jobs in parallel #########
NUM_JOBS=3   # change this to however many parallel runs you want

for i in $(seq 1 $NUM_JOBS); do
    echo "=== starting job $i ==="
    python src/contextual.py Qwen/Qwen3-Next-80B-A3B-Instruct Gender 5 &
done

for i in $(seq 1 $NUM_JOBS); do
    echo "=== starting job $i ==="
    python src/contextual.py Qwen/Qwen3-Next-80B-A3B-Instruct Race 5 &
done
