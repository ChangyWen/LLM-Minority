set -x

# model=Qwen/Qwen3-Next-80B-A3B-Instruct
# model=meta-llama/Llama-3.3-70B-Instruct
# model=openai/gpt-oss-120b
application=$1
model=$2
gpu_count=$3
disable_thinking=$4
total_count=$5

######### start vllm server #########
nohup vllm serve $model \
  --trust-remote-code \
  --tensor-parallel-size $gpu_count \
  --max-model-len 5120 \
  --port 8000 >/dev/null 2>&1 &

echo "*********** Waiting for vllm server to start ***********"
sleep 300
echo "*********** Done waiting ***********"

for ((i=0; i<total_count; i++)); do
    python src/$application/societal.py $model "Religious Affiliation" $disable_thinking &
    sleep 10
done
python src/$application/societal.py $model "Religious Affiliation" $disable_thinking