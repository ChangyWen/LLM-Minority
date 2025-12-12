set -x

# model=Qwen/Qwen3-Next-80B-A3B-Instruct
# model=meta-llama/Llama-3.3-70B-Instruct
# model=openai/gpt-oss-120b
application=$1
model=$2
attribute_type=$3
gpu_count=$4
disable_thinking=$5

if [ "$application" == "hiring" ]; then
  pool_count=200
  max_model_len=12288
elif [ "$application" == "loan" ] || [ "$application" == "edu" ]; then
  pool_count=500
  max_model_len=5120
else
  echo "Invalid application: $application"
  exit 1
fi

######### start vllm server #########
nohup vllm serve $model \
  --trust-remote-code \
  --tensor-parallel-size $gpu_count \
  --max-model-len $max_model_len \
  --port 8000 >/dev/null 2>&1 &

echo "*********** Waiting for vllm server to start ***********"
sleep 600
echo "*********** Done waiting ***********"

python src/$application/contextual.py $model "$attribute_type" 5 $pool_count $disable_thinking &
sleep 5
python src/$application/contextual.py $model "$attribute_type" 5 $pool_count $disable_thinking &
sleep 5
python src/$application/contextual.py $model "$attribute_type" 5 $pool_count $disable_thinking