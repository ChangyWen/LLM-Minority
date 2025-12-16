set -x

application=$1
model=$2
attribute_type=$3
gpu_count=$4
disable_thinking=$5
context_size=$6

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

if [ "$context_size" == "10" ]; then
  max_model_len=$((max_model_len * 2))
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

if [ "$context_size" == "5" ]; then
  python src/$application/contextual.py $model "$attribute_type" $context_size $pool_count $disable_thinking &
  sleep 5
  python src/$application/contextual.py $model "$attribute_type" $context_size $pool_count $disable_thinking &
  sleep 5
  python src/$application/contextual.py $model "$attribute_type" $context_size $pool_count $disable_thinking
elif [ "$context_size" == "10" ]; then
  python src/$application/contextual.py $model "$attribute_type" $context_size $pool_count $disable_thinking &
  sleep 5
  python src/$application/contextual.py $model "$attribute_type" $context_size $pool_count $disable_thinking
fi