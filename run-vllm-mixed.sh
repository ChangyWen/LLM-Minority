set -x

application=$1
model=$2
attribute_type=$3
gpu_count=$4
disable_thinking=$5


if [ "$application" == "hiring" ]; then
  pool_count=200
  max_model_len=24576
elif [ "$application" == "loan" ] || [ "$application" == "edu" ]; then
  pool_count=500
  max_model_len=10240
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

python src/$application/contextual.py $model "$attribute_type" 2 $pool_count $disable_thinking &
sleep 5
python src/$application/contextual.py $model "$attribute_type" 4 $pool_count $disable_thinking &
sleep 5
python src/$application/contextual.py $model "$attribute_type" 6 $pool_count $disable_thinking &
sleep 5
python src/$application/contextual.py $model "$attribute_type" 8 $pool_count $disable_thinking &
sleep 5
python src/$application/contextual.py $model "$attribute_type" 10 $pool_count $disable_thinking
