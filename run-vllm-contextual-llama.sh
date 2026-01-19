set -x

application=$1
model=$2
attribute_type=$3
gpu_count=$4


if [ "$application" == "hiring" ]; then
  max_model_len=12288
elif [ "$application" == "loan" ] || [ "$application" == "edu" ]; then
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
sleep 300
echo "*********** Done waiting ***********"


python src/$application/contextual-llama.py $model "$attribute_type" &
sleep 5
python src/$application/contextual-llama.py $model "$attribute_type" &
sleep 5
python src/$application/contextual-llama.py $model "$attribute_type"
