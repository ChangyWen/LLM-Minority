set -x

# model=Qwen/Qwen3-Next-80B-A3B-Instruct
# model=meta-llama/Llama-3.3-70B-Instruct
# model=openai/gpt-oss-120b
application=$1
model=$2
attribute_type=$3

######### start vllm server #########
nohup vllm serve $model \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 12288 \
  --port 8000 >/dev/null 2>&1 &

echo "*********** Waiting for vllm server to start ***********"
sleep 600
echo "*********** Done waiting ***********"


if [ "$application" == "hiring" ]; then
  pool_count=200
elif [ "$application" == "loan" ]; then
  pool_count=500
else
  echo "Invalid application: $application"
  exit 1
fi

python src/${application}/contextual.py $model "$attribute_type" 5 $pool_count &
sleep 5
python src/${application}/contextual.py $model "$attribute_type" 5 $pool_count &
sleep 5
python src/${application}/contextual.py $model "$attribute_type" 5 $pool_count