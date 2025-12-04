set -x

# model=Qwen/Qwen3-Next-80B-A3B-Instruct
# model=meta-llama/Llama-3.3-70B-Instruct
# model=openai/gpt-oss-120b
application=$1
model=$2
gpu_count=$3

if [ "$gpu_count" == "" ]; then
  gpu_count=8
fi

######### start vllm server #########
nohup vllm serve $model \
  --trust-remote-code \
  --tensor-parallel-size $gpu_count \
  --max-model-len 5120 \
  --port 8000 >/dev/null 2>&1 &

echo "*********** Waiting for vllm server to start ***********"
sleep 600
echo "*********** Done waiting ***********"

python src/$application/societal.py $model "Gender Identity" &
sleep 5
python src/$application/societal.py $model "Gender Identity" &
sleep 5
python src/$application/societal.py $model "Sexual Orientation" &
sleep 5
python src/$application/societal.py $model "Sexual Orientation"
# python src/$application/societal.py $model "Race" &
# sleep 5
# python src/$application/societal.py $model "Race" &
# sleep 5
# python src/$application/societal.py $model "Religious Affiliation" &
# sleep 5
# python src/$application/societal.py $model "Religious Affiliation"