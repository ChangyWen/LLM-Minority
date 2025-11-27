set -x

# model=Qwen/Qwen3-Next-80B-A3B-Instruct
# model=meta-llama/Llama-3.3-70B-Instruct
# model=openai/gpt-oss-120b
model=$1

######### start vllm server #########
nohup vllm serve $model \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 5120 \
  --port 8000 >/dev/null 2>&1 &

echo "*********** Waiting for vllm server to start ***********"
sleep 600
echo "*********** Done waiting ***********"

python src/societal.py $model "Gender Identity" 200 &
sleep 5
python src/societal.py $model "Sexual Orientation" 200 &
sleep 5
python src/societal.py $model "Disability Status" 200 &
sleep 5
python src/societal.py $model "Chronic Health Condition Status" 200