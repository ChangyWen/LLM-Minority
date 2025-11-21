set -x

######### start vllm server #########
# nohup vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
nohup vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 12288 \
  --port 8000 >/dev/null 2>&1 &

echo "*********** Waiting for vllm server to start ***********"
sleep 600
echo "*********** Done waiting ***********"

# python src/contextual.py Qwen/Qwen3-Next-80B-A3B-Instruct Gender 5 200 &
python src/contextual.py meta-llama/Llama-3.3-70B-Instruct Gender 5 200 &
sleep 5
# python src/contextual.py Qwen/Qwen3-Next-80B-A3B-Instruct Gender 5 200 &
python src/contextual.py meta-llama/Llama-3.3-70B-Instruct Gender 5 200 &
sleep 5
# python src/contextual.py Qwen/Qwen3-Next-80B-A3B-Instruct Gender 5 200
python src/contextual.py meta-llama/Llama-3.3-70B-Instruct Gender 5 200