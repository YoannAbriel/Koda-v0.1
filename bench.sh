#!/bin/bash
# Benchmark KodaLite + comparable 1B models on standard zero-shot tasks.
set -e
cd /opt/yoann-test
source .venv_bench/bin/activate

RESULTS_DIR=/opt/yoann-test/bench_results
mkdir -p "$RESULTS_DIR"

# Core zero-shot suite — fast, good for small models.
CORE_TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,boolq,openbookqa,lambada_openai"

# Models to evaluate (name:model_id pairs)
MODELS=(
  "kodalite-1.3b:YoAbriel/KodaLite-1.3B"
  "gpt2-124m:openai-community/gpt2"
  "gpt2-medium-355m:openai-community/gpt2-medium"
  "gpt2-large-774m:openai-community/gpt2-large"
  "gpt2-xl-1.5b:openai-community/gpt2-xl"
  "pythia-1b:EleutherAI/pythia-1b"
  "pythia-1.4b:EleutherAI/pythia-1.4b"
  "opt-1.3b:facebook/opt-1.3b"
  "tinyllama-1.1b:TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  model_id="${entry#*:}"
  out="$RESULTS_DIR/$name"

  if [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null)" ]; then
    echo "=== SKIP $name (already done) ==="
    continue
  fi

  echo ""
  echo "=============================="
  echo "=== $name ($model_id) ==="
  echo "=============================="
  mkdir -p "$out"

  lm_eval --model hf \
    --model_args "pretrained=$model_id,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$CORE_TASKS" \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "$out" \
    --log_samples 2>&1 | tee "$out/run.log"
done

echo ""
echo "=== ALL DONE ==="
ls -la "$RESULTS_DIR"
