#!/bin/bash
# Run my_bench.py on all comparison models + KodaLite
set -e
cd /opt/yoann-test
source .venv_bench/bin/activate

OUT=/opt/yoann-test/bench_results
mkdir -p "$OUT"

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
  out="$OUT/$name.json"

  if [ -f "$out" ]; then
    echo "=== SKIP $name (already done) ==="
    continue
  fi

  echo ""
  echo "=============================="
  echo "=== $name ($model_id) ==="
  echo "=============================="
  python3 my_bench.py "$model_id" "$out" 2>&1 | tee "$OUT/$name.log"
done

echo ""
echo "=== ALL DONE ==="
ls -la "$OUT/"
