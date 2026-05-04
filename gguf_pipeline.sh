#!/bin/bash
# GGUF conversion + quantization + upload for KodaLite-1.3B
set -e
cd /opt/yoann-test
source .venv/bin/activate

GGUF_DIR=/opt/yoann-test/gguf
HF_SRC=/opt/yoann-test/hf_llama
mkdir -p "$GGUF_DIR"

echo "=== [1/4] Building llama-quantize ==="
cd /opt/yoann-test/llama.cpp
if [ ! -f build/bin/llama-quantize ]; then
  cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF 2>&1 | tail -5
  cmake --build build --target llama-quantize -j 4 2>&1 | tail -5
fi
echo "llama-quantize: $(ls -la build/bin/llama-quantize 2>/dev/null || echo FAIL)"

echo "=== [2/4] Converting HF -> GGUF F16 ==="
python convert_hf_to_gguf.py "$HF_SRC" --outfile "$GGUF_DIR/kodalite-f16.gguf" --outtype f16 2>&1 | tail -10

echo "=== [3/4] Quantizing Q4_K_M and Q8_0 ==="
./build/bin/llama-quantize "$GGUF_DIR/kodalite-f16.gguf" "$GGUF_DIR/kodalite-Q4_K_M.gguf" Q4_K_M 2>&1 | tail -5
./build/bin/llama-quantize "$GGUF_DIR/kodalite-f16.gguf" "$GGUF_DIR/kodalite-Q8_0.gguf" Q8_0 2>&1 | tail -5
ls -la "$GGUF_DIR"

echo "=== [4/4] README + Upload ==="
cat > "$GGUF_DIR/README.md" << 'EOF'
---
language: en
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
tags:
  - text-generation
  - gguf
  - llama.cpp
  - ollama
base_model: YoAbriel/KodaLite-1.3B
---

# KodaLite-1.3B — GGUF quantizations

GGUF versions of [YoAbriel/KodaLite-1.3B](https://huggingface.co/YoAbriel/KodaLite-1.3B), compatible with **llama.cpp**, **Ollama**, **LM Studio**, and **Jan**.

## Files

| File | Quant | Size | Use case |
|---|---|---|---|
| `kodalite-f16.gguf` | F16 | ~2.5 GB | Full precision reference |
| `kodalite-Q8_0.gguf` | Q8_0 | ~1.3 GB | Near-lossless, good for Mac |
| `kodalite-Q4_K_M.gguf` | Q4_K_M | ~800 MB | Best size/quality tradeoff |

## Usage

### llama.cpp
```bash
llama-cli -m kodalite-Q4_K_M.gguf -p "<|user|>\nHello\n<|assistant|>\n" -n 80
```

### Ollama
```bash
cat > Modelfile << 'MF'
FROM ./kodalite-Q4_K_M.gguf
TEMPLATE """<|user|>
{{ .Prompt }}
<|assistant|>
"""
PARAMETER stop "<|endoftext|>"
MF
ollama create kodalite -f Modelfile
ollama run kodalite
```

### LM Studio
Download the `.gguf` file and load it directly.

## License

Apache 2.0
EOF

python - << 'PYEOF'
from huggingface_hub import HfApi, create_repo
repo = "YoAbriel/KodaLite-1.3B-GGUF"
create_repo(repo, exist_ok=True)
HfApi().upload_folder(folder_path="/opt/yoann-test/gguf", repo_id=repo)
print(f"Uploaded! https://huggingface.co/{repo}")
PYEOF

echo "=== DONE ==="
