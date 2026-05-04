# Koda-v0.1 / KodaLite-1.3B

A 1.27B parameter LLaMA-style decoder-only language model trained from scratch in JAX + Flax NNX on 2x NVIDIA L40S GPUs, plus the full pipeline to convert to HuggingFace Transformers, GGUF (llama.cpp / Ollama / LM Studio), and MLX (Apple Silicon).

Released checkpoints: [YoAbriel/KodaLite-1.3B](https://huggingface.co/YoAbriel/KodaLite-1.3B) (HF), [-mlx](https://huggingface.co/YoAbriel/KodaLite-1.3B-mlx), [-mlx-8bit](https://huggingface.co/YoAbriel/KodaLite-1.3B-mlx-8bit), [-GGUF](https://huggingface.co/YoAbriel/KodaLite-1.3B-GGUF).

## Architecture

| Component | Value |
|---|---|
| Parameters | 1.27B |
| Layers | 24 |
| Hidden size | 2048 |
| Attention | GQA (32 query / 8 KV heads) |
| Head dim | 64 |
| FFN | SwiGLU, intermediate 5504 |
| Normalization | RMSNorm (pre-norm) |
| Position | RoPE (theta=10000) |
| Context | 1024 tokens (extended to 2048 in phase 3) |
| Vocab | 50,257 (GPT-2 BPE via tiktoken) |

LLaMA-compatible: after `convert_to_llama.py`, weights load directly into `LlamaForCausalLM`.

## Layout

```
config.py                 # Centralized paths and HF repo IDs (env-overridable)
model.py                  # Architecture (RoPE, GQA, RMSNorm, SwiGLU)
lora.py                   # LoRA injection / merge

data.py                   # SlimPajama streaming loader
sft_data.py               # Dolly + OASST formatters
sft_loader.py             # SFT dataloader (assistant-only loss mask)
sft_loader_eos.py         # Same loader, appends <|endoftext|> after <|end|>

train.py                  # From-scratch pretrain (phase 1)
train_continue.py         # Resume pretrain to target step count
phase1.py phase2.py phase3.py   # Markered phase wrappers
sft_lora.py               # Standalone LoRA SFT
sft_eos_finetune.py       # ~200-step LoRA pass to emit EOS after <|end|>
orchestrator.py           # Runs phases 1 -> 2 -> 3 sequentially with markers

convert_to_llama.py       # Flax NNX safetensors -> HF LlamaForCausalLM
convert_to_llama_eos.py   # Same, EOS-fixed source
upload_to_hf.py           # Merge LoRA + dump Flax safetensors
upload_to_hf_eos.py       # Same with EOS LoRA on top
gguf_pipeline.sh          # HF -> GGUF F16 -> Q8_0 / Q4_K_M (needs llama.cpp)
generate.py               # Quick JAX generation demo

my_bench.py               # Custom zero-shot benchmark (8 tasks)
run_bench.sh              # Run my_bench.py across the comparison set
make_charts.py            # Bench result plots

test_eos_emission.py      # Verify model emits token 50256 after <|end|> (JAX)
test_hf_eos.py            # Same check on HF Transformers
test_koda_final.py        # End-to-end JAX chat test

notebooks/quickstart.ipynb  # Small-scale walkthrough
```

## Hardware

The full pipeline was run on 2x NVIDIA L40S (96 GB VRAM total, bf16). Training the 1.27B `xl` config from scratch needs roughly that level of VRAM. To experiment locally, drop `MODEL_SIZE` to `'small'` or `'medium'` in `model.py:CONFIGS` before running.

## Setup

```bash
git clone https://github.com/YoannAbriel/Koda-v0.1.git
cd Koda-v0.1

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# JAX needs matching CUDA + cuDNN. On a fresh box:
pip install --upgrade 'nvidia-cudnn-cu12>=9.8' 'numpy>=2.0'

python -c "import jax; print(jax.devices())"
```

## Configure paths

All scripts read paths from `config.py`, which honors a `KODA_ROOT` env var (default `./work`):

```bash
export KODA_ROOT=$PWD/work    # or wherever you want artefacts to land
mkdir -p "$KODA_ROOT"
python config.py              # prints the resolved paths
```

Optional overrides: `LLAMA_CPP_DIR`, `HF_REPO_HF`, `HF_REPO_GGUF`, `HF_REPO_MLX_FP16`, `HF_REPO_MLX_8BIT`.

## Pipeline (full run)

```bash
# Phase 1: pretrain on SlimPajama from scratch (~1.6B tokens, ~25h on 2x L40S)
python train.py
# To resume from the latest checkpoint: python train_continue.py

# Phase 2: SFT (LoRA on Dolly + OASST)
python phase2.py

# Phase 3: extend context 1024 -> 2048 with NTK-aware RoPE
python phase3.py

# EOS fix: 200 LoRA steps, teaches the model to emit <|endoftext|> after <|end|>
python sft_eos_finetune.py

# Or run phases 1 -> 2 -> 3 sequentially with crash-recovery markers
python orchestrator.py
```

## Pipeline (export)

```bash
# 1. Merge LoRA into base, dump Flax safetensors
python upload_to_hf_eos.py            # writes $KODA_ROOT/hf_upload_eos/

# 2. Convert to HF Transformers / LLaMA format (transposes + RoPE permutation)
python convert_to_llama_eos.py        # writes $KODA_ROOT/hf_llama_eos/

# 3. GGUF (needs a llama.cpp checkout at $LLAMA_CPP_DIR)
bash gguf_pipeline.sh                 # writes $KODA_ROOT/gguf_eos/

# 4. MLX (Apple Silicon)
python -m mlx_lm convert --hf-path $KODA_ROOT/hf_llama_eos --mlx-path mlx-fp16 --dtype float16
python -m mlx_lm convert --hf-path $KODA_ROOT/hf_llama_eos --mlx-path mlx-8bit -q --q-bits 8

# 5. Push (any of these)
hf upload $HF_REPO_HF        $KODA_ROOT/hf_llama_eos
hf upload $HF_REPO_GGUF      $KODA_ROOT/gguf_eos
hf upload $HF_REPO_MLX_FP16  mlx-fp16
hf upload $HF_REPO_MLX_8BIT  mlx-8bit
```

## Benchmarks

```bash
bash run_bench.sh             # writes $KODA_ROOT/bench_results/<model>.json
python make_charts.py         # writes $KODA_ROOT/bench_charts/*.png
```

8 zero-shot tasks: HellaSwag, ARC-E/C, WinoGrande, PIQA, BoolQ, OpenBookQA, LAMBADA-OpenAI. KodaLite-1.3B reaches 36.8% average accuracy, which places it just below GPT-2-124M because of severe Chinchilla undertraining (1.64B tokens for a 1.3B model, ~6.5% of the optimal 25B target). See the model card on HuggingFace for the full table.

## Quickstart notebook

`notebooks/quickstart.ipynb` walks through:

1. Architecture sanity check (build the model, count params, run a forward pass).
2. A tiny pretrain step on a couple of SlimPajama batches.
3. Inject LoRA, peek at SFT batches.
4. Load the released checkpoint from HuggingFace (MLX or Transformers) and generate.

It is meant for sanity-checking the codebase on a small machine before running the full pipeline.

## License

Apache 2.0
