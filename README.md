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
model.py                  # Architecture (RoPE, GQA, RMSNorm, SwiGLU)
lora.py                   # LoRA injection / merge
data.py                   # SlimPajama streaming loader
sft_data.py               # Dolly + OASST formatters
sft_loader.py             # SFT dataloader (assistant-only loss mask)
sft_loader_eos.py         # Same loader, appends <|endoftext|> after <|end|>

train.py                  # From-scratch pretrain (phase 1)
train_continue.py         # Resume pretrain to target step count
phase1.py / phase2.py / phase3.py  # Markered phase wrappers
sft_train.py / sft_lora.py         # SFT entry points
sft_eos_finetune.py       # 200-step LoRA pass to teach EOS token after <|end|>
orchestrator.py           # Runs phases 1->2->3 sequentially with markers

convert_to_llama.py       # Flax NNX safetensors -> HF LlamaForCausalLM
upload_to_hf.py           # Build HF repo from JAX state and push
upload_to_hf_eos.py       # Same but loads the EOS LoRA on top
gguf_pipeline.sh          # HF -> GGUF F16 -> Q8_0 / Q4_K_M (needs llama.cpp)
generate.py               # Quick JAX generation demo

my_bench.py               # Custom zero-shot benchmark (8 tasks)
make_charts.py            # Bench result plots

test_eos_emission.py      # Verify model emits token 50256 after <|end|>
test_hf_eos.py            # Same check on HF Transformers
test_koda_final.py        # End-to-end JAX chat test

notebooks/quickstart.ipynb  # Walkthrough at small scale
```

## Hardware

The full pipeline was run on 2x NVIDIA L40S (96 GB VRAM total, bf16). Training from scratch at this size needs roughly that level of VRAM. To experiment locally, scale down via `model.py:CONFIGS` (the `small` / `medium` configs) before running.

## Setup

```bash
git clone <this-repo>
cd Koda-v0.1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# JAX needs the matching CUDA + cuDNN. On a fresh box:
pip install --upgrade 'nvidia-cudnn-cu12>=9.8' 'numpy>=2.0'

python -c "import jax; print(jax.devices())"
```

## Pipeline (full run)

Hard-coded paths inside the scripts assume `/opt/yoann-test/`. Either edit them or symlink your working dir there. Then:

```bash
# Phase 1: pretrain on SlimPajama from scratch (~1.6B tokens, ~25h on 2x L40S)
python train.py
# To resume from a checkpoint: python train_continue.py

# Phase 2: SFT v2 (LoRA on Dolly + OASST)
python phase2.py

# Phase 3: extend context 1024 -> 2048 with NTK-aware RoPE
python phase3.py

# EOS fix (200 LoRA steps, teaches model to emit <|endoftext|> after <|end|>)
python sft_eos_finetune.py

# Or run all phases sequentially with crash-recovery markers
python orchestrator.py
```

## Pipeline (export and upload)

```bash
# Merge LoRA + dump Flax safetensors
python upload_to_hf_eos.py     # writes hf_upload_eos/

# Convert Flax safetensors to HF LLaMA format (transposes + RoPE permutation)
python convert_to_llama_eos.py # writes hf_llama_eos/

# GGUF (needs llama.cpp built locally)
bash gguf_pipeline.sh          # writes gguf/
```

## Benchmarks

```bash
python my_bench.py             # writes bench_results/<model>.json
python make_charts.py          # writes bench_charts/*.png
```

8 zero-shot tasks: HellaSwag, ARC-E/C, WinoGrande, PIQA, BoolQ, OpenBookQA, LAMBADA-OpenAI. KodaLite-1.3B reaches 36.8% average accuracy, which places it just below GPT-2-124M because of severe Chinchilla undertraining (1.64B tokens for a 1.3B model, ~6.5% of the optimal 25B target). See the model card on HuggingFace for the full table.

## Quickstart notebook

`notebooks/quickstart.ipynb` walks through:
1. Loading the architecture and counting parameters
2. Tokenizing a sample with the GPT-2 BPE
3. Running a tiny pretrain loop on a few hundred SlimPajama steps
4. Doing a 50-step SFT LoRA on Dolly
5. Loading the released checkpoint from HuggingFace and generating

It is meant for sanity-checking the codebase on a small machine before running the full pipeline.

## License

Apache 2.0
