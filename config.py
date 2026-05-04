"""
Centralized paths for the Koda-v0.1 pipeline.

Override the working directory via the KODA_ROOT environment variable:

    export KODA_ROOT=/path/to/your/work/dir

All checkpoints, markers, exports, and benchmark results live under KODA_ROOT.
The HuggingFace repo IDs are also defined here so a fork can retarget them in
one place rather than grepping every script.
"""
import os
from pathlib import Path

KODA_ROOT = Path(os.environ.get("KODA_ROOT", "./work")).resolve()

# Training artefacts
CHECKPOINT_DIR        = KODA_ROOT / "checkpoints"               # phase 1 pretrain
SFT_V2_DIR            = KODA_ROOT / "sft_v2_checkpoints"        # phase 2 SFT LoRA
LORA_CKPT_DIR         = KODA_ROOT / "lora_checkpoints"          # legacy SFT LoRA
LONG_CONTEXT_DIR      = KODA_ROOT / "long_context_checkpoints"  # phase 3
LORA_EOS_DIR          = KODA_ROOT / "lora_eos_checkpoints"      # EOS retrain

# Phase markers
MARKERS_DIR           = KODA_ROOT / "markers"

# Export targets
HF_UPLOAD_DIR         = KODA_ROOT / "hf_upload"        # Flax-format dump
HF_UPLOAD_EOS_DIR     = KODA_ROOT / "hf_upload_eos"
HF_LLAMA_DIR          = KODA_ROOT / "hf_llama"         # HF Transformers format
HF_LLAMA_EOS_DIR      = KODA_ROOT / "hf_llama_eos"
GGUF_DIR              = KODA_ROOT / "gguf"
GGUF_EOS_DIR          = KODA_ROOT / "gguf_eos"

# Benchmark
BENCH_RESULTS_DIR     = KODA_ROOT / "bench_results"
BENCH_CHARTS_DIR      = KODA_ROOT / "bench_charts"

# llama.cpp checkout (for GGUF conversion + quantization)
LLAMA_CPP_DIR         = Path(os.environ.get("LLAMA_CPP_DIR", KODA_ROOT / "llama.cpp")).resolve()

# HuggingFace repo IDs (override via env if you fork)
HF_REPO_HF            = os.environ.get("HF_REPO_HF",        "YoAbriel/KodaLite-1.3B")
HF_REPO_GGUF          = os.environ.get("HF_REPO_GGUF",      "YoAbriel/KodaLite-1.3B-GGUF")
HF_REPO_MLX_FP16      = os.environ.get("HF_REPO_MLX_FP16",  "YoAbriel/KodaLite-1.3B-mlx")
HF_REPO_MLX_8BIT      = os.environ.get("HF_REPO_MLX_8BIT",  "YoAbriel/KodaLite-1.3B-mlx-8bit")


if __name__ == "__main__":
    print(f"KODA_ROOT = {KODA_ROOT}")
    for name, value in sorted(globals().items()):
        if name.endswith("_DIR") or name.startswith("HF_REPO_"):
            print(f"  {name:22s} = {value}")
