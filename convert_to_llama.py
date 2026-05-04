"""
Convert KodaLite-1.3B from custom Flax NNX format to HuggingFace LLaMA format.

This unlocks: AutoModelForCausalLM, vLLM, llama.cpp, Transformers UI on HF hub,
Inference Providers, Deploy button, Model tree, etc.
"""
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from config import HF_LLAMA_DIR, HF_UPLOAD_DIR

SRC = Path(f"{HF_UPLOAD_DIR}/model.safetensors")
OUT_DIR = Path(str(HF_LLAMA_DIR))
OUT_DIR.mkdir(exist_ok=True)

# ----- Load source tensors -----
print("Loading source safetensors...", flush=True)
f = safe_open(str(SRC), framework="pt")
src = {k: f.get_tensor(k) for k in f.keys()}
print(f"Loaded {len(src)} tensors", flush=True)

# ----- Key mapping & transpose -----
# Flax Linear stores weights as [in, out]; PyTorch nn.Linear as [out, in].
# Embedding layers keep [vocab, hidden] (same shape in both frameworks).

def key_for(block, name):
    return f"['blocks'][{block}]['{name}'].value" if name in ("norm1", "norm2") else f"['blocks'][{block}]['{name}']['kernel'].value"


def permute_for_llama_rope(w, num_heads, head_dim):
    """Reorder output dim of a Q/K projection from interleaved (GPT-NeoX/original RoPE)
    to half-rotation (LLaMA RoPE).

    JAX layout per head: [x0, x1, x2, x3, ..., x_{d-2}, x_{d-1}] with pairs (2i, 2i+1)
    LLaMA layout per head: [x0, x2, ..., x_{d-2}, x1, x3, ..., x_{d-1}]

    w shape: [num_heads * head_dim, in] (already transposed to PyTorch layout).
    """
    out_features, in_features = w.shape
    assert out_features == num_heads * head_dim
    w = w.view(num_heads, head_dim, in_features)
    # Gather even-indexed rows then odd-indexed rows
    even = w[:, 0::2, :]
    odd = w[:, 1::2, :]
    w = torch.cat([even, odd], dim=1)  # [num_heads, head_dim, in]
    return w.reshape(out_features, in_features).contiguous()


def remap(src):
    out = {}
    # Token embeddings
    out["model.embed_tokens.weight"] = src["['token_emb']['embedding'].value"]
    # Final norm
    out["model.norm.weight"] = src["['norm_final']['scale'].value"]
    # LM head (Flax: [hidden, vocab] → HF: [vocab, hidden])
    out["lm_head.weight"] = src["['output_layer']['kernel'].value"].T.contiguous()

    # Blocks
    block_indices = sorted({
        int(m.group(1))
        for k in src
        if (m := re.match(r"\['blocks'\]\[(\d+)\]", k))
    })

    hidden = 2048
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = hidden // num_q_heads  # 64

    for i in block_indices:
        prefix = f"model.layers.{i}"
        # Norms (RMSNorm scale, shape [hidden] — no transpose)
        out[f"{prefix}.input_layernorm.weight"] = src[f"['blocks'][{i}]['norm1']['scale'].value"]
        out[f"{prefix}.post_attention_layernorm.weight"] = src[f"['blocks'][{i}]['norm2']['scale'].value"]
        # Attention projections (Flax [in, out] → HF [out, in])
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = src[f"['blocks'][{i}]['attention']['{name}']['kernel'].value"].T.contiguous()
            # Fix RoPE layout for Q and K only (V and O aren't rotated)
            if name == "q_proj":
                w = permute_for_llama_rope(w, num_q_heads, head_dim)
            elif name == "k_proj":
                w = permute_for_llama_rope(w, num_kv_heads, head_dim)
            out[f"{prefix}.self_attn.{name}.weight"] = w
        # FFN SwiGLU (gate, up, down)
        for src_name, dst_name in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            w = src[f"['blocks'][{i}]['ffn']['{src_name}']['kernel'].value"]
            out[f"{prefix}.mlp.{dst_name}.weight"] = w.T.contiguous()

    return out, len(block_indices)


print("Remapping keys...", flush=True)
out_tensors, num_layers = remap(src)
print(f"Remapped to {len(out_tensors)} tensors, {num_layers} layers", flush=True)

# ----- Save safetensors -----
print("Saving converted safetensors...", flush=True)
save_file(out_tensors, str(OUT_DIR / "model.safetensors"), metadata={"format": "pt"})

# ----- LLaMA config -----
hidden = 2048
num_heads = 32
kv_heads = 8
head_dim = hidden // num_heads  # 64
intermediate = 5504

config = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 50257,
    "hidden_size": hidden,
    "intermediate_size": intermediate,
    "num_hidden_layers": num_layers,
    "num_attention_heads": num_heads,
    "num_key_value_heads": kv_heads,
    "head_dim": head_dim,
    "hidden_act": "silu",
    "max_position_embeddings": 1024,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "bos_token_id": 50256,
    "eos_token_id": 50256,
    "pad_token_id": 50256,
    "initializer_range": 0.02,
    "use_cache": True,
    "transformers_version": "4.46.0",
}
with open(OUT_DIR / "config.json", "w") as fp:
    json.dump(config, fp, indent=2)
print("Config written", flush=True)

# ----- Tokenizer (GPT-2 + chat template) -----
print("Loading GPT-2 tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token

chat_template = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}<|user|>\n{{ m['content'] }}\n<|assistant|>\n"
    "{% elif m['role'] == 'assistant' %}{{ m['content'] }}<|endoftext|>"
    "{% endif %}{% endfor %}"
)
tok.chat_template = chat_template
tok.save_pretrained(str(OUT_DIR))
print("Tokenizer saved", flush=True)

# ----- Model card -----
readme = """---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - text-generation
  - llama
  - from-scratch
  - jax
base_model: YoAbriel/KodaLite-1.3B
---

# KodaLite-1.3B (Koda-v0.1)

A **1.27B** parameter LLaMA-style decoder-only language model, trained **entirely from scratch** on 2x NVIDIA L40S GPUs using JAX + Flax NNX, then converted to HuggingFace Transformers format.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "YoAbriel/KodaLite-1.3B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False).to(model.device)
out = model.generate(inputs, max_new_tokens=80, temperature=0.7, do_sample=True)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Architecture (LLaMA-compatible)

| Component | Value |
|---|---|
| Parameters | 1.27B |
| Layers | 24 |
| Hidden size | 2048 |
| Attention | Grouped Query Attention (32Q / 8KV heads) |
| Head dim | 64 |
| FFN | SwiGLU, intermediate size 5504 |
| Normalization | RMSNorm (pre-norm) |
| Position | RoPE (theta=10000) |
| Context | 1024 tokens |
| Vocab | 50,257 (GPT-2 BPE) |

## Training

### Pre-training
- **Dataset**: SlimPajama-6B (streaming)
- **Tokens seen**: 1.64B
- **Hardware**: 2x NVIDIA L40S (96GB VRAM total)
- **Precision**: bfloat16
- **Duration**: ~25 hours
- **Framework**: JAX + Flax NNX (trained from scratch, no base model)

### SFT (Supervised Fine-Tuning)
- **Datasets**: Databricks Dolly-15K + OpenAssistant OASST1
- **Method**: LoRA (rank=16, alpha=32), then merged into base weights
- **Trainable params**: 5.11M (0.4% of total)
- **Loss masking**: Only assistant tokens contribute to loss

## Chat Format

The model expects:

```
<|user|>
Your question
<|assistant|>
Model response<|endoftext|>
```

## Limitations

- Small model (1.27B params) — limited factual knowledge
- Significantly undertrained (1.64B tokens vs Chinchilla-optimal ~25B for this size)
- May produce repetitive or inaccurate responses
- English only
- Educational / research project — not production-ready

## License

Apache 2.0
"""
with open(OUT_DIR / "README.md", "w") as fp:
    fp.write(readme)

print("\nFiles written to", OUT_DIR)
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")

# ----- Quick sanity load with Transformers -----
print("\nSanity check: loading with transformers...", flush=True)
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(str(OUT_DIR), torch_dtype=torch.bfloat16)
print(f"Loaded OK. Total params: {sum(p.numel() for p in m.parameters()) / 1e9:.3f}B")

# Upload separately with `hf upload <repo_id> <OUT_DIR>` if you want to push.
