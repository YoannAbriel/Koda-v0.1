"""
Upload Koda-v0.1 (KodaLite-1.3B) to HuggingFace.
"""
import jax, jax.numpy as jnp, flax.nnx as nnx, numpy as np
import orbax.checkpoint
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken
import json
from pathlib import Path
from safetensors.numpy import save_file
from huggingface_hub import HfApi, create_repo

from model import MiniGPT, CONFIGS
from lora import inject_lora, LoRALinear

REPO_ID = "yoann/KodaLite-1.3B"
OUTPUT_DIR = Path("/opt/yoann-test/hf_upload")

devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=("data",))
config = CONFIGS["xl"].copy()
tokenizer = tiktoken.get_encoding("gpt2")
config["vocab_size"] = tokenizer.n_vocab

# 1. Load base model
print("Loading base model (step_100000)...", flush=True)
with mesh:
    model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=False, rngs=nnx.Rngs(0))

sharding = NamedSharding(mesh, P())
ra = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
cp = orbax.checkpoint.PyTreeCheckpointer()
nnx.update(model, cp.restore("/opt/yoann-test/checkpoints/step_100000.orbax", item=nnx.state(model), restore_args=ra))

# 2. Inject LoRA + load SFT weights
print("Inject LoRA + load SFT v1...", flush=True)
model = inject_lora(model, rank=16, alpha=32.0, rngs=nnx.Rngs(42))
ra2 = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
nnx.update(model, cp.restore("/opt/yoann-test/lora_checkpoints/lora_step_014283.orbax", item=nnx.state(model), restore_args=ra2))

# 3. Merge LoRA into base weights
print("Merging LoRA into base weights...", flush=True)
for block in model.blocks:
    attn = block.attention
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        layer = getattr(attn, name)
        if isinstance(layer, LoRALinear):
            base_weight = layer.base.kernel[...]
            lora_a = layer.lora_a[...]
            lora_b = layer.lora_b[...]
            scaling = layer.scaling
            delta = (lora_a @ lora_b * scaling).astype(base_weight.dtype)
            merged_weight = base_weight + delta
            new_linear = nnx.Linear(
                merged_weight.shape[0], merged_weight.shape[1],
                use_bias=False, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0),
            )
            new_linear.kernel = nnx.Param(merged_weight)
            setattr(attn, name, new_linear)

print("LoRA merged!", flush=True)

# 4. Save as safetensors
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...", flush=True)

state = nnx.state(model, nnx.Param)  # Only trainable params, skip RNGs
tensors = {}
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    key = jax.tree_util.keystr(path).replace("/", ".").lstrip(".")
    arr = np.array(leaf)
    if arr.dtype == object or "PRNG" in str(arr.dtype):
        continue
    tensors[key] = arr

save_file(tensors, str(OUTPUT_DIR / "model.safetensors"))
print(f"Saved {len(tensors)} tensors", flush=True)

# 5. Config
model_config = {
    "model_type": "kodalite",
    "architecture": "KodaLite",
    "vocab_size": config["vocab_size"],
    "embed_dim": config["embed_dim"],
    "num_heads": config["num_heads"],
    "num_kv_heads": config["num_kv_heads"],
    "ff_dim": config["ff_dim"],
    "num_blocks": config["num_blocks"],
    "maxlen": config["maxlen"],
    "dropout_rate": config["dropout_rate"],
    "rope_base": 10000,
    "total_params": "1.27B",
    "dtype": "bfloat16",
    "tokenizer": "gpt2 (tiktoken)",
    "training": {
        "pretraining_tokens": "1.64B",
        "pretraining_dataset": "SlimPajama-6B",
        "sft_dataset": "Dolly-15K + OpenAssistant OASST1",
        "sft_method": "LoRA (rank=16, alpha=32)",
        "framework": "JAX + Flax NNX",
    },
}

with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(model_config, f, indent=2)

# 6. Model card
readme = """---
language: en
license: apache-2.0
tags:
  - text-generation
  - jax
  - flax
  - transformer
  - from-scratch
---

# KodaLite-1.3B (Koda-v0.1)

A 1.27B parameter decoder-only transformer language model, trained **entirely from scratch** using JAX + Flax NNX.

## Architecture (LLaMA-inspired)

| Component | Details |
|---|---|
| Parameters | 1.27B |
| Layers | 24 transformer blocks |
| Attention | Grouped Query Attention (32Q / 8KV heads) |
| FFN | SwiGLU activation |
| Normalization | RMSNorm (pre-norm) |
| Position | RoPE (Rotary Position Embeddings) |
| Context | 1024 tokens |
| Vocab | 50,257 tokens (GPT-2 tokenizer) |

## Training

### Pre-training
- **Dataset**: SlimPajama-6B (streaming)
- **Tokens seen**: 1.64B
- **Hardware**: 2x NVIDIA L40S (96GB VRAM total)
- **Precision**: bfloat16
- **Duration**: ~25 hours

### SFT (Supervised Fine-Tuning)
- **Datasets**: Databricks Dolly-15K + OpenAssistant OASST1
- **Method**: LoRA (rank=16, alpha=32)
- **Trainable params**: 5.11M (0.4% of total)
- **Loss masking**: Only assistant tokens contribute to loss

## Chat Format

```
<|user|>
Your question here
<|assistant|>
Model response
<|end|>
```

## Limitations

- Small model (1.27B) with limited factual knowledge
- Undertrained (1.64B tokens vs Chinchilla-optimal 25B)
- May produce repetitive or inaccurate responses
- English only
- Educational project, not production-ready

## Built With

- JAX 0.9.2 + Flax NNX 0.12
- Optax (AdamW optimizer)
- tiktoken (GPT-2 tokenizer)
- Trained on 2x NVIDIA L40S GPUs

## License

Apache 2.0
"""

with open(OUTPUT_DIR / "README.md", "w") as f:
    f.write(readme)

print("Config and README saved", flush=True)
for f in OUTPUT_DIR.iterdir():
    size = f.stat().st_size
    print(f"  {f.name}: {size / 1e6:.1f} MB", flush=True)

# 7. Upload
print(f"\nUploading to {REPO_ID}...", flush=True)
try:
    api = HfApi()
    create_repo(REPO_ID, exist_ok=True)
    api.upload_folder(folder_path=str(OUTPUT_DIR), repo_id=REPO_ID)
    print(f"\nDone! https://huggingface.co/{REPO_ID}", flush=True)
except Exception as e:
    print(f"Upload failed: {e}", flush=True)
    print(f"Files saved in {OUTPUT_DIR} — upload manually.", flush=True)
