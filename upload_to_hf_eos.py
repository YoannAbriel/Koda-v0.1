"""
Build EOS-fixed KodaLite Flax safetensors at hf_upload_eos/.
Same logic as upload_to_hf.py but loads the EOS-finetuned LoRA on top of SFT v1.
"""
import jax, jax.numpy as jnp, flax.nnx as nnx, numpy as np
import orbax.checkpoint
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken
import json
from pathlib import Path
from safetensors.numpy import save_file

from model import MiniGPT, CONFIGS
from lora import inject_lora, LoRALinear
from config import CHECKPOINT_DIR, HF_UPLOAD_EOS_DIR, LORA_EOS_DIR

OUTPUT_DIR = Path(str(HF_UPLOAD_EOS_DIR))
PRETRAINED = f"{CHECKPOINT_DIR}/step_100000.orbax"
LORA_EOS = f"{LORA_EOS_DIR}/lora_eos_step_000200.orbax"

devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=("data",))
config = CONFIGS["xl"].copy()
tokenizer = tiktoken.get_encoding("gpt2")
config["vocab_size"] = tokenizer.n_vocab

print(f"Loading base from {PRETRAINED}...", flush=True)
with mesh:
    model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=False, rngs=nnx.Rngs(0))

sharding = NamedSharding(mesh, P())
ra = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
cp = orbax.checkpoint.PyTreeCheckpointer()
nnx.update(model, cp.restore(PRETRAINED, item=nnx.state(model), restore_args=ra))

print("Inject LoRA...", flush=True)
model = inject_lora(model, rank=16, alpha=32.0, rngs=nnx.Rngs(42))

print(f"Load EOS LoRA from {LORA_EOS}...", flush=True)
ra2 = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
nnx.update(model, cp.restore(LORA_EOS, item=nnx.state(model), restore_args=ra2))

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
print("LoRA merged", flush=True)

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Saving to {OUTPUT_DIR}...", flush=True)

state = nnx.state(model, nnx.Param)
tensors = {}
for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    key = jax.tree_util.keystr(path).replace("/", ".").lstrip(".")
    arr = np.array(leaf)
    if arr.dtype == object or "PRNG" in str(arr.dtype):
        continue
    tensors[key] = arr

save_file(tensors, str(OUTPUT_DIR / "model.safetensors"))
print(f"Saved {len(tensors)} tensors", flush=True)
