"""
Phase 3: Extend context 1024 -> 4096 with NTK-aware RoPE.

1. Modify RoPE base from 10000 to 10000 * (4^(64/62)) for NTK scaling
2. Recompute RoPE frequencies for new maxlen
3. Fine-tune ~1000 steps on long documents (PG19 or similar)
"""
import jax, jax.numpy as jnp, flax.nnx as nnx, optax, numpy as np
import orbax.checkpoint
import re
import sys
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken

from model import MiniGPT, CONFIGS, precompute_rope_frequencies
from lora import inject_lora
from datasets import load_dataset
from config import LONG_CONTEXT_DIR, MARKERS_DIR, SFT_V2_DIR

DONE_MARKER = Path(f'{MARKERS_DIR}/phase3.done')
DONE_MARKER.parent.mkdir(exist_ok=True)
if DONE_MARKER.exists():
    print('Phase 3 already done', flush=True)
    sys.exit(0)

# Find latest SFT v2 checkpoint
SFT_V2_DIR = Path(str(SFT_V2_DIR))
pattern = re.compile(r'sft_v2_step_(\d+)\.orbax')
max_step = 0
max_path = None
for p in SFT_V2_DIR.iterdir():
    m = pattern.match(p.name)
    if m:
        step = int(m.group(1))
        if step > max_step:
            max_step = step
            max_path = p
print(f'Loading from: {max_path}', flush=True)

NEW_MAXLEN = 2048
ORIG_MAXLEN = 1024
SCALE = NEW_MAXLEN / ORIG_MAXLEN  # 4
LONG_CKPT_DIR = Path(str(LONG_CONTEXT_DIR))

devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=('data',))
config = CONFIGS['xl'].copy()
tokenizer = tiktoken.get_encoding('gpt2')
config['vocab_size'] = tokenizer.n_vocab
config['maxlen'] = NEW_MAXLEN  # Override

print(f'Init model with maxlen={NEW_MAXLEN}...', flush=True)
with mesh:
    model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))

# NTK-aware: increase RoPE base
# new_base = base * scale^(d/(d-2))
head_dim = config['embed_dim'] // config['num_heads']
new_base = 10000.0 * (SCALE ** (head_dim / (head_dim - 2)))
print(f'NTK-aware RoPE: new base = {new_base:.0f} (was 10000)', flush=True)

# Recompute RoPE frequencies for all attention layers
new_cos, new_sin = precompute_rope_frequencies(head_dim, NEW_MAXLEN, base=new_base, dtype=jnp.float32)
for block in model.blocks:
    object.__setattr__(block.attention, 'rope_cos', new_cos)
    object.__setattr__(block.attention, 'rope_sin', new_sin)
print('RoPE updated for all layers', flush=True)

# Inject LoRA + load SFT v2 weights
model = inject_lora(model, rank=16, alpha=32.0, rngs=nnx.Rngs(42))
sharding = NamedSharding(mesh, P())
ra = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
cp = orbax.checkpoint.PyTreeCheckpointer()
nnx.update(model, cp.restore(str(max_path), item=nnx.state(model), restore_args=ra))
print('Weights loaded', flush=True)

# Re-apply NTK RoPE AFTER checkpoint load (checkpoint overwrites the arrays)
print('Re-applying NTK RoPE after checkpoint load...', flush=True)
for block in model.blocks:
    object.__setattr__(block.attention, 'rope_cos', new_cos)
    object.__setattr__(block.attention, 'rope_sin', new_sin)
print('RoPE re-applied', flush=True)

# Long-context training data (PG19 books)
print('Loading PG19...', flush=True)
ds = load_dataset('emozilla/pg19', split='train', streaming=True)

def long_batch_gen(batch_size=2):
    end_token = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
    buf = []
    for ex in ds:
        text = ex['text']
        toks = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        toks.append(end_token)
        buf.extend(toks)
        while len(buf) >= NEW_MAXLEN * batch_size:
            chunk = buf[:NEW_MAXLEN * batch_size]
            buf = buf[NEW_MAXLEN * batch_size:]
            arr = np.array(chunk).reshape(batch_size, NEW_MAXLEN)
            yield jnp.array(arr, dtype=jnp.int32)

MAX_STEPS = 1000
PEAK_LR = 3e-5

def label_fn(state):
    return jax.tree_util.tree_map(lambda x: 'train' if x.dtype == jnp.float32 else 'freeze', state)

warmup_steps = max(1, MAX_STEPS // 20)
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=PEAK_LR,
    warmup_steps=warmup_steps, decay_steps=MAX_STEPS,
    end_value=PEAK_LR * 0.1,
)
tx = optax.multi_transform(
    {
        'train': optax.chain(optax.zero_nans(), optax.clip_by_global_norm(0.5),
                              optax.adamw(learning_rate=lr_schedule, weight_decay=0.0)),
        'freeze': optax.set_to_zero(),
    },
    label_fn,
)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

def loss_fn(model, batch):
    tokens = batch
    targets = jnp.concatenate([tokens[:, 1:], jnp.zeros((tokens.shape[0], 1), dtype=tokens.dtype)], axis=1)
    logits = model(tokens, deterministic=True)
    logits_f32 = jnp.clip(logits.astype(jnp.float32), -25.0, 25.0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, targets).mean()
    return loss, logits

@nnx.jit
def train_step(model, opt, batch):
    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    gnorm = optax.global_norm(grads)
    opt.update(model, grads)
    return loss, gnorm

LONG_CKPT_DIR.mkdir(exist_ok=True)
save_cp = orbax.checkpoint.PyTreeCheckpointer()

print(f'Phase 3: Long-context fine-tune at maxlen={NEW_MAXLEN}, {MAX_STEPS} steps', flush=True)
recent_losses = []
step = 0
for batch in long_batch_gen(batch_size=2):
    if step >= MAX_STEPS:
        break
    with mesh:
        shard = NamedSharding(mesh, P('data'))
        bs = jax.device_put(batch, shard)
        loss, gnorm = train_step(model, optimizer, bs)
    loss_val = float(loss)
    if loss_val == loss_val and 0.0 <= loss_val <= 100.0:
        recent_losses.append(loss_val)
    if (step + 1) % 25 == 0:
        avg = sum(recent_losses) / max(len(recent_losses), 1)
        lr = lr_schedule(step)
        print(f'Step {step+1}/{MAX_STEPS} | Loss: {avg:.4f} | LR: {lr:.2e}', flush=True)
        recent_losses = []
    if (step + 1) % 250 == 0:
        path = LONG_CKPT_DIR / f'long_step_{step+1:06d}.orbax'
        save_cp.save(path, nnx.state(model), force=True)
        print(f'  Saved: {path}', flush=True)
    step += 1

# Final save
final = LONG_CKPT_DIR / f'long_step_{step:06d}.orbax'
save_cp.save(final, nnx.state(model), force=True)
print(f'Saved final: {final}', flush=True)

DONE_MARKER.write_text('completed')
print('Phase 3 done', flush=True)
