"""
Phase 2: SFT v2 with long responses.

Loads the latest pretrain checkpoint, injects fresh LoRA, trains on
Dolly + OASST filtered for long responses (>= 300 chars).
"""
import jax, jax.numpy as jnp, flax.nnx as nnx, optax, numpy as np
import orbax.checkpoint
import re
import sys
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken

from model import MiniGPT, CONFIGS
from sft_data import dolly_to_messages, oasst_to_conversations
from sft_loader import encode_with_mask, pad_to_maxlen
from lora import inject_lora, count_lora_params
from datasets import load_dataset
from config import CHECKPOINT_DIR, KODA_ROOT, MARKERS_DIR, SFT_V2_DIR

DONE_MARKER = Path(f'{MARKERS_DIR}/phase2.done')
DONE_MARKER.parent.mkdir(exist_ok=True)
if DONE_MARKER.exists():
    print('Phase 2 already done', flush=True)
    sys.exit(0)

SFT_V2_DIR = Path(str(SFT_V2_DIR))
MIN_RESPONSE_CHARS = 300

# Find latest pretrain checkpoint
pattern = re.compile(r'step_(\d+)\.orbax')
max_step = 0
max_path = None
for p in CHECKPOINT_DIR.iterdir():
    m = pattern.match(p.name)
    if m:
        step = int(m.group(1))
        if step > max_step:
            max_step = step
            max_path = p
print(f'Loading from: {max_path} (step {max_step})', flush=True)

devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=('data',))
config = CONFIGS['xl'].copy()
tokenizer = tiktoken.get_encoding('gpt2')
config['vocab_size'] = tokenizer.n_vocab

# Load conversations and filter
print('Loading data...', flush=True)
import random
dolly = load_dataset('databricks/databricks-dolly-15k', split='train')
oasst = load_dataset('OpenAssistant/oasst1', split='train')
dolly_msgs = [dolly_to_messages(ex) for ex in dolly]
oasst_msgs = oasst_to_conversations(oasst)

# Filter: only keep conversations with long assistant responses
def has_long_response(conv):
    for msg in conv:
        if msg['role'] == 'assistant' and len(msg['content']) >= MIN_RESPONSE_CHARS:
            return True
    return False

filtered = [c for c in dolly_msgs + oasst_msgs if has_long_response(c)]
random.seed(42)
random.shuffle(filtered)
print(f'Filtered to {len(filtered):,} conversations with long responses', flush=True)

# Filter bad samples (from phase1 analysis)
import pickle
bad_path = Path(f'{KODA_ROOT}/bad_indices.pkl')
if bad_path.exists():
    with open(bad_path, 'rb') as f:
        bad = set(pickle.load(f))
    # Note: indices are based on the unfiltered shuffled list, so just trust our filter

BATCH_SIZE = 8
MAX_STEPS = 5000  # ~3-4h
PEAK_LR = 5e-5
LOG_EVERY = 25
SAVE_EVERY = 500

# Init model
print('Init model...', flush=True)
with mesh:
    model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))

print('Load pretrain weights...', flush=True)
sharding = NamedSharding(mesh, P())
ra = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
cp = orbax.checkpoint.PyTreeCheckpointer()
nnx.update(model, cp.restore(str(max_path), item=nnx.state(model), restore_args=ra))

print('Inject LoRA...', flush=True)
model = inject_lora(model, rank=16, alpha=32.0, rngs=nnx.Rngs(42))

# Optimizer (only train fp32 LoRA params)
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
    tokens, masks = batch
    targets = jnp.concatenate([tokens[:, 1:], jnp.zeros((tokens.shape[0], 1), dtype=tokens.dtype)], axis=1)
    loss_mask = jnp.concatenate([masks[:, 1:], jnp.zeros((masks.shape[0], 1), dtype=masks.dtype)], axis=1).astype(jnp.float32)
    logits = model(tokens, deterministic=True)  # NO dropout for stability
    logits_f32 = jnp.clip(logits.astype(jnp.float32), -25.0, 25.0)
    per_token = optax.softmax_cross_entropy_with_integer_labels(logits_f32, targets)
    return (per_token * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0), logits

@nnx.jit
def train_step(model, opt, batch):
    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    gnorm = optax.global_norm(grads)
    opt.update(model, grads)
    return loss, gnorm

SFT_V2_DIR.mkdir(exist_ok=True)
save_cp = orbax.checkpoint.PyTreeCheckpointer()
def save_checkpoint(step):
    path = SFT_V2_DIR / f'sft_v2_step_{step:06d}.orbax'
    save_cp.save(path, nnx.state(model), force=True)
    print(f'  Saved: {path}', flush=True)

print(f'Phase 2: SFT v2 on {len(filtered):,} long convs, {MAX_STEPS} steps', flush=True)

pad_token = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
recent_losses = []
step = 0

def batch_generator():
    bt, bm = [], []
    for conv in filtered * 100:  # cycle if needed
        t, m = encode_with_mask(conv, tokenizer, config['maxlen'])
        t, m = pad_to_maxlen(t, m, config['maxlen'], pad_token)
        bt.append(t)
        bm.append(m)
        if len(bt) == BATCH_SIZE:
            yield jnp.array(bt, dtype=jnp.int32), jnp.array(bm, dtype=jnp.int32)
            bt, bm = [], []

for tokens, masks in batch_generator():
    if step >= MAX_STEPS:
        break
    with mesh:
        shard = NamedSharding(mesh, P('data'))
        ts = jax.device_put(tokens, shard)
        ms = jax.device_put(masks, shard)
        loss, gnorm = train_step(model, optimizer, (ts, ms))

    loss_val = float(loss)
    if loss_val == loss_val and 0.0 <= loss_val <= 100.0:
        recent_losses.append(loss_val)

    if (step + 1) % LOG_EVERY == 0:
        avg = sum(recent_losses) / max(len(recent_losses), 1)
        lr = lr_schedule(step)
        print(f'Step {step+1}/{MAX_STEPS} | Loss: {avg:.4f} | LR: {lr:.2e}', flush=True)
        recent_losses = []

    if (step + 1) % SAVE_EVERY == 0:
        save_checkpoint(step + 1)

    step += 1

save_checkpoint(step)
DONE_MARKER.write_text('completed')
print('Phase 2 done', flush=True)
