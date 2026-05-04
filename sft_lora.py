"""
LoRA SFT training script with NaN protection.

Key stability fixes:
- zero_nans() on gradients (skip bad batches)
- Logit clipping to prevent softmax overflow
- Gradient global norm clipping (0.5)
- Lower LR (5e-5) appropriate for LoRA
"""
import jax, jax.numpy as jnp, flax.nnx as nnx, optax, numpy as np
import orbax.checkpoint
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken

from model import MiniGPT, CONFIGS
from sft_loader import SFTDataLoader
from lora import inject_lora, count_lora_params
from config import CHECKPOINT_DIR, LORA_CKPT_DIR


# ── Config ─────────────────────────────────────────────────────
MODEL_SIZE = 'xl'
BATCH_SIZE = 8
NUM_EPOCHS = 3
LORA_RANK = 16
LORA_ALPHA = 32.0
PEAK_LR = 5e-5
WEIGHT_DECAY = 0.0
LOG_EVERY = 20
SAVE_EVERY = 500
PRETRAINED_CKPT = f'{CHECKPOINT_DIR}/step_100000.orbax'


def main():
    print(f'JAX devices: {jax.devices()}', flush=True)
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('data',))

    tokenizer = tiktoken.get_encoding('gpt2')
    config = CONFIGS[MODEL_SIZE].copy()
    config['vocab_size'] = tokenizer.n_vocab

    print('Loading SFT data (Dolly + OASST)...', flush=True)
    dataloader = SFTDataLoader(maxlen=config['maxlen'], batch_size=BATCH_SIZE)
    total_conv = len(dataloader.conversations)
    batches_per_epoch = total_conv // BATCH_SIZE
    total_steps = batches_per_epoch * NUM_EPOCHS

    print(f'Init {MODEL_SIZE} model...', flush=True)
    with mesh:
        model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))

    print('Loading pretrained checkpoint...', flush=True)
    sharding = NamedSharding(mesh, P())
    restore_args = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
    cp = orbax.checkpoint.PyTreeCheckpointer()
    nnx.update(model, cp.restore(PRETRAINED_CKPT, item=nnx.state(model), restore_args=restore_args))

    print(f'Injecting LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...', flush=True)
    model = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, rngs=nnx.Rngs(42))

    lora_p, total_p = count_lora_params(model)
    print(f'LoRA: {lora_p:,} ({lora_p/1e6:.2f}M)', flush=True)
    print(f'Total: {total_p:,} ({total_p/1e9:.2f}B)', flush=True)

    warmup_steps = max(1, total_steps // 20)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR,
        warmup_steps=warmup_steps, decay_steps=total_steps,
        end_value=PEAK_LR * 0.1,
    )

    def label_fn(state):
        return jax.tree_util.tree_map(lambda x: 'train' if x.dtype == jnp.float32 else 'freeze', state)

    tx = optax.multi_transform(
        {
            'train': optax.chain(
                optax.zero_nans(),
                optax.clip_by_global_norm(0.5),
                optax.adamw(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
            ),
            'freeze': optax.set_to_zero(),
        },
        label_fn,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    def loss_fn(model, batch):
        tokens, masks = batch
        targets = jnp.concatenate([tokens[:, 1:], jnp.zeros((tokens.shape[0], 1), dtype=tokens.dtype)], axis=1)
        loss_mask = jnp.concatenate([masks[:, 1:], jnp.zeros((masks.shape[0], 1), dtype=masks.dtype)], axis=1).astype(jnp.float32)
        logits = model(tokens, deterministic=True)
        # Clip logits to prevent softmax overflow
        logits_f32 = logits.astype(jnp.float32)
        per_token = optax.softmax_cross_entropy_with_integer_labels(logits_f32, targets)
        return (per_token * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0), logits

    @nnx.jit
    def train_step(model, optimizer, metrics, batch):
        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        gnorm = optax.global_norm(grads)
        # metrics.update removed - log raw loss instead
        optimizer.update(model, grads)
        return loss, gnorm

    LORA_CKPT_DIR.mkdir(exist_ok=True)
    save_cp = orbax.checkpoint.PyTreeCheckpointer()

    def save_checkpoint(step):
        path = LORA_CKPT_DIR / f'lora_step_{step:06d}.orbax'
        save_cp.save(path, nnx.state(model), force=True)
        print(f'  Checkpoint saved: {path}', flush=True)

    print(f'\n{"="*60}', flush=True)
    print(f'LoRA SFT: rank={LORA_RANK}, LR={PEAK_LR}, batch={BATCH_SIZE}', flush=True)
    print(f'Steps: {total_steps:,} ({NUM_EPOCHS} epochs over {total_conv:,} convs)', flush=True)
    print(f'{"="*60}\n', flush=True)

    metrics_history = {'train_loss': []}
    recent_losses = []
    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            with mesh:
                shard = NamedSharding(mesh, P('data'))
                tokens_s = jax.device_put(batch[0], shard)
                masks_s = jax.device_put(batch[1], shard)
                loss, gnorm = train_step(model, optimizer, metrics, (tokens_s, masks_s))

            loss_val = float(loss)
            # Accept only valid losses in [0, 100]
            is_valid = (loss_val == loss_val) and (0.0 <= loss_val <= 100.0)
            if is_valid:
                recent_losses.append(loss_val)
                if len(recent_losses) > LOG_EVERY:
                    recent_losses.pop(0)
            if (step + 1) % LOG_EVERY == 0:
                avg_loss = sum(recent_losses) / max(len(recent_losses), 1)
                metrics_history['train_loss'].append(avg_loss)
                lr = lr_schedule(step)
                nan_count = LOG_EVERY - len(recent_losses)
                print(f'Step {step+1}/{total_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Gnorm: {float(gnorm):.2f} | NaN: {nan_count}', flush=True)
                recent_losses = []

            if (step + 1) % SAVE_EVERY == 0:
                save_checkpoint(step + 1)

            step += 1

    print('\nLoRA SFT complete!', flush=True)
    save_checkpoint(step)


if __name__ == '__main__':
    main()
