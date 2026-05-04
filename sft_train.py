"""
SFT training script.

Loads the pre-trained checkpoint and fine-tunes on a mix of
Dolly 15K + OpenAssistant conversations.

Key differences from pre-training:
- Lower learning rate (5e-5 vs 3e-4)
- Fewer epochs (3-5 vs unlimited)
- MASKED loss (only on assistant tokens)
- Loads from checkpoint instead of random init
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import orbax.checkpoint
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from model import MiniGPT, CONFIGS
from sft_loader import SFTDataLoader


# ── Config ─────────────────────────────────────────────────────

MODEL_SIZE = 'xl'
DTYPE = jnp.bfloat16
BATCH_SIZE = 16
NUM_EPOCHS = 3
PEAK_LR = 1e-5  # Much lower than pre-training (3e-4)
WEIGHT_DECAY = 0.0  # Often disabled for SFT
LOG_EVERY = 20
SAVE_EVERY = 500

PRETRAINED_CKPT = '/opt/yoann-test/checkpoints/step_100000.orbax'
SFT_CKPT_DIR = Path('/opt/yoann-test/sft_checkpoints')


def main():
    print(f'JAX devices: {jax.devices()}')
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('data',))

    # ── Data ───────────────────────────────────────────────────
    print('\nLoading SFT data (Dolly + OASST)...')
    config = CONFIGS[MODEL_SIZE].copy()
    import tiktoken
    config['vocab_size'] = tiktoken.get_encoding('gpt2').n_vocab

    dataloader = SFTDataLoader(
        maxlen=config['maxlen'],
        batch_size=BATCH_SIZE,
    )
    total_conversations = len(dataloader.conversations)
    batches_per_epoch = total_conversations // BATCH_SIZE
    total_steps = batches_per_epoch * NUM_EPOCHS

    print(f'\nTraining config:')
    print(f'  Conversations: {total_conversations:,}')
    print(f'  Batches per epoch: {batches_per_epoch:,}')
    print(f'  Total steps: {total_steps:,}')
    print(f'  Epochs: {NUM_EPOCHS}')

    # ── Model ──────────────────────────────────────────────────
    print(f'\nInitializing {MODEL_SIZE} model...')
    with mesh:
        model = MiniGPT(config, dtype=DTYPE, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))

    pc = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model)))
    print(f'Parameters: {pc/1e9:.2f}B')

    # ── Load pre-trained checkpoint ────────────────────────────
    print(f'\nLoading pre-trained checkpoint from {PRETRAINED_CKPT}...')
    sharding = NamedSharding(mesh, P())
    restore_args = jax.tree_util.tree_map(
        lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding),
        nnx.state(model),
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(
        PRETRAINED_CKPT,
        item=nnx.state(model),
        restore_args=restore_args,
    )
    nnx.update(model, restored)
    print('Pre-trained weights loaded!')

    # ── Optimizer ──────────────────────────────────────────────
    warmup_steps = max(1, total_steps // 20)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=PEAK_LR,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=PEAK_LR * 0.1,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
        ),
        wrt=nnx.Param,
    )

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    # ── Masked loss function ──────────────────────────────────
    def loss_fn(model, batch):
        tokens, masks = batch
        # Inputs are tokens[:-1], targets are tokens[1:]
        inputs = tokens
        targets = jnp.concatenate([tokens[:, 1:], jnp.zeros((tokens.shape[0], 1), dtype=tokens.dtype)], axis=1)
        # Loss mask is shifted: position i should compute loss if mask[i+1] is set
        loss_mask = jnp.concatenate([masks[:, 1:], jnp.zeros((masks.shape[0], 1), dtype=masks.dtype)], axis=1)

        logits = model(inputs, deterministic=False)
        logits_f32 = logits.astype(jnp.float32)
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, targets)

        # Apply mask: only compute loss on assistant tokens
        masked_loss = per_token_loss * loss_mask
        loss = masked_loss.sum() / jnp.maximum(loss_mask.sum(), 1.0)
        return loss, logits

    # ── Train step ─────────────────────────────────────────────
    @nnx.jit
    def train_step(model, optimizer, metrics, batch):
        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        metrics.update(loss=loss)
        optimizer.update(model, grads)
        return loss

    # ── Checkpointing ─────────────────────────────────────────
    SFT_CKPT_DIR.mkdir(exist_ok=True)
    save_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def save_checkpoint(step):
        path = SFT_CKPT_DIR / f'sft_step_{step:06d}.orbax'
        save_checkpointer.save(path, nnx.state(model), force=True)
        print(f'  Checkpoint saved: {path}')

    # ── Training loop ──────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'SFT Training: {MODEL_SIZE} ({pc/1e9:.2f}B)')
    print(f'  Pre-trained from: step_100000')
    print(f'  Datasets: Dolly + OASST ({total_conversations:,})')
    print(f'  LR: {PEAK_LR} | Epochs: {NUM_EPOCHS}')
    print(f'{"="*60}\n')

    metrics_history = {'train_loss': []}
    step = 0

    for epoch in range(NUM_EPOCHS):
        print(f'\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---')
        for batch in dataloader:
            with mesh:
                shard = NamedSharding(mesh, P('data'))
                tokens_s = jax.device_put(batch[0], shard)
                masks_s = jax.device_put(batch[1], shard)
                loss = train_step(model, optimizer, metrics, (tokens_s, masks_s))

            if (step + 1) % LOG_EVERY == 0:
                for metric, value in metrics.compute().items():
                    metrics_history[f'train_{metric}'].append(float(value))
                metrics.reset()
                current_lr = lr_schedule(step)
                print(f'Epoch {epoch+1} | Step {step+1}/{total_steps} | Loss: {metrics_history["train_loss"][-1]:.4f} | LR: {current_lr:.2e}')

            if (step + 1) % SAVE_EVERY == 0:
                save_checkpoint(step + 1)

            step += 1

    print('\nSFT complete!')
    save_checkpoint(step)


if __name__ == '__main__':
    main()
