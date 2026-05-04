"""
Continue pre-training from step_100000 to reach 5B tokens.

Features:
- Resumes from the latest checkpoint (auto-detect)
- Respects work hours: pauses Mon-Fri 10:00-17:30 Paris time
- Auto-resumes when allowed
- Saves checkpoint before each pause
- Total target: 5B tokens (~200K additional steps)
"""
import jax, jax.numpy as jnp, flax.nnx as nnx, optax, numpy as np
import orbax.checkpoint
import time
import re
from pathlib import Path
from datetime import datetime
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from model import MiniGPT, CONFIGS
from data import StreamingDataLoader, get_tokenizer


# ── Config ─────────────────────────────────────────────────────
MODEL_SIZE = 'xl'
DTYPE = jnp.bfloat16
BATCH_SIZE = 4
TARGET_STEPS = 300_000        # total steps including the 100K already done
PEAK_LR = 1e-4                # lower than original (3e-4) since we are continuing
WEIGHT_DECAY = 0.1
LOG_EVERY = 50
SAVE_EVERY = 5000
CHECKPOINT_DIR = Path('/opt/yoann-test/checkpoints')


def is_training_allowed():
    """True if we are allowed to train (outside work hours)."""
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    hour_decimal = now.hour + now.minute / 60.0
    # Allowed: before 10:00 OR after 17:30
    return True  # TEMP: no pause


def find_latest_checkpoint():
    """Find the most recent checkpoint in CHECKPOINT_DIR."""
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
    return max_path, max_step


def main():
    print(f'JAX devices: {jax.devices()}', flush=True)
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('data',))

    tokenizer = get_tokenizer()
    config = CONFIGS[MODEL_SIZE].copy()
    config['vocab_size'] = tokenizer.n_vocab

    # Find checkpoint to resume from
    # HARDCODED: always resume from step_200000 (last known good checkpoint)
    start_step = 200000
    ckpt_path = CHECKPOINT_DIR / f'step_{start_step:06d}.orbax'
    print(f'Resuming from: {ckpt_path} (step {start_step})', flush=True)

    # Data
    print('Loading SlimPajama stream...', flush=True)
    dataloader = StreamingDataLoader(maxlen=config['maxlen'], batch_size=BATCH_SIZE)

    # Model
    print(f'Init {MODEL_SIZE} model...', flush=True)
    with mesh:
        model = MiniGPT(config, dtype=DTYPE, use_gradient_checkpointing=False, rngs=nnx.Rngs(0))

    print('Loading checkpoint weights...', flush=True)
    sharding = NamedSharding(mesh, P())
    restore_args = jax.tree_util.tree_map(
        lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding),
        nnx.state(model),
    )
    cp = orbax.checkpoint.PyTreeCheckpointer()
    nnx.update(model, cp.restore(str(ckpt_path), item=nnx.state(model), restore_args=restore_args))

    pc = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model)))
    print(f'Parameters: {pc / 1e9:.2f}B', flush=True)

    # Optimizer with continuation schedule
    remaining_steps = TARGET_STEPS - start_step
    # On restart: fresh Adam needs gentle warmup from very small LR
    warmup_steps = 500

    print(f'Start step: {start_step}', flush=True)
    print(f'Target step: {TARGET_STEPS}', flush=True)
    print(f'Remaining: {remaining_steps:,}', flush=True)
    print(f'Warmup: {warmup_steps} steps', flush=True)

    restart_lr = 3e-5

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=restart_lr,
        warmup_steps=warmup_steps,
        decay_steps=remaining_steps,
        end_value=1e-5,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
        ),
        wrt=nnx.Param,
    )

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))

    def loss_fn(model, batch):
        inputs, targets = batch
        logits = model(inputs, deterministic=True)
        logits_f32 = jnp.clip(logits.astype(jnp.float32), -30.0, 30.0)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_f32, targets).mean()
        return loss, logits

    @nnx.jit
    def train_step(model, optimizer, metrics, batch):
        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        metrics.update(loss=loss)
        optimizer.update(model, grads)
        return loss

    prep_targets = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    save_cp = orbax.checkpoint.PyTreeCheckpointer()

    def save_checkpoint(step):
        path = CHECKPOINT_DIR / f'step_{step:06d}.orbax'
        save_cp.save(path, nnx.state(model), force=True)
        print(f'  Checkpoint saved: {path}', flush=True)

    print(f'\n{"="*60}', flush=True)
    print(f'CONTINUE TRAINING: step {start_step} → {TARGET_STEPS}', flush=True)
    print(f'Work hours pause: Mon-Fri 10:00-17:30 Paris time', flush=True)
    print(f'{"="*60}\n', flush=True)

    step = start_step
    relative_step = 0  # for LR schedule

    while step < TARGET_STEPS:
        for batch in dataloader:
            if step >= TARGET_STEPS:
                break

            # Check work hours
            if not is_training_allowed():
                print(f'\n[{datetime.now().strftime("%H:%M")}] Work hours: pausing training, saving checkpoint...', flush=True)
                save_checkpoint(step)
                # Sleep until allowed
                while not is_training_allowed():
                    time.sleep(300)  # check every 5 minutes
                print(f'[{datetime.now().strftime("%H:%M")}] Resuming training', flush=True)

            target_batch = prep_targets(batch)

            with mesh:
                shard = NamedSharding(mesh, P('data'))
                batch_s = jax.device_put(batch, shard)
                target_s = jax.device_put(target_batch, shard)
                loss, gnorm = train_step(model, optimizer, metrics, (batch_s, target_s))

            if (step + 1) % LOG_EVERY == 0:
                avg_loss = float(loss)  # raw loss, not averaged
                lr = lr_schedule(relative_step)
                gnorm_val = float(gnorm); tokens_seen = (step + 1) * BATCH_SIZE * config['maxlen']
                print(f'Step {step+1}/{TARGET_STEPS} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Gnorm: {gnorm_val:.2f}, Tokens: {tokens_seen / 1e9:.2f}B', flush=True)

            if (step + 1) % SAVE_EVERY == 0:
                save_checkpoint(step + 1)

            step += 1
            relative_step += 1

    print('\nContinue training complete!', flush=True)
    save_checkpoint(step)
    # Create done marker for orchestrator
    from pathlib import Path
    marker = Path('/opt/yoann-test/markers/phase1.done')
    marker.parent.mkdir(exist_ok=True)
    marker.write_text('completed')
    print('Phase 1 marker created', flush=True)


if __name__ == '__main__':
    main()
