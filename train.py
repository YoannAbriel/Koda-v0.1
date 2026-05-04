"""
Training script for MiniGPT v2 — 2x L40S GPUs.

Features:
- BFloat16 mixed precision (2x memory savings, faster compute)
- Multi-GPU data parallelism via JAX mesh sharding
- Gradient checkpointing (remat) for ~80% activation memory savings
- Gradient clipping (global norm 1.0)
- Cosine LR schedule with warmup
- Periodic checkpointing
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import orbax.checkpoint
import matplotlib.pyplot as plt
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from model import MiniGPT, CONFIGS
from data import StreamingDataLoader, WikiDataLoader, get_tokenizer


# ── Config ─────────────────────────────────────────────────────

MODEL_SIZE = "xl"             # "medium" (400M), "large" (1B), "xl" (1.5B)
DATA_SOURCE = "slimpajama"    # "slimpajama" or "wikipedia"
DTYPE = jnp.bfloat16          # bfloat16 for mixed precision
BATCH_SIZE = 16               # per step (split across GPUs)
MAX_STEPS = 100_000
PEAK_LR = 3e-4
WEIGHT_DECAY = 0.1
LOG_EVERY = 50
SAVE_EVERY = 5000
CHECKPOINT_DIR = Path("/opt/yoann-test/checkpoints")


def setup_mesh():
    """Create a 1D mesh across all available GPUs for data parallelism.

    With 2 GPUs and batch_size=32:
    - GPU 0 gets samples 0-15
    - GPU 1 gets samples 16-31
    Each GPU computes gradients on its half, then they are averaged.
    """
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=("data",))
    print(f"  Mesh: {len(devices)} devices on axis 'data'")
    return mesh


def shard_batch(batch, mesh):
    """Shard a batch across the data axis of the mesh."""
    sharding = NamedSharding(mesh, P("data"))
    return jax.device_put(batch, sharding)


def replicate_on_mesh(state, mesh):
    """Replicate model state across all devices (each GPU has full copy)."""
    replicated = NamedSharding(mesh, P())  # no partitioning = replicated
    return jax.device_put(state, replicated)


def main():
    # ── Devices & Mesh ─────────────────────────────────────────
    print(f"JAX devices: {jax.devices()}")
    print(f"GPU count: {jax.device_count()}")
    print(f"Training dtype: {DTYPE}")
    mesh = setup_mesh()

    # ── Config ─────────────────────────────────────────────────
    tokenizer = get_tokenizer()
    config = CONFIGS[MODEL_SIZE]
    config["vocab_size"] = tokenizer.n_vocab

    # ── Data ───────────────────────────────────────────────────
    if DATA_SOURCE == "slimpajama":
        print(f"\nStreaming from SlimPajama...")
        dataloader = StreamingDataLoader(
            maxlen=config["maxlen"],
            batch_size=BATCH_SIZE,
        )
    else:
        print("\nLoading Wikipedia...")
        dataloader = WikiDataLoader(
            maxlen=config["maxlen"],
            batch_size=BATCH_SIZE,
            num_articles=50_000,
        )

    # ── Model ──────────────────────────────────────────────────
    print(f"\nInitializing {MODEL_SIZE} model in {DTYPE}...")
    print(f"  Gradient checkpointing: ON")

    with mesh:
        model = MiniGPT(config, dtype=DTYPE, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))

    param_count = sum(
        p.size for p in jax.tree_util.tree_leaves(nnx.state(model))
    )
    param_bytes = sum(
        p.size * p.dtype.itemsize
        for p in jax.tree_util.tree_leaves(nnx.state(model))
    )
    print(f"  Parameters: {param_count:,} ({param_count / 1e9:.2f}B)")
    print(f"  Model size: {param_bytes / 1e9:.2f} GB")
    print(f"  Per-GPU model: ~{param_bytes / 1e9:.2f} GB (replicated)")

    # ── Optimizer ──────────────────────────────────────────────
    warmup_steps = max(1, MAX_STEPS // 20)  # 5% warmup

    print(f"\nTraining config:")
    print(f"  Batch size: {BATCH_SIZE} (split across {jax.device_count()} GPUs)")
    print(f"  Per-GPU batch: {BATCH_SIZE // jax.device_count()}")
    print(f"  Max steps: {MAX_STEPS:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Tokens per step: {BATCH_SIZE * config['maxlen']:,}")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=PEAK_LR,
        warmup_steps=warmup_steps,
        decay_steps=MAX_STEPS,
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

    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    # ── Loss function ──────────────────────────────────────────
    def loss_fn(model, batch):
        inputs, targets = batch
        logits = model(inputs, deterministic=False)
        logits_f32 = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_f32, targets
        ).mean()
        return loss, logits

    # ── Train step (JIT compiled, runs across mesh) ────────────
    @nnx.jit
    def train_step(model, optimizer, metrics, batch):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        metrics.update(loss=loss)
        optimizer.update(model, grads)
        return loss

    # ── Prepare targets ────────────────────────────────────────
    prep_targets = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    # ── Checkpointing ─────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def save_checkpoint(step):
        path = CHECKPOINT_DIR / f"step_{step:06d}.orbax"
        checkpointer.save(path, nnx.state(model), force=True)
        print(f"  Checkpoint saved: {path}")

    # ── Training loop ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training: {MODEL_SIZE} ({param_count/1e9:.2f}B) on {DATA_SOURCE}")
    print(f"GPUs: {jax.device_count()} | dtype: {DTYPE} | remat: ON")
    print(f"{'='*60}\n")

    metrics_history = {"train_loss": []}
    step = 0

    for batch in dataloader:
        target_batch = prep_targets(batch)

        # Shard batch across GPUs
        with mesh:
            batch_sharded = shard_batch(batch, mesh)
            target_sharded = shard_batch(target_batch, mesh)
            loss = train_step(model, optimizer, metrics, (batch_sharded, target_sharded))

        if (step + 1) % LOG_EVERY == 0:
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(float(value))
            metrics.reset()

            current_lr = lr_schedule(step)
            tokens_seen = (step + 1) * BATCH_SIZE * config["maxlen"]
            print(
                f"Step {step+1}/{MAX_STEPS} | "
                f"Loss: {metrics_history['train_loss'][-1]:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Tokens: {tokens_seen / 1e9:.2f}B"
            )

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(step + 1)

        step += 1
        if step >= MAX_STEPS:
            break

    # ── Save final & plot ──────────────────────────────────────
    print("\nTraining complete!")
    save_checkpoint(step)

    plt.figure(figsize=(12, 5))
    plt.plot(metrics_history["train_loss"])
    plt.title(f"Training Loss — {MODEL_SIZE} ({param_count/1e9:.2f}B) on {DATA_SOURCE}")
    plt.xlabel(f"Step (x{LOG_EVERY})")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("training_loss.png", dpi=150)
    print("Plot saved: training_loss.png")

    total_tokens = step * BATCH_SIZE * config["maxlen"]
    print(f"Total tokens seen: {total_tokens / 1e9:.2f}B")


if __name__ == "__main__":
    main()
