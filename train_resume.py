"""
Minimal resume training script for Koda-v0.1.
No orchestrator, no markers, no fancy features.
Just: load checkpoint, train, save checkpoints, respect work hours.
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
import orbax.checkpoint
import time
from pathlib import Path
from datetime import datetime
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from model import MiniGPT, CONFIGS
from data import StreamingDataLoader, get_tokenizer

# ── Config ─────────────────────────────────────────────────
CHECKPOINT_TO_LOAD = "/opt/yoann-test/checkpoints/step_200000.orbax"
CHECKPOINT_DIR = Path("/opt/yoann-test/checkpoints")
START_STEP = 200000
TARGET_STEP = 300000
BATCH_SIZE = 8
LR = 1e-7
LOG_EVERY = 50
SAVE_EVERY = 5000


def is_work_hours():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    h = now.hour + now.minute / 60.0
    return 9.5 <= h < 17.5


def main():
    print(f"Devices: {jax.devices()}", flush=True)
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=("data",))

    tokenizer = get_tokenizer()
    config = CONFIGS["xl"].copy()
    config["vocab_size"] = tokenizer.n_vocab

    # Data
    print("Loading data stream...", flush=True)
    loader = StreamingDataLoader(maxlen=config["maxlen"], batch_size=BATCH_SIZE)

    # Model — NO gradient checkpointing, NO dropout
    print("Init model...", flush=True)
    with mesh:
        model = MiniGPT(
            config,
            dtype=jnp.bfloat16,
            use_gradient_checkpointing=True,
            rngs=nnx.Rngs(0),
        )

    # Load checkpoint
    print(f"Loading {CHECKPOINT_TO_LOAD}...", flush=True)
    sharding = NamedSharding(mesh, P())
    restore_args = jax.tree_util.tree_map(
        lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding),
        nnx.state(model),
    )
    cp = orbax.checkpoint.PyTreeCheckpointer()
    nnx.update(
        model,
        cp.restore(CHECKPOINT_TO_LOAD, item=nnx.state(model), restore_args=restore_args),
    )
    print("Loaded!", flush=True)

    # Optimizer — with zero_nans to survive bad batches
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optax.adamw(LR, weight_decay=0.1),
        ),
        wrt=nnx.Param,
    )

    # Prep targets
    prep_targets = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    # Loss — simple, clip logits for safety
    def loss_fn(model, batch):
        inputs, targets = batch
        logits = model(inputs, deterministic=True)
        logits_f32 = logits.astype(jnp.float32)  # no clip - let cross_entropy handle it
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_f32, targets
        ).mean()
        return loss

    # Train step — simple, returns loss scalar
    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        optimizer.update(model, grads)
        return loss

    # Checkpointer
    save_cp = orbax.checkpoint.PyTreeCheckpointer()

    def save(step):
        path = CHECKPOINT_DIR / f"step_{step:06d}.orbax"
        save_cp.save(path, nnx.state(model), force=True)
        print(f"  Saved: {path}", flush=True)

    # Training loop
    print(f"\nTraining: step {START_STEP} → {TARGET_STEP}", flush=True)
    print(f"Batch={BATCH_SIZE}, LR={LR}, no remat, deterministic=True", flush=True)
    print(f"Work hours pause: Mon-Fri 9:30-17:30\n", flush=True)

    step = START_STEP
    losses = []

    for batch in loader:
        if step >= TARGET_STEP:
            break

        # Work hours check
        if is_work_hours():
            print(f"\n[{datetime.now().strftime('%H:%M')}] Work hours, pausing...", flush=True)
            save(step)
            while is_work_hours():
                time.sleep(300)
            print(f"[{datetime.now().strftime('%H:%M')}] Resuming", flush=True)

        target_batch = prep_targets(batch)

        with mesh:
            shard = NamedSharding(mesh, P("data"))
            b = jax.device_put(batch, shard)
            t = jax.device_put(target_batch, shard)
            loss = train_step(model, optimizer, (b, t))

        loss_val = float(loss)
        if loss_val == loss_val:  # not NaN
            losses.append(loss_val)

        if (step + 1) % LOG_EVERY == 0:
            if losses:
                avg = sum(losses) / len(losses)
                nan_pct = 100 * (LOG_EVERY - len(losses)) / LOG_EVERY
                tokens = (step + 1) * BATCH_SIZE * config["maxlen"]
                print(
                    f"Step {step+1}/{TARGET_STEP} | "
                    f"Loss: {avg:.4f} | "
                    f"NaN: {nan_pct:.0f}% | "
                    f"Tokens: {tokens/1e9:.2f}B",
                    flush=True,
                )
            else:
                print(f"Step {step+1}/{TARGET_STEP} | Loss: ALL NaN", flush=True)
            losses = []

        if (step + 1) % SAVE_EVERY == 0:
            save(step + 1)

        step += 1

    print("\nTraining complete!", flush=True)
    save(step)
    # Marker for orchestrator
    Path("/opt/yoann-test/markers").mkdir(exist_ok=True)
    Path("/opt/yoann-test/markers/phase1.done").write_text("completed")


if __name__ == "__main__":
    main()
