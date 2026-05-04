"""
Resume LoRA SFT from lora_step_014283 with EOS-appended data.
Goal: teach the model to emit token 50256 (<|endoftext|>) after <|end|>,
so GGUF/llama.cpp/LM Studio stop natively.
"""
import jax, jax.numpy as jnp, flax.nnx as nnx, optax, numpy as np
import orbax.checkpoint
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken

from model import MiniGPT, CONFIGS
from sft_loader_eos import SFTDataLoaderEOS
from lora import inject_lora, count_lora_params
from config import CHECKPOINT_DIR, LORA_CKPT_DIR, LORA_EOS_DIR

MODEL_SIZE = 'xl'
BATCH_SIZE = 8
NUM_STEPS = 500
LORA_RANK = 16
LORA_ALPHA = 32.0
PEAK_LR = 3e-5  # lower than initial SFT, this is a quick adaptation
WEIGHT_DECAY = 0.0
LOG_EVERY = 20
SAVE_EVERY = 100

PRETRAINED_CKPT = f'{CHECKPOINT_DIR}/step_100000.orbax'
RESUME_LORA_CKPT = f'{LORA_CKPT_DIR}/lora_step_014283.orbax'
EOS_LORA_DIR = Path(str(LORA_EOS_DIR))


def main():
    print(f'JAX devices: {jax.devices()}', flush=True)
    devices = jax.devices()
    mesh = Mesh(np.array(devices), axis_names=('data',))

    tokenizer = tiktoken.get_encoding('gpt2')
    config = CONFIGS[MODEL_SIZE].copy()
    config['vocab_size'] = tokenizer.n_vocab

    print('Loading SFT data with EOS appended...', flush=True)
    dataloader = SFTDataLoaderEOS(maxlen=config['maxlen'], batch_size=BATCH_SIZE)

    print(f'Init {MODEL_SIZE} model...', flush=True)
    with mesh:
        model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=True, rngs=nnx.Rngs(0))

    sharding = NamedSharding(mesh, P())
    cp = orbax.checkpoint.PyTreeCheckpointer()

    print('Loading pretrained step_100000...', flush=True)
    restore_args = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
    nnx.update(model, cp.restore(PRETRAINED_CKPT, item=nnx.state(model), restore_args=restore_args))

    print(f'Injecting LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...', flush=True)
    model = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, rngs=nnx.Rngs(42))

    print(f'Loading SFT v1 LoRA from {RESUME_LORA_CKPT}...', flush=True)
    restore_args2 = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
    nnx.update(model, cp.restore(RESUME_LORA_CKPT, item=nnx.state(model), restore_args=restore_args2))

    lora_p, total_p = count_lora_params(model)
    print(f'LoRA: {lora_p:,} ({lora_p/1e6:.2f}M), Total: {total_p:,} ({total_p/1e9:.2f}B)', flush=True)

    warmup_steps = max(1, NUM_STEPS // 10)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=PEAK_LR,
        warmup_steps=warmup_steps, decay_steps=NUM_STEPS,
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

    def loss_fn(model, batch):
        tokens, masks = batch
        targets = jnp.concatenate([tokens[:, 1:], jnp.zeros((tokens.shape[0], 1), dtype=tokens.dtype)], axis=1)
        loss_mask = jnp.concatenate([masks[:, 1:], jnp.zeros((masks.shape[0], 1), dtype=masks.dtype)], axis=1).astype(jnp.float32)
        logits = model(tokens, deterministic=True)
        logits_f32 = logits.astype(jnp.float32)
        per_token = optax.softmax_cross_entropy_with_integer_labels(logits_f32, targets)
        return (per_token * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0), logits

    @nnx.jit
    def train_step(model, optimizer, batch):
        (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        gnorm = optax.global_norm(grads)
        optimizer.update(model, grads)
        return loss, gnorm

    EOS_LORA_DIR.mkdir(exist_ok=True)
    save_cp = orbax.checkpoint.PyTreeCheckpointer()

    def save_checkpoint(step):
        path = EOS_LORA_DIR / f'lora_eos_step_{step:06d}.orbax'
        save_cp.save(path, nnx.state(model), force=True)
        print(f'  Saved: {path}', flush=True)

    print(f'\n{"="*60}', flush=True)
    print(f'EOS fine-tune: {NUM_STEPS} steps from lora_step_014283', flush=True)
    print(f'{"="*60}\n', flush=True)

    recent_losses = []
    step = 0
    for batch in dataloader:
        if step >= NUM_STEPS:
            break
        with mesh:
            shard = NamedSharding(mesh, P('data'))
            tokens_s = jax.device_put(batch[0], shard)
            masks_s = jax.device_put(batch[1], shard)
            loss, gnorm = train_step(model, optimizer, (tokens_s, masks_s))

        loss_val = float(loss)
        is_valid = (loss_val == loss_val) and (0.0 <= loss_val <= 100.0)
        if is_valid:
            recent_losses.append(loss_val)
            if len(recent_losses) > LOG_EVERY:
                recent_losses.pop(0)

        if (step + 1) % LOG_EVERY == 0:
            avg_loss = sum(recent_losses) / max(len(recent_losses), 1)
            lr = lr_schedule(step)
            nan_count = LOG_EVERY - len(recent_losses)
            print(f'Step {step+1}/{NUM_STEPS} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Gnorm: {float(gnorm):.2f} | NaN: {nan_count}', flush=True)
            recent_losses = []

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(step + 1)
        step += 1

    save_checkpoint(step)
    print('\nEOS fine-tune complete!', flush=True)


if __name__ == '__main__':
    main()
