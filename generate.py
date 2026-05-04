"""Text generation with a trained MiniGPT v2 model."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint
from pathlib import Path

from model import MiniGPT, CONFIGS
from data import get_tokenizer


def load_model(config_name="xl", checkpoint_path=None, dtype=jnp.bfloat16):
    """Load model and restore from checkpoint."""
    tokenizer = get_tokenizer()
    config = CONFIGS[config_name]
    config["vocab_size"] = tokenizer.n_vocab

    model = MiniGPT(config, dtype=dtype, rngs=nnx.Rngs(0))

    if checkpoint_path:
        from jax.sharding import SingleDeviceSharding
        device = jax.devices()[0]
        sharding = SingleDeviceSharding(device)

        restore_args = jax.tree_util.tree_map(
            lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding),
            nnx.state(model),
        )
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored = checkpointer.restore(
            str(checkpoint_path),
            item=nnx.state(model),
            restore_args=restore_args,
        )
        nnx.update(model, restored)
        print(f"Loaded checkpoint: {checkpoint_path}")

    return model, tokenizer, config


def generate(model, tokenizer, prompt, maxlen, max_tokens=200,
             temperature=0.7, top_k=40):
    """Generate text with top-k sampling — variable length input.

    JAX will recompile for each new sequence length, but only once per length.
    With max_tokens=80, that's 80 compilations — slower than KV cache but
    correct and simple. Each compilation is ~5s, total ~7 minutes for 80 tokens.
    """
    tokens = tokenizer.encode(prompt)
    end_token = tokenizer.encode(
        "<|endoftext|>", allowed_special={"<|endoftext|>"}
    )[0]

    print(f"Generating from prompt: {prompt!r}")
    print(f"Each new token triggers a recompile (~5s each)...")

    for i in range(max_tokens):
        # Use only the real tokens, no padding (slice to maxlen if too long)
        current = tokens[-maxlen:]
        x = jnp.array([current], dtype=jnp.int32)

        logits = model(x, deterministic=True)
        next_logits = logits[0, -1, :].astype(jnp.float32) / temperature

        # Top-k filtering
        top_k_logits, top_k_indices = jax.lax.top_k(next_logits, top_k)
        probs = jax.nn.softmax(top_k_logits)

        next_token = jax.random.choice(
            jax.random.PRNGKey(len(tokens) + i),
            a=top_k_indices,
            p=probs,
        )
        token_id = int(next_token)

        if token_id == end_token:
            print(f"  EOS at step {i+1}, stopping")
            break

        tokens.append(token_id)

        # Live decode of just the new token
        new_text = tokenizer.decode([token_id])
        print(f"  [{i+1:3d}] +{new_text!r}", flush=True)

    return tokenizer.decode(tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="xl", choices=["medium", "large", "xl"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The history of")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    model, tokenizer, config = load_model(args.config, args.checkpoint)

    text = generate(
        model, tokenizer, args.prompt,
        maxlen=config["maxlen"],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
