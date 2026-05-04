"""
LoRA (Low-Rank Adaptation) implementation for the MiniGPT model.

LoRA freezes the original weights and adds small trainable adapters:
    h = W·x + (B·A)·x · (alpha/r)

Where:
    - W is the original weight matrix (frozen, bfloat16)
    - A is a (r, in_features) matrix initialized randomly (trainable, fp32)
    - B is a (out_features, r) matrix initialized to zero (trainable, fp32)
    - r is the rank (typically 8-64)
    - alpha is a scaling factor (typically 2*r)

Memory cost for r=16:
    Instead of training 1.27B params, we train ~10M params
    (only the LoRA matrices in attention layers).
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx


class LoRALinear(nnx.Module):
    """A Linear layer wrapped with a LoRA adapter.

    Forward: y = x @ W^T + (x @ A^T @ B^T) * (alpha/r)
    Where W is frozen and A, B are trainable.
    """
    def __init__(self, base_linear: nnx.Linear, rank: int = 16, alpha: float = 32.0, *, rngs):
        # Store the base linear (frozen)
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.kernel.value.shape[0]
        out_features = base_linear.kernel.value.shape[1]

        # LoRA A: (in_features, rank) - initialized with small random values
        # We use a Kaiming-like init to keep variance reasonable
        key = rngs.params()
        a_init = jax.random.normal(key, (in_features, rank), dtype=jnp.float32) * (1.0 / jnp.sqrt(in_features))
        self.lora_a = nnx.Param(a_init)

        # LoRA B: (rank, out_features) - initialized to ZERO
        # This makes the LoRA contribution zero at initialization,
        # so the model behaves identically to the base model at start.
        self.lora_b = nnx.Param(jnp.zeros((rank, out_features), dtype=jnp.float32))

    def __call__(self, x):
        # Base path (frozen weights)
        base_out = self.base(x)

        # LoRA path (in float32 for stability)
        x_f32 = x.astype(jnp.float32)
        lora_out = x_f32 @ self.lora_a.value @ self.lora_b.value
        lora_out = lora_out * self.scaling

        # Cast back to base output dtype
        return base_out + lora_out.astype(base_out.dtype)


def freeze_base_params(model):
    """Mark all non-LoRA parameters as frozen by changing their type.

    In NNX, we use a custom variable type to distinguish trainable vs frozen.
    Actually we'll filter at optimizer level using nnx.filterlib.
    """
    pass  # Filtering happens in the optimizer setup


def inject_lora(model, rank=16, alpha=32.0, target_modules=('q_proj', 'k_proj', 'v_proj', 'o_proj'), *, rngs):
    """Replace target Linear layers in the model with LoRALinear wrappers.

    Walks through transformer blocks and wraps the specified projections.
    """
    n_wrapped = 0
    for block in model.blocks:
        attn = block.attention
        for name in target_modules:
            if hasattr(attn, name):
                base = getattr(attn, name)
                if isinstance(base, nnx.Linear):
                    wrapped = LoRALinear(base, rank=rank, alpha=alpha, rngs=rngs)
                    setattr(attn, name, wrapped)
                    n_wrapped += 1
    print(f'Wrapped {n_wrapped} layers with LoRA (rank={rank}, alpha={alpha})')
    return model


def count_lora_params(model):
    """Count the number of trainable LoRA parameters."""
    state = nnx.state(model)
    total = 0
    lora_count = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        path_str = jax.tree_util.keystr(path)
        if 'lora_a' in path_str or 'lora_b' in path_str:
            lora_count += leaf.size
        total += leaf.size
    return lora_count, total
