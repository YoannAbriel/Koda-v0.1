"""
MiniGPT v2 — Production-grade Transformer (1.5B+ parameters).

Architecture inspired by LLaMA 3, Mistral, and DeepSeek papers:

1. RoPE (Rotary Position Embeddings) — LLaMA, Mistral, DeepSeek
   Encodes position through rotation of Q/K vectors.
   Better generalization to unseen sequence lengths than learned embeddings.

2. GQA (Grouped Query Attention) — LLaMA 3, Mistral
   Multiple query heads share fewer key/value heads.
   Reduces KV cache memory by ~75% with minimal quality loss.

3. RMSNorm — LLaMA, Mistral, DeepSeek
   Simpler and faster than LayerNorm.

4. SwiGLU — LLaMA, PaLM, DeepSeek
   Gated FFN with Swish activation. Better than GELU/ReLU.

5. Pre-norm — GPT-2, LLaMA, all modern LLMs
   LayerNorm before each sublayer for stable deep training.

6. BFloat16 support — for mixed precision training.

7. No bias in linear layers — LLaMA style.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from functools import partial

# Gradient checkpointing: recompute activations during backward pass
# instead of storing them. Trades ~30% more compute for ~80% less memory.
checkpoint = jax.checkpoint


# ── RoPE (Rotary Position Embeddings) ─────────────────────────

def precompute_rope_frequencies(dim, maxlen, base=10000.0, dtype=jnp.float32):
    """Precompute the sin/cos frequencies for RoPE.

    For each pair of dimensions (2i, 2i+1), we compute:
        theta_i = base^(-2i/dim)
        freqs[pos, i] = pos * theta_i

    Then we return cos(freqs) and sin(freqs).
    """
    # theta_i for each dimension pair
    freqs = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))
    # positions
    t = jnp.arange(maxlen, dtype=dtype)
    # outer product: (maxlen, dim/2)
    freqs = jnp.outer(t, freqs)
    cos = jnp.cos(freqs).astype(dtype)
    sin = jnp.sin(freqs).astype(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to x.

    x shape: (..., seq_len, num_heads, head_dim)
    cos/sin shape: (seq_len, head_dim/2)

    Splits x into pairs and applies 2D rotation:
        x_rot[..., 2i]   = x[..., 2i] * cos - x[..., 2i+1] * sin
        x_rot[..., 2i+1] = x[..., 2i] * sin + x[..., 2i+1] * cos
    """
    seq_len = x.shape[-3]
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    # Reshape for broadcasting: (seq_len, 1, head_dim/2)
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    # Split into even/odd pairs
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    return jnp.stack([out1, out2], axis=-1).reshape(x.shape)


# ── GQA (Grouped Query Attention) ─────────────────────────────

class GroupedQueryAttention(nnx.Module):
    """Grouped Query Attention with RoPE.

    Instead of num_heads separate K/V projections, we use num_kv_heads.
    Each KV head is shared by (num_heads // num_kv_heads) query heads.

    Example: num_heads=16, num_kv_heads=4
    → 4 groups of 4 query heads, each group shares 1 KV head
    → 75% less KV memory than full MHA
    """
    def __init__(self, embed_dim, num_heads, num_kv_heads, maxlen,
                 dropout_rate=0.0, dtype=jnp.float32, *, rngs):
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = embed_dim // num_heads

        # Q projection: full num_heads
        self.q_proj = nnx.Linear(embed_dim, num_heads * self.head_dim,
                                 use_bias=False, param_dtype=dtype, rngs=rngs)
        # K/V projections: reduced num_kv_heads
        self.k_proj = nnx.Linear(embed_dim, num_kv_heads * self.head_dim,
                                 use_bias=False, param_dtype=dtype, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, num_kv_heads * self.head_dim,
                                 use_bias=False, param_dtype=dtype, rngs=rngs)
        # Output projection
        self.o_proj = nnx.Linear(num_heads * self.head_dim, embed_dim,
                                 use_bias=False, param_dtype=dtype, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        # Precompute RoPE frequencies
        self.rope_cos, self.rope_sin = precompute_rope_frequencies(
            self.head_dim, maxlen, dtype=dtype
        )

    def __call__(self, x, mask=None, deterministic=False):
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Repeat K/V heads to match Q heads for GQA
        if self.num_groups > 1:
            k = jnp.repeat(k, self.num_groups, axis=2)
            v = jnp.repeat(v, self.num_groups, axis=2)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(q.dtype)
        attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / scale

        # Apply causal mask
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)

        # Attend to values
        attn_output = jnp.matmul(attn_weights, v)

        # Reshape back: (batch, seq_len, embed_dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch, seq_len, -1)

        return self.o_proj(attn_output)


# ── RMSNorm ───────────────────────────────────────────────────

class RMSNorm(nnx.Module):
    def __init__(self, dim, eps=1e-6, dtype=jnp.float32, *, rngs):
        self.eps = eps
        self.scale = nnx.Param(jnp.ones(dim, dtype=dtype))

    def __call__(self, x):
        # Always compute norm in float32 for stability
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        return (x_f32 / rms * self.scale.value).astype(x.dtype)


# ── SwiGLU FFN ────────────────────────────────────────────────

class SwiGLUFFN(nnx.Module):
    def __init__(self, embed_dim, ff_dim, dtype=jnp.float32, *, rngs):
        hidden = int(2 * ff_dim / 3)
        # Round to nearest multiple of 64 for GPU efficiency
        hidden = ((hidden + 63) // 64) * 64
        self.gate = nnx.Linear(embed_dim, hidden, use_bias=False,
                               param_dtype=dtype, rngs=rngs)
        self.up = nnx.Linear(embed_dim, hidden, use_bias=False,
                             param_dtype=dtype, rngs=rngs)
        self.down = nnx.Linear(hidden, embed_dim, use_bias=False,
                               param_dtype=dtype, rngs=rngs)

    def __call__(self, x):
        return self.down(nnx.swish(self.gate(x)) * self.up(x))


# ── Transformer Block ─────────────────────────────────────────

class TransformerBlock(nnx.Module):
    def __init__(self, config, dtype=jnp.float32, *, rngs):
        embed_dim = config["embed_dim"]

        self.attention = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=config["num_heads"],
            num_kv_heads=config["num_kv_heads"],
            maxlen=config["maxlen"],
            dropout_rate=config["dropout_rate"],
            dtype=dtype,
            rngs=rngs,
        )
        self.norm1 = RMSNorm(embed_dim, dtype=dtype, rngs=rngs)
        self.ffn = SwiGLUFFN(embed_dim, config["ff_dim"], dtype=dtype, rngs=rngs)
        self.norm2 = RMSNorm(embed_dim, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(rate=config["dropout_rate"], rngs=rngs)

    def __call__(self, x, mask=None, deterministic=False):
        # Pre-norm attention
        h = self.norm1(x)
        h = self.attention(h, mask=mask, deterministic=deterministic)
        h = self.dropout(h, deterministic=deterministic)
        x = x + h

        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ffn(h)
        h = self.dropout(h, deterministic=deterministic)
        x = x + h

        return x


def checkpointed_block_fn(block, x, mask):
    """Wrapper for gradient checkpointing a single block.

    During forward pass: runs normally, but does NOT save activations.
    During backward pass: re-runs the forward to recompute activations.
    Result: ~80% less memory for activations, ~30% more compute.

    Note: deterministic=False is hardcoded here because remat only runs
    during training (backward pass). Inference skips remat entirely.
    """
    return block(x, mask=mask, deterministic=True)


# ── Full Model ─────────────────────────────────────────────────

class MiniGPT(nnx.Module):
    def __init__(self, config, dtype=jnp.float32, use_gradient_checkpointing=True, *, rngs):
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token embedding (no position embedding — RoPE handles position)
        self.token_emb = nnx.Embed(
            config["vocab_size"], config["embed_dim"],
            param_dtype=dtype, rngs=rngs,
        )
        self.blocks = nnx.List([
            TransformerBlock(config, dtype=dtype, rngs=rngs)
            for _ in range(config["num_blocks"])
        ])
        self.norm_final = RMSNorm(config["embed_dim"], dtype=dtype, rngs=rngs)
        self.output_layer = nnx.Linear(
            config["embed_dim"], config["vocab_size"],
            use_bias=False, param_dtype=dtype, rngs=rngs,
        )

    def causal_mask(self, seq_len):
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        return mask[None, None, :, :]  # (1, 1, seq, seq) for broadcast over batch & heads

    def __call__(self, token_ids, deterministic=False):
        seq_len = token_ids.shape[1]
        mask = self.causal_mask(seq_len)

        x = self.token_emb(token_ids)

        for block in self.blocks:
            if self.use_gradient_checkpointing and not deterministic:
                # Gradient checkpointing: recompute activations during backward
                x = nnx.remat(checkpointed_block_fn)(block, x, mask)
            else:
                # No checkpointing during inference
                x = block(x, mask=mask, deterministic=deterministic)

        x = self.norm_final(x)
        logits = self.output_layer(x)
        return logits


# ── Model configs ──────────────────────────────────────────────
#
# Inspired by LLaMA 3 scaling:
# - num_kv_heads < num_heads (GQA)
# - ff_dim ≈ 4x embed_dim (SwiGLU uses 2/3 internally)
# - No position embedding (RoPE instead)

CONFIGS = {
    # ~400M params — for testing
    "medium": {
        "maxlen": 1024,
        "vocab_size": 50257,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_kv_heads": 4,    # GQA: 4 groups of 4 query heads
        "ff_dim": 4096,
        "num_blocks": 24,
        "dropout_rate": 0.1,
    },
    # ~1B params
    "large": {
        "maxlen": 2048,
        "vocab_size": 50257,
        "embed_dim": 2048,
        "num_heads": 16,
        "num_kv_heads": 4,    # GQA
        "ff_dim": 5632,
        "num_blocks": 22,
        "dropout_rate": 0.1,
    },
    # ~1.3B params — target config for 2x L40S
    "xl": {
        "maxlen": 1024,
        "vocab_size": 50257,
        "embed_dim": 2048,
        "num_heads": 32,
        "num_kv_heads": 8,    # GQA: 4 groups of 4 query heads
        "ff_dim": 8192,
        "num_blocks": 24,
        "dropout_rate": 0.1,
    },
}


# ── Class alias ──────────────────────────────────────────────
# KodaLite is the official name of our architecture.
# Style: SQLite, TFLite — small, efficient, well-crafted.
# The Koda-v0.1 model uses this class.
KodaLite = MiniGPT
