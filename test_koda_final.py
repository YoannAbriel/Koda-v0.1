"""Test Koda-v0.1 final model (pretrain + SFT v2 + context extension)."""
import jax, jax.numpy as jnp, flax.nnx as nnx, numpy as np
import orbax.checkpoint
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken
from model import MiniGPT, CONFIGS, precompute_rope_frequencies
from lora import inject_lora

devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=('data',))
config = CONFIGS['xl'].copy()
tokenizer = tiktoken.get_encoding('gpt2')
config['vocab_size'] = tokenizer.n_vocab

# Use extended maxlen
config['maxlen'] = 2048

print('Init model (maxlen=2048)...', flush=True)
with mesh:
    model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=False, rngs=nnx.Rngs(0))

# Apply NTK-aware RoPE
head_dim = config['embed_dim'] // config['num_heads']
scale = 2048 / 1024
new_base = 10000.0 * (scale ** (head_dim / (head_dim - 2)))
new_cos, new_sin = precompute_rope_frequencies(head_dim, 2048, base=new_base, dtype=jnp.float32)

print('Inject LoRA...', flush=True)
model = inject_lora(model, rank=16, alpha=32.0, rngs=nnx.Rngs(42))

print('Load final checkpoint...', flush=True)
sharding = NamedSharding(mesh, P())
ra = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
cp = orbax.checkpoint.PyTreeCheckpointer()
nnx.update(model, cp.restore('/opt/yoann-test/long_context_checkpoints/long_step_001000.orbax', item=nnx.state(model), restore_args=ra))

# Re-apply NTK RoPE after checkpoint load
for block in model.blocks:
    object.__setattr__(block.attention, 'rope_cos', new_cos)
    object.__setattr__(block.attention, 'rope_sin', new_sin)

print('Ready!\n', flush=True)

end_token = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]

def chat(question, max_tokens=150, temperature=0.7):
    prompt = f'<|user|>\n{question}\n<|assistant|>\n'
    tokens = tokenizer.encode(prompt, allowed_special='all')
    print(f'USER: {question}', flush=True)
    print(f'ASSISTANT: ', end='', flush=True)

    generated = []
    for i in range(max_tokens):
        x = jnp.array([tokens], dtype=jnp.int32)
        logits = model(x, deterministic=True)
        next_logits = logits[0, -1, :].astype(jnp.float32) / temperature
        top_k_logits, top_k_indices = jax.lax.top_k(next_logits, 40)
        probs = jax.nn.softmax(top_k_logits)
        next_token = jax.random.choice(jax.random.PRNGKey(len(tokens) + i), a=top_k_indices, p=probs)
        token_id = int(next_token)

        if token_id == end_token:
            break
        generated.append(token_id)
        tokens.append(token_id)

    response = tokenizer.decode(generated)
    print(response, flush=True)
    print('---\n', flush=True)

print('=== TESTING KODA-v0.1 FINAL ===\n', flush=True)
chat('What is the capital of France?')
chat('Who wrote Romeo and Juliet?')
chat('How does photosynthesis work?')
chat('Explain what gravity is in simple terms.')
chat('What are the main ingredients in pizza dough?')
