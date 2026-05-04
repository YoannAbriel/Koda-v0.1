"""Verify the EOS-finetuned model emits token 50256 (<|endoftext|>) after <|end|>."""
import jax, jax.numpy as jnp, flax.nnx as nnx, numpy as np
import orbax.checkpoint
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import tiktoken
from model import MiniGPT, CONFIGS
from lora import inject_lora
from config import CHECKPOINT_DIR, LORA_EOS_DIR

LORA_EOS_CKPT = f'{LORA_EOS_DIR}/lora_eos_step_000200.orbax'
PRETRAINED_CKPT = f'{CHECKPOINT_DIR}/step_100000.orbax'

devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=('data',))
config = CONFIGS['xl'].copy()
tokenizer = tiktoken.get_encoding('gpt2')
config['vocab_size'] = tokenizer.n_vocab

print('Init xl model...', flush=True)
with mesh:
    model = MiniGPT(config, dtype=jnp.bfloat16, use_gradient_checkpointing=False, rngs=nnx.Rngs(0))

sharding = NamedSharding(mesh, P())
cp = orbax.checkpoint.PyTreeCheckpointer()

print(f'Loading pretrained {PRETRAINED_CKPT}...', flush=True)
ra = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
nnx.update(model, cp.restore(PRETRAINED_CKPT, item=nnx.state(model), restore_args=ra))

print('Inject LoRA...', flush=True)
model = inject_lora(model, rank=16, alpha=32.0, rngs=nnx.Rngs(42))

print(f'Loading EOS LoRA {LORA_EOS_CKPT}...', flush=True)
ra2 = jax.tree_util.tree_map(lambda _: orbax.checkpoint.ArrayRestoreArgs(sharding=sharding), nnx.state(model))
nnx.update(model, cp.restore(LORA_EOS_CKPT, item=nnx.state(model), restore_args=ra2))

print('Ready!\n', flush=True)

EOS_TOKEN = 50256
END_TAG_TOKENS = tokenizer.encode('<|end|>', allowed_special='all')
print(f'EOS token: {EOS_TOKEN}', flush=True)
print(f'<|end|> tokens: {END_TAG_TOKENS}', flush=True)

def chat_greedy(question, max_tokens=200):
    prompt = f'<|user|>\n{question}\n<|assistant|>\n'
    tokens = tokenizer.encode(prompt, allowed_special='all')
    print(f'USER: {question}', flush=True)
    print(f'ASSISTANT: ', end='', flush=True)

    generated = []
    saw_end_tag = False
    saw_eos = False
    eos_position = -1
    for i in range(max_tokens):
        x = jnp.array([tokens], dtype=jnp.int32)
        logits = model(x, deterministic=True)
        next_logits = logits[0, -1, :].astype(jnp.float32)
        next_token = int(jnp.argmax(next_logits))
        token_id = next_token

        generated.append(token_id)
        tokens.append(token_id)

        # Check for <|end|> sequence ending
        if len(generated) >= len(END_TAG_TOKENS) and generated[-len(END_TAG_TOKENS):] == END_TAG_TOKENS:
            saw_end_tag = True
            print(f'\n[Got <|end|> at position {len(generated)}]', flush=True)

        if token_id == EOS_TOKEN:
            saw_eos = True
            eos_position = len(generated)
            print(f'\n[EOS 50256 emitted at position {len(generated)}]', flush=True)
            break

    response = tokenizer.decode(generated)
    print(response, flush=True)
    print(f'last 8 generated tokens: {generated[-8:]}', flush=True)
    print(f'saw_end_tag={saw_end_tag}, saw_eos={saw_eos}, eos_pos={eos_position}', flush=True)
    print('---\n', flush=True)
    return saw_end_tag, saw_eos

print('=== TEST: does model emit 50256 after <|end|>? ===\n', flush=True)
results = []
for q in [
    'What is the capital of France?',
    'Who wrote Romeo and Juliet?',
    'Explain gravity in one sentence.',
    'What is 2 + 2?',
    'Name three planets.',
]:
    results.append(chat_greedy(q))

end_count = sum(1 for e, _ in results if e)
eos_count = sum(1 for _, e in results if e)
print(f'\n=== SUMMARY ===', flush=True)
print(f'<|end|> emitted in {end_count}/{len(results)} responses', flush=True)
print(f'EOS 50256 emitted in {eos_count}/{len(results)} responses', flush=True)
