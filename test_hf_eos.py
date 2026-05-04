"""Verify the EOS-fixed HF LLaMA model stops natively (no stop_strings)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import HF_LLAMA_EOS_DIR

MODEL_DIR = str(HF_LLAMA_EOS_DIR)

print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
print("Loaded.", flush=True)

EOS = 50256
END_TAG_TOKENS = tok("<|end|>", add_special_tokens=False).input_ids
print(f"EOS={EOS}, <|end|>={END_TAG_TOKENS}", flush=True)


def chat(question, max_new=120):
    messages = [{"role": "user", "content": question}]
    inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False).to(model.device)
    in_len = inputs.shape[-1]
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=EOS,
            pad_token_id=EOS,
        )
    gen = out[0, in_len:].tolist()
    print(f"\nUSER: {question}", flush=True)
    print(f"GENERATED ({len(gen)} tokens): {tok.decode(gen, skip_special_tokens=False)}", flush=True)
    print(f"last 8 tokens: {gen[-8:]}", flush=True)
    saw_eos = EOS in gen
    eos_pos = gen.index(EOS) if saw_eos else -1
    saw_end_tag = False
    for i in range(len(gen) - len(END_TAG_TOKENS) + 1):
        if gen[i:i + len(END_TAG_TOKENS)] == END_TAG_TOKENS:
            saw_end_tag = True
            break
    print(f"saw_end_tag={saw_end_tag}, saw_eos={saw_eos}, eos_pos={eos_pos}, hit_max={len(gen) >= max_new}", flush=True)
    return saw_end_tag, saw_eos, len(gen) < max_new


print("\n=== TEST: HF transformers without stop_strings ===", flush=True)
results = []
for q in [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is 2 + 2?",
    "Name three planets.",
    "Explain gravity briefly.",
]:
    results.append(chat(q))

end_count = sum(1 for e, _, _ in results if e)
eos_count = sum(1 for _, e, _ in results if e)
stop_count = sum(1 for _, _, s in results if s)
print(f"\n=== SUMMARY ===", flush=True)
print(f"<|end|> emitted in {end_count}/{len(results)} responses", flush=True)
print(f"EOS 50256 emitted in {eos_count}/{len(results)} responses", flush=True)
print(f"Stopped naturally (before max_new_tokens): {stop_count}/{len(results)}", flush=True)
