"""
Minimal benchmark script: zero-shot multiple-choice accuracy + perplexity
for a set of models on standard tasks (HellaSwag, ARC, WinoGrande, PIQA,
BoolQ, OpenBookQA, LAMBADA).

Each task reduces to: given context + several candidate continuations,
pick the one with highest log-likelihood per token (or total log-likelihood
depending on convention). Gold answer = argmax.

Usage:
    python my_bench.py <model_id> <output.json> [limit]
"""
import sys
import json
import time
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def loglikelihood(model, tokenizer, context, continuation):
    """Return sum log-prob of continuation tokens given context (i.e. P(cont|ctx)).
    Returns (log_prob, num_cont_tokens).
    """
    # Tokenize separately
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    cont_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # Full input = ctx + cont
    full_ids = torch.cat([ctx_ids, cont_ids], dim=1)

    # Cap at model max length
    max_len = getattr(model.config, "max_position_embeddings", 2048)
    if full_ids.shape[1] > max_len:
        full_ids = full_ids[:, -max_len:]
        # Recompute how many tokens are "continuation" portion
        cont_len = min(cont_ids.shape[1], full_ids.shape[1])
    else:
        cont_len = cont_ids.shape[1]

    # Forward pass
    logits = model(full_ids).logits  # [1, T, V]
    # Shift: predict position t from position t-1
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits.float(), dim=-1)
    gathered = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    # Sum over continuation tokens (the last cont_len of the shifted sequence)
    cont_logprob = gathered[:, -cont_len:].sum().item()
    return cont_logprob, cont_len


def mc_eval(model, tokenizer, items, name, normalize_by_length=True, limit=None):
    """items: list of dicts {ctx, choices: [...], gold: int}. Return accuracy."""
    if limit:
        items = items[:limit]
    correct = 0
    t0 = time.time()
    for i, it in enumerate(items):
        scores = []
        for c in it["choices"]:
            lp, n = loglikelihood(model, tokenizer, it["ctx"], c)
            if normalize_by_length and n > 0:
                scores.append(lp / n)
            else:
                scores.append(lp)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        if pred == it["gold"]:
            correct += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {name}: {i+1}/{len(items)} | acc={correct/(i+1):.4f} | {rate:.1f}/s", flush=True)
    acc = correct / len(items) if items else 0.0
    return {"accuracy": acc, "n": len(items)}


# ----- Dataset loaders -----

def load_hellaswag(limit=None):
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        ctx = ex["ctx"] + " "
        items.append({"ctx": ctx, "choices": ex["endings"], "gold": int(ex["label"])})
    return items[:limit] if limit else items


def load_arc(subset, limit=None):
    # subset: "ARC-Easy" or "ARC-Challenge"
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    items = []
    for ex in ds:
        ctx = f"Question: {ex['question']}\nAnswer:"
        choices = [f" {t}" for t in ex["choices"]["text"]]
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"])
        items.append({"ctx": ctx, "choices": choices, "gold": gold})
    return items[:limit] if limit else items


def load_winogrande(limit=None):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        sentence = ex["sentence"]
        idx = sentence.index("_")
        ctx = sentence[:idx]
        tail = sentence[idx+1:]
        choices = [ex["option1"] + tail, ex["option2"] + tail]
        gold = int(ex["answer"]) - 1
        items.append({"ctx": ctx, "choices": choices, "gold": gold})
    return items[:limit] if limit else items


def load_piqa(limit=None):
    ds = load_dataset("baber/piqa", split="validation")
    items = []
    for ex in ds:
        ctx = f"Question: {ex['goal']}\nAnswer:"
        choices = [f" {ex['sol1']}", f" {ex['sol2']}"]
        items.append({"ctx": ctx, "choices": choices, "gold": int(ex["label"])})
    return items[:limit] if limit else items


def load_boolq(limit=None):
    ds = load_dataset("aps/super_glue", "boolq", split="validation", trust_remote_code=True)
    items = []
    for ex in ds:
        ctx = f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:"
        choices = [" no", " yes"]
        items.append({"ctx": ctx, "choices": choices, "gold": int(ex["label"])})
    return items[:limit] if limit else items


def load_openbookqa(limit=None):
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    items = []
    for ex in ds:
        ctx = ex["question_stem"]
        choices = [f" {t}" for t in ex["choices"]["text"]]
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"])
        items.append({"ctx": ctx, "choices": choices, "gold": gold})
    return items[:limit] if limit else items


@torch.no_grad()
def lambada_eval(model, tokenizer, limit=None):
    """LAMBADA: last-word prediction. Accuracy = model's argmax for last token matches gold."""
    ds = load_dataset("EleutherAI/lambada_openai", split="test", trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    max_len = getattr(model.config, "max_position_embeddings", 2048)

    for i, ex in enumerate(ds):
        text = ex["text"]
        # Last word = everything after the last space
        last_space = text.rstrip().rfind(" ")
        ctx = text[:last_space]
        target = text[last_space:]  # includes leading space

        ctx_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        tgt_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([ctx_ids, tgt_ids], dim=1)
        if full_ids.shape[1] > max_len:
            full_ids = full_ids[:, -max_len:]

        logits = model(full_ids).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = full_ids[:, 1:]
        logprobs = F.log_softmax(shift_logits.float(), dim=-1)
        # Target tokens are the last tgt_ids.shape[1] of the shifted sequence
        n_tgt = min(tgt_ids.shape[1], shift_labels.shape[1])
        tgt_logprobs = logprobs[:, -n_tgt:, :]
        tgt_labels = shift_labels[:, -n_tgt:]
        # Accuracy: argmax at each target position matches
        pred = tgt_logprobs.argmax(dim=-1)
        if torch.equal(pred, tgt_labels):
            correct += 1
        # NLL for perplexity
        gathered = tgt_logprobs.gather(-1, tgt_labels.unsqueeze(-1)).squeeze(-1)
        total_nll += -gathered.sum().item()
        total_tokens += n_tgt

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  lambada: {i+1}/{len(ds)} | acc={correct/(i+1):.4f} | {(i+1)/elapsed:.1f}/s", flush=True)

    acc = correct / len(ds)
    ppl = math.exp(total_nll / max(1, total_tokens))
    return {"accuracy": acc, "perplexity": ppl, "n": len(ds)}


# ----- Main -----

def main():
    model_id = sys.argv[1]
    out_path = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print(f"=== Loading {model_id} ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {n_params/1e9:.3f}B params in {time.time()-t0:.1f}s", flush=True)

    results = {"model": model_id, "params": n_params, "tasks": {}}

    tasks = [
        ("hellaswag", lambda: load_hellaswag(limit)),
        ("arc_easy", lambda: load_arc("ARC-Easy", limit)),
        ("arc_challenge", lambda: load_arc("ARC-Challenge", limit)),
        ("winogrande", lambda: load_winogrande(limit)),
        ("piqa", lambda: load_piqa(limit)),
        ("boolq", lambda: load_boolq(limit)),
        ("openbookqa", lambda: load_openbookqa(limit)),
    ]
    for name, loader in tasks:
        print(f"\n=== Task: {name} ===", flush=True)
        try:
            items = loader()
            res = mc_eval(model, tok, items, name)
            print(f"  => {name}: accuracy = {res['accuracy']:.4f} ({res['n']} samples)", flush=True)
            results["tasks"][name] = res
        except Exception as e:
            print(f"  FAILED {name}: {e}", flush=True)
            results["tasks"][name] = {"error": str(e)}

    print(f"\n=== Task: lambada_openai ===", flush=True)
    try:
        res = lambada_eval(model, tok, limit)
        print(f"  => lambada: acc={res['accuracy']:.4f}, ppl={res['perplexity']:.2f}", flush=True)
        results["tasks"]["lambada_openai"] = res
    except Exception as e:
        print(f"  FAILED lambada: {e}", flush=True)
        results["tasks"]["lambada_openai"] = {"error": str(e)}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
