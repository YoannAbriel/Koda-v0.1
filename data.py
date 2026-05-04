"""
Data loading for pre-training with SlimPajama (streaming).

SlimPajama (Cerebras) — 627B tokens, already mixed and deduplicated:
  - CommonCrawl (web)      ~67%
  - C4 (web, cleaned)      ~15%
  - GitHub (code)           ~4.5%
  - Wikipedia               ~4.5%
  - Books (Project Gutenberg) ~4.5%
  - ArXiv (papers)          ~2.5%
  - StackExchange (Q&A)     ~2%

Streaming mode: data is downloaded on-the-fly, no 900GB on disk.
We tokenize and chunk into fixed-length sequences as we go,
filling a buffer that the training loop pulls batches from.
"""

import jax.numpy as jnp
import tiktoken
from datasets import load_dataset


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


class StreamingDataLoader:
    """Streams data from SlimPajama, tokenizes on the fly, yields batches.

    How it works:
    1. Stream text from HuggingFace (no full download)
    2. Tokenize each document
    3. Concatenate all tokens into one long stream
       (separated by <|endoftext|>)
    4. Cut into chunks of maxlen
    5. Accumulate chunks into batches of batch_size
    6. Yield each batch as a jnp array

    This is how real LLM training works — no padding waste,
    every token is useful.
    """

    def __init__(self, maxlen, batch_size, split="train", seed=42):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer()
        self.end_token = self.tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

        self.ds = load_dataset(
            "DKYoon/SlimPajama-6B",
            split=split,
            streaming=True,
        ).shuffle(seed=seed, buffer_size=10_000)

    def __iter__(self):
        token_buffer = []
        batch = []

        for example in self.ds:
            text = example["text"]
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            tokens.append(self.end_token)
            token_buffer.extend(tokens)

            # Cut buffer into chunks of maxlen
            while len(token_buffer) >= self.maxlen:
                chunk = token_buffer[: self.maxlen]
                token_buffer = token_buffer[self.maxlen :]
                batch.append(chunk)

                # Yield when we have a full batch
                if len(batch) == self.batch_size:
                    yield jnp.array(batch, dtype=jnp.int32)
                    batch = []


class WikiDataLoader:
    """Fallback: load from Wikipedia (finite dataset, for testing)."""

    def __init__(self, maxlen, batch_size, num_articles=50_000, seed=42):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer()
        self.end_token = self.tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.en",
            split=f"train[:{num_articles}]",
        )
        print(f"Loaded {len(ds):,} articles")

        # Pre-tokenize everything
        self.chunks = []
        for article in ds:
            tokens = self.tokenizer.encode(
                article["text"], allowed_special={"<|endoftext|>"}
            )
            tokens.append(self.end_token)
            for i in range(0, len(tokens), maxlen):
                chunk = tokens[i : i + maxlen]
                if len(chunk) < 64:
                    continue
                chunk.extend([self.end_token] * (maxlen - len(chunk)))
                self.chunks.append(chunk)

        print(f"Created {len(self.chunks):,} chunks")

    def __iter__(self):
        import random
        indices = list(range(len(self.chunks)))
        random.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(self.chunks[idx])
            if len(batch) == self.batch_size:
                yield jnp.array(batch, dtype=jnp.int32)
                batch = []
